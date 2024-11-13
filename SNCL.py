import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from data.augmentor import GraphAugmentor
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from scipy.sparse import eye
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from util.algorithm import sim, randint_choice
from data.social import Relation
from random import random, sample
import numpy as np
from tqdm import tqdm
import json
import scipy.sparse as sp
GPU = torch.cuda.is_available()
device = torch.device('cuda:3' if GPU else "cpu")

class SNCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(SNCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SNCL'])
        self.n_layers = int(args['-n_layer'])
        self.ssl_reg = float(args['-ssl_reg'])
        self.snc_temp = float(args['-snc_temp'])
        self.ui_p_drop = float(args['-ui_p_drop'])
        self.sn_p_drop = float(args['-sn_p_drop'])
        self.mix_ratio = float(args['-mix_ratio'])
        self.uicontrast = str(args['-uicontrast'])
        self.snc_joint = bool(int(args['-snc_joint']))
        self.bisn_joint = bool(int(args["-bisn_joint"]))
        self.snc_enable = bool(int(args['-snc_enable']))
        self.social_data = Relation(conf, kwargs['social.data'], self.data.user)
        self.model = SNCL_Encoder(self.data, self.social_data, self.emb_size, self.n_layers, self.sn_p_drop, self.bisn_joint, self.snc_temp, self.snc_enable, self.uicontrast, self.ui_p_drop, self.mix_ratio).to(device)

    def print_model_info(self):
        super(SNCL, self).print_model_info()
        # # print social relation statistics
        print('Social data size: (user number: %d, relation number: %d).' % (self.social_data.size()))
        print('=' * 80)
    
    def train(self):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1500, 2500], gamma = 0.2)
        for epoch in range(self.maxEpoch):
            # print("[Drop]")
            if self.snc_joint:
                contrast_views = self.model.get_views()
            else:
                contrast_views = self.model.get_views("ui")
            uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
            snv1, snv2 = contrast_views["snv1"], contrast_views["snv2"]
            # print('[Joint Learning]')
            for n, batch in tqdm(enumerate(next_batch_pairwise(self.data, self.batch_size)), total=int(self.data.training_size()[2] / self.batch_size) + 1, mininterval = 1, disable=True):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.ssl_reg * (model.cal_cl_loss([user_idx,pos_idx], uiv1, uiv2) + 0.0001 * model.cal_cl_loss_social(user_idx, snv1, snv2))
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb,neg_item_emb) + cl_loss
                """ batch_loss_main = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                batch_loss_ssl = list()
                items = pos_idx
                if self.uicontrast!="NO":
                    # do SGL:
                        # readout
                    if self.snc_joint:
                        usersv1_ro, itemsv1_ro = model.view_computer_all(uiv1, snv1)
                        usersv2_ro, itemsv2_ro = model.view_computer_all(uiv2, snv2)
                    else:
                        usersv1_ro, itemsv1_ro = model.view_computer_ui(uiv1)
                        usersv2_ro, itemsv2_ro = model.view_computer_ui(uiv2)
                    # from SGL source
                    items_uiv1 = itemsv1_ro[items]
                    items_uiv2 = itemsv2_ro[items]
                    l_item = InfoNCE_overall(items_uiv1, items_uiv2, itemsv2_ro, self.snc_temp)

                    users = user_idx
                    users_uiv1 = usersv1_ro[users]
                    users_uiv2 = usersv2_ro[users]
                    l_user = InfoNCE_overall(users_uiv1, users_uiv2, usersv2_ro, self.snc_temp)
                    # l_user = contrast_model.grace_loss(users_uiv1, users_uiv2)
                    # L = L_main + L_user + L_item + L_sn + R^2
                    batch_loss_ssl.extend([l_user*self.ssl_reg, l_item*self.ssl_reg])
                if batch_loss_ssl:
                    batch_loss_ssl = torch.stack(batch_loss_ssl).sum()
                    batch_loss = batch_loss_main+batch_loss_ssl
                else:
                    batch_loss = batch_loss_main
                # Backward and optimize """
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = model()
            if epoch % 5 == 0:
                self.fast_evaluation(epoch)
            scheduler.step()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        # save embeddings
        with torch.no_grad():
            self.save_embeddings("draws")

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()
    
    def _sorted_dict(self, arr, topk):
        float_list = arr

        # 使用argsort()函数获取排序后的索引
        sorted_indices = np.argsort(float_list)[::-1]

        # 提取排序后的值和对应位置
        sorted_values = float_list[sorted_indices]
        sorted_positions = sorted_indices[:topk]

        # 创建字典
        result_dict = {int(pos): float(val) for pos, val in zip(sorted_positions, sorted_values)}

        return result_dict
    
    def save_embeddings(self, filepath):
        with torch.no_grad():
            scores = {}
            user_embedding_dict = {}
            item_embedding_dict = {}
            for u in range(self.user_emb.size()[0]):
                score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
                scores[u] = self._sorted_dict(score.cpu().numpy(), 20)
                user_embedding_dict[u] = list(self.user_emb[u].cpu().numpy().astype(float))
            for i in range(self.item_emb.size()[0]):
                item_embedding_dict[i] = list(self.item_emb[i].cpu().numpy().astype(float))
            with open(filepath+"\\scores_SNCL.txt", 'w', encoding='utf-8')as f:
                json.dump(scores, f, indent=4)
            with open(filepath+"\\user_SNCL.txt", 'w', encoding='utf-8')as f:
                json.dump(user_embedding_dict, f, indent=4)
            with open(filepath+"\\item_SNCL.txt", 'w', encoding='utf-8')as f:
                json.dump(item_embedding_dict, f, indent=4)
    
    
class SNCL_Encoder(nn.Module):
    def __init__(self, data, social_data, emb_size, n_layers, sn_p_drop, bisn_joint, snc_temp, snc_enable, uicontrast, ui_p_drop, mix_ratio):
        super(SNCL_Encoder, self).__init__()
        self.data = data
        self.social_data = social_data
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.sn_p_drop = sn_p_drop
        self.norm_adj = data.norm_adj
        self.bisn_joint = bisn_joint
        self.snc_enable = snc_enable
        self.snc_temp = snc_temp
        self.uicontrast = uicontrast
        self.ui_p_drop = ui_p_drop
        self.mix_ratio = mix_ratio
        self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)

    def _init_model(self):
        self.num_users = self.data.user_num
        self.num_items = self.data.item_num
        # 初始化嵌入为均匀分布
        initializer = nn.init.xavier_uniform_
        self.embedding_user = nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size)))
        self.embedding_item = nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size)))
        self.bi_social_mat = self.social_data.get_birectional_social_mat()
        self.f = nn.Sigmoid()

    def forward(self, g_droped=None):
        ego_embeddings = torch.cat([self.embedding_user, self.embedding_item], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if g_droped is not None:
                ego_embeddings = torch.sparse.mm(g_droped, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings
        
    def view_computer_all(self, g_droped, sn_auged):
        """
        propagate methods for contrastive lightGCN
        """       
        users_emb = self.social_encoder(self.embedding_user, sn_auged, self.n_layers)
        items_emb = self.embedding_item
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def get_views(self, aug_side="both"):
        if aug_side=="ui" or not self.snc_enable:
            snv1, snv2 = None, None
        else:
            snv1, snv2 = self.get_social_related_views(self.bi_social_mat, self.data.interaction_mat)
        if aug_side=="sn" or self.uicontrast=="NO" or self.uicontrast=="USER-BI":
            uiv1, uiv2 = None, None
        else:
            if self.uicontrast=="WEIGHTED" or self.uicontrast=="WEIGHTED-MIX":
                #sim_social = self.user_sn_stability(snv1).to(device)
                #sim_sharing = self.user_sn_stability(snv2).to(device)
                sims = self.user_sn_stability(snv1, snv2).to(device)
                uiv1 = self.get_ui_views_weighted(sims, 1).to(device)
                uiv2 = self.get_ui_views_weighted(sims, 1).to(device)
            elif self.uicontrast=="RANDOM":
                uig1 = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.ui_p_drop)
                uig2 = GraphAugmentor.edge_dropout(self.data.interaction_mat, self.ui_p_drop)
                uig1 = self.data.convert_to_laplacian_mat(uig1)
                uig2 = self.data.convert_to_laplacian_mat(uig2)
                uiv1 = TorchGraphInterface.convert_sparse_mat_to_tensor(uig1).to(device)
                uiv2 = TorchGraphInterface.convert_sparse_mat_to_tensor(uig2).to(device)

        contrast_views = {
            "snv1":snv1,
            "snv2":snv2,
            "uiv1":uiv1,
            "uiv2":uiv2
        }
        return contrast_views
    
    def user_sn_stability(self, adj1, adj2):
        aug_user_embeddings1 = self.social_encoder(adj1)
        aug_user_embeddings2 = self.social_encoder(adj2)
        sims = sim(aug_user_embeddings1, aug_user_embeddings2)
        return sims

    def get_ui_views_weighted(self, user_stabilities, stab_weight):

        # sn probability of keep
        user_stabilities = torch.exp(user_stabilities)
        sn_weights = (user_stabilities - user_stabilities.min()) / (user_stabilities.max() - user_stabilities.min())
        sn_weights = sn_weights.where(sn_weights > 0.2, torch.ones_like(sn_weights) * 0.2) # 小于0.3的全部变成0.3,以减轻低值效应
        # print(torch.mean(sn_weights))
        # print(torch.min(sn_weights))
        # print(torch.max(sn_weights))

        # overall probability of keep
        weights = (1-self.ui_p_drop)/torch.mean(stab_weight*sn_weights)*(stab_weight*sn_weights)
        # weights = weights.where(weights>0.3, torch.ones_like(weights) * 0.3)
        weights = weights.where(weights<0.95, torch.ones_like(weights) * 0.95) # 大于0.95的全部变成0.95
        # print(torch.mean(weights))
        # print(torch.min(weights))
        # print(torch.max(weights))
        user_mask = torch.bernoulli(weights).to(torch.bool)
        print(f"keep ratio: {user_mask.sum()/user_mask.size()[0]:.2f}")
        # drop
        if self.uicontrast == 'WEIGHTED-EDGE':
            g_weighted = self.ui_drop_weighted(weights)
        else:
            g_weighted = self.ui_drop_weighted(user_mask)
        g_weighted.requires_grad = False
        return g_weighted
    
    def ui_drop_weighted(self, user_mask):
        # user_mask: [user_num]
        user_mask = user_mask.tolist()
        n_nodes = self.num_users + self.num_items
        # [interaction_num]
        user_np = np.array([self.data.user[pair[0]] for pair in self.data.training_data])
        keep_idx = list()
        if self.uicontrast=="WEIGHTED-MIX":
            # overall sample rate = 0.4*0.9 = 0.36
            for i, j in enumerate(user_np):
                if user_mask[j] and random()>0.7:
                    keep_idx.append(i)
            # add random samples
            interaction_random_sample = sample(list(range(len(user_np))), int(len(user_np)*self.mix_ratio))
            keep_idx = list(set(keep_idx+interaction_random_sample))
        elif self.uicontrast=="WEIGHTED":
            for i, j in enumerate(user_np.tolist()):
                if user_mask[j]:
                    keep_idx.append(i)
        
        else:
            for i,j in enumerate(user_np.tolist()):
                if user_mask[j] > random():
                    keep_idx.append(i)

        print(f"finally keep ratio: {len(keep_idx)/len(user_np):.2f}")
        keep_idx = np.array(keep_idx)
        item_np = np.array([self.data.item[pair[1]] for pair in self.data.training_data])[keep_idx]
        user_np = user_np[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # to coo
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce().to(device)
        g.requires_grad = False
        return g

    def social_edge_dropout(self, social_mat, sn_p_drop):
        cols = social_mat.tocoo().col
        rows = social_mat.tocoo().row
        # 计算节点的度
        node_degrees = np.array(social_mat.sum(axis=1)).flatten()
        # 将度差异较大的边赋予更大的权值
        weight_edge = node_degrees[rows] - node_degrees[cols]

        # 获取去除边的数量上限
        max_edges_to_remove = int(sn_p_drop * len(social_mat.data))

        # 选择权值最大的边进行去除
        sorted_indices = np.argsort(weight_edge)[::-1]
        indices_to_remove = sorted_indices[:max_edges_to_remove]

        # 将选择的边的权值设为0
        social_mat.data[indices_to_remove] = 0

        # 返回处理后的图
        return social_mat
    
    def get_social_related_views(self, social_mat, interaction_mat):
        social_mat_drop1 = GraphAugmentor.edge_dropout(social_mat, self.sn_p_drop)
        social_mat_drop2 = GraphAugmentor.edge_dropout(social_mat, self.sn_p_drop)
        social_matrix = social_mat_drop1.dot(social_mat_drop1)
        social_matrix = social_matrix.multiply(social_mat_drop1) + eye(self.data.user_num, dtype=np.float32)
        sharing_matrix = interaction_mat.dot(interaction_mat.T)
        sharing_matrix = sharing_matrix.multiply(social_mat_drop2) + eye(self.data.user_num, dtype=np.float32)
        if self.bisn_joint:
            social_matrix = self.social_data.normalize_graph_mat(social_matrix)
            sharing_matrix = self.social_data.normalize_graph_mat(sharing_matrix)
        else:
            social_matrix = self.social_data.normalize_graph_mat(social_mat_drop1)
            sharing_matrix = self.social_data.normalize_graph_mat(social_mat_drop2)
        
        """ if not self.bisn_joint:
            social_matrix = GraphAugmentor.edge_dropout(social_mat, self.sn_p_drop)
            sharing_matrix = GraphAugmentor.edge_dropout(social_mat, self.sn_p_drop)
            social_matrix = self.social_data.normalize_graph_mat(social_matrix)
            sharing_matrix = self.social_data.normalize_graph_mat(sharing_matrix)
        else:
            social_matrix = social_mat.dot(social_mat)
            social_matrix = social_matrix.multiply(social_mat) + eye(self.data.user_num, dtype=np.float32)
            
            sharing_matrix = interaction_mat.dot(interaction_mat.T)
            sharing_matrix = sharing_matrix.multiply(social_mat) + eye(self.data.user_num, dtype=np.float32)
            
            social_mat_drop1 = GraphAugmentor.edge_dropout(social_matrix, self.sn_p_drop)
            social_mat_drop2 = GraphAugmentor.edge_dropout(sharing_matrix, self.sn_p_drop)
            
            social_matrix = self.social_data.normalize_graph_mat(social_mat_drop1)
            sharing_matrix = self.social_data.normalize_graph_mat(social_mat_drop2) """
        
        social_matrix = TorchGraphInterface.convert_sparse_mat_to_tensor(social_matrix).to(device)
        sharing_matrix = TorchGraphInterface.convert_sparse_mat_to_tensor(sharing_matrix).to(device)
        
        return social_matrix, sharing_matrix


    def view_computer_ui(self, g_droped):
        """
        propagate methods for contrastive lightGCN
        """       
        users_emb = self.embedding_user
        items_emb = self.embedding_item
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def cal_cl_loss(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(device)
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(device)
        user_view_1, item_view_1 = self.forward(perturbed_mat1)
        user_view_2, item_view_2 = self.forward(perturbed_mat2)
        view1 = torch.cat((user_view_1[u_idx],item_view_1[i_idx]),0)
        view2 = torch.cat((user_view_2[u_idx],item_view_2[i_idx]),0)
        # user_cl_loss = InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.temp)
        # item_cl_loss = InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.temp)
        #return user_cl_loss + item_cl_loss
        return InfoNCE(view1, view2, self.snc_temp)
    
    def cal_cl_loss_social(self, idx, perturbed_mat1, perturbed_mat2):
        u_idx = torch.unique(torch.Tensor(idx).type(torch.long)).to(device)
        user_view_1 = self.social_encoder(perturbed_mat1)
        user_view_2 = self.social_encoder(perturbed_mat2)

        view1 = user_view_1[u_idx]
        view2 = user_view_2[u_idx]
        return InfoNCE(view1, view2, self.snc_temp)
    
    def social_encoder(self, social_adj):
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [self.embedding_user]
        for layer in range(self.n_layers):
            emb = embs[layer]
            emb = torch.sparse.mm(social_adj, emb)
            embs.append(emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.sum(embs, dim=1)
        return light_out
    