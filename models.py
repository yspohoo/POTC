import pickle
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from utils_edited import *
from variable import *
import ot
from torch.distributions.normal import Normal
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix



def initialize_parameters(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


class POTC(nn.Module):
    def __init__(self, sencoder, tencoder, cluster_size, lambda_e, maxiter, rho,
                 overlap_src2tgt, device='cpu', skip_ot=False):
        super().__init__()
        self.device = device
        self.cluster_size = cluster_size
        self.lambda_e = lambda_e
        self.maxiter = maxiter
        self.skip_ot = skip_ot
        self.sencoder = sencoder
        self.tencoder = tencoder
        self.rho = rho
        self.overlap_src2tgt = overlap_src2tgt
        for p in self.sencoder.parameters():
            p.requires_grad = False
        for p in self.tencoder.parameters():
            p.requires_grad = False

        # Frozen user representations and domain-specific prototypes.
        su_all, _ = self.sencoder.computer()
        tu_all, _ = self.tencoder.computer()
        self.skmeans, self.sp = prepare(su_all, cluster_size, device)
        _, self.tp = prepare(tu_all, cluster_size, device)

        if self.skip_ot and cluster_size <= 0:
            raise ValueError("skip_ot=True requires cluster_size > 0.")

        if cluster_size > 0 and len(self.overlap_src2tgt) > 0:
            overlap_src_ids = list(self.overlap_src2tgt.keys())
            overlap_tgt_ids = [self.overlap_src2tgt[i] for i in overlap_src_ids]
            su_overlap = su_all[overlap_src_ids].detach()
            tu_overlap = tu_all[overlap_tgt_ids].detach()
            p_s = torch.softmax(su_overlap @ self.sp.t(), dim=1)
            p_t = torch.softmax(tu_overlap @ self.tp.t(), dim=1)
            association = p_s.t() @ p_t
        else:
            k = max(cluster_size, 0)
            association = torch.zeros(k, k, device=device)

        if cluster_size > 0:
            row_max = association.amax(dim=1, keepdim=True)
            A_tilde = association / row_max.clamp_min(1e-8)
            row_sum = A_tilde.sum(dim=1, keepdim=True)
            uniform = torch.full_like(A_tilde, 1.0 / cluster_size)
            association_weights = torch.where(
                row_sum > 1e-8,
                A_tilde / row_sum.clamp_min(1e-8),
                uniform,
            )
            association_destination = association_weights @ self.tp
        else:
            A_tilde = association
            association_weights = A_tilde
            association_destination = self.tp

        self.register_buffer('A_tilde', A_tilde)
        self.register_buffer('association_weights', association_weights)
        self.register_buffer('association_destination', association_destination)
        self.saved_transport_plan = None
        self.saved_prototype_destination = None

        self.cluster_map = nn.Sequential(
            nn.Linear(sencoder.embed_dim, sencoder.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(sencoder.embed_dim // 2, sencoder.embed_dim)
        )
        self.cluster_map.apply(initialize_parameters)
        # Fuse user, source prototype, projected prototype, and target prompt.
        self.mlp = MLP(input_size=sencoder.embed_dim * 4,
                       output_size=sencoder.embed_dim,
                       hidden_size=sencoder.embed_dim // 2)

    def _align_prototypes(self, mappedsp):
        """Build prototype prompts and alignment loss, optionally using OT."""
        if self.skip_ot:
            ot_loss = mappedsp.new_zeros(())
            cost = ot.dist(mappedsp, self.tp, metric='cosine')
            cost = cost + self.rho * (1.0 - self.A_tilde)
            if self.lambda_e <= 0:
                raise ValueError("Softmax correspondence requires positive temperature")
            probabilities = torch.softmax(-cost / self.lambda_e, dim=1)
            destinations = probabilities @ self.tp
            self.saved_transport_plan = None
        else:
            cost_bias = self.rho * (1.0 - self.A_tilde)
            ot_loss, plan = WDLOSS(
                mappedsp,
                self.tp,
                self.lambda_e,
                self.maxiter,
                self.device,
                cost_bias=cost_bias,
            )
            transport_probs = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-8)
            destinations = transport_probs @ self.tp
            self.saved_transport_plan = plan.detach()

        align_loss = (
            1.0
            - F.cosine_similarity(
                mappedsp,
                destinations.detach(),
                dim=1,
                eps=1e-8,
            )
        ).mean()

        self.saved_prototype_destination = destinations.detach()
        return ot_loss, align_loss, destinations

    def forward(self, su_idx, tu_idx):
        su_all, _ = self.sencoder.computer()
        tu_all, ti_all = self.tencoder.computer()

        seu = su_all[su_idx]
        seup, seup_labels = assign_prototype(self.skmeans, seu, self.device)

        if self.cluster_size > 0:
            mappedsp = self.cluster_map(self.sp)
            ot_loss, align_loss, destinations = self._align_prototypes(mappedsp)
            sdestination = destinations[seup_labels]
            mappedsp_input = mappedsp[seup_labels]
        else:
            ot_loss = seu.new_zeros(())
            align_loss = seu.new_zeros(())
            sdestination = torch.zeros_like(seup)
            mappedsp_input = torch.zeros_like(seup)
            self.saved_transport_plan = None
            self.saved_prototype_destination = None

        mapped_source = self.mlp(
            torch.cat([seu, seup, sdestination, mappedsp_input], dim=-1)
        )
        return mapped_source, (ot_loss, align_loss), ti_all



class LightGCN(nn.Module):
    """
    LightGCN recommendation model (Wang et al., 2020).

    `dataset` is a utils.Loader instance: it owns the user-item interaction
    graph (built from train.txt) and is responsible for producing the
    normalized sparse adjacency tensor via dataset.getSparseGraph(device).
    """
    def __init__(self, dataset, latent_dim, n_layers, device):
        super(LightGCN, self).__init__()
        self.dataset = dataset
        self.num_users  = dataset.n_user
        self.num_items  = dataset.m_item
        self.latent_dim = latent_dim
        self.embed_dim = latent_dim  # alias for POTC's encoder.embed_dim usage
        self.n_layers = n_layers
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        initialize_parameters(self.embedding_user)
        initialize_parameters(self.embedding_item)
        self.f = nn.Sigmoid()
        self.Graph = dataset.getSparseGraph(device)

    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
    
    # original forward method
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1.apply(initialize_parameters)
        self.fc2.apply(initialize_parameters)
        # self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.fc2(out)
        # out = self.soft(out)
        return out

    
class MF(nn.Module):
    def __init__(self, n_users, m_items, latent_dim):
        super(MF, self).__init__()
        self.num_users  = n_users
        self.num_items  = m_items
        self.latent_dim = latent_dim
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.f = nn.Sigmoid()
        self.apply(initialize_parameters)

    def forward(self, users, items):
        users_emb = self.embedding_user.weight[users.long()]
        items_emb = self.embedding_item.weight[items.long()]
        rating = self.f(torch.sum(users_emb * items_emb, dim=1, keepdim=True))
        # rating = torch.sum(users_emb * items_emb, dim=1, keepdim=True)
        return rating
    

class EMCDR(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Sigmoid(),
            nn.Linear(input_dim * 2, input_dim)
        )
        self.fc.apply(initialize_parameters)
    def forward(self, x):
        return self.fc(x)
