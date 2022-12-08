import numpy as np
import math
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import random
import time


'''https://github.com/yumaloop/dnn_hsic/blob/master/calc_hsic.ipynb'''
'''
class RBFkernel():
    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, x, y):
        numerator = -1 * np.linalg.norm(x - y, ord=2) ** 2
        denominator = 2 * (self.sigma ** 2)
        return np.exp(numerator / denominator)

    def get_params(self):
        return self.sigma

    def set_params(self, sigma):
        self.sigma = sigma


def gram_matrix(kernel, data, m):
    """
    Arguments:
    =========
    - kernel : kernel function
    - data : data samples, shape=(m, dim(data_i))
    - m : number of samples
    """
    gram_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            gram_matrix[i][j] = kernel(data[i], data[j])

    return gram_matrix


def hsic(k, l, m, X, Y):
    """
    Arguments:
    =========
    - k : kernel function for X
    - l : kernel function for Y
    - m : number of samples
    - X : data samples, shape=(m, dim(X_i))
    - Y : data samples, shape=(m, dim(Y_i))
    """
    H = np.full((m, m), -(1 / m)) + np.eye(m)
    K = gram_matrix(k, X, m)
    L = gram_matrix(l, Y, m)
    HSIC = np.trace(np.dot(K, np.dot(H, np.dot(L, H)))) / ((m - 1) ** 2)
    return HSIC
'''

'''https://github.com/Gaelic98/HSIC/blob/master/HSIC%20code.ipynb'''
def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    Q = I - unit / n

    return np.dot(np.dot(Q, K), Q)


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / sigma / sigma
    np.exp(KX, KX)
    return KX


def HSIC(X, Y):
    return np.sum(centering(rbf(X)) * centering(rbf(Y)))


def Embedding_dist(model, data, args, use_intervene=False):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    feat_i, adj_i, diff_i, num_nodes_i = data
    feat_i = feat_i.cuda()
    adj_i = adj_i.cuda()
    diff_i = diff_i.cuda()
    ft_size = feat_i.shape[1]

    intervene_mask = torch.zeros(feat_i.shape).to(device)
    int_mask = []
    if use_intervene:
        int_mask = torch.zeros(num_nodes_i).to(device)
        int_node = random.sample(range(num_nodes_i), int(num_nodes_i*0.3))
        int_mask[int_node] = 1
        # int_mask = torch.rand(num_nodes_i, device=device) > 0.8
        int_mask = int_mask.int()
        intervene_mask[0:num_nodes_i] = int_mask.repeat([ft_size, 1]).T


    Embeddings = []
    for n_aug in range(args.aug_num):
        attr_mask = torch.randn(feat_i.shape, device=device) / 5 + 1
        # attr_mask = torch.rand(data.x.shape, device=device) > 0.1
        feat_aug = feat_i.clone().detach()
        feat_aug = feat_aug * attr_mask * (1 - intervene_mask) + feat_aug * intervene_mask
        embeds, _, _, _ = model(adj_i.unsqueeze(dim=0), diff_i.unsqueeze(dim=0),
                                   feat_aug.unsqueeze(dim=0), num_nodes_i)
        Embeddings.append(embeds.squeeze())
    Embeddings = torch.stack(Embeddings, dim=2)
    return Embeddings, int_mask


'''
def Structure_pruning(edge_idx, sub_structure, intervene_idx, score_mat):
    sub_structure_adj = to_dense_adj(sub_structure, max_num_nodes=intervene_idx.size()[0]).squeeze()
    interested_edge = []
    for int_node in intervene_idx.nonzero():
        neighbors = edge_idx[1, edge_idx[0] == int_node]
        if neighbors.size()[0] > 0:
            for neighbor in neighbors:
                interested_edge.append([int_node[0], neighbor])
            
            if neighbors.size()[0] > 1:
                for i in range(neighbors.size()[0]):
                    for j in range(i + 1, neighbors.size()[0]):
                        interested_edge.append([neighbors[i], neighbors[j]])
                        interested_edge.append([neighbors[j], neighbors[i]])
            
    evaluate_vec = []
    for edge in interested_edge:
        evaluate_vec.append(score_mat[edge[0], edge[1]])
    print(score_mat[sub_structure_adj==0].mean(), score_mat[sub_structure_adj!=0].mean(),
          torch.tensor(evaluate_vec).mean(), score_mat[sub_structure_adj==0].std(),
          score_mat[sub_structure_adj!=0].std(), torch.tensor(evaluate_vec).std())
    # return 0
'''


def Learn_structure(model, data, structure, args):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    feat, adj, diff, _, num_nodes = data
    feat = feat.cuda()
    adj = adj.cuda()
    n_graph, max_nodes, ft_size = feat.shape

    model.zero_grad()
    start_time = time.time()
    for idx in range(n_graph):
        data_i = feat[idx], structure[idx], diff[idx], num_nodes[idx]

        Embeddings, _ = Embedding_dist(model, data_i, args)
        Embedding_intervened, Intervene_idx = Embedding_dist(model, data_i, use_intervene=True, args=args)
        int_nodes = Intervene_idx.nonzero().squeeze()

        # triangle link
        int_link = torch.zeros(adj[idx].shape).to(device)
        for int_node in int_nodes:
            neighbors = structure[idx][int_node].nonzero()
            torch.save(neighbors, 'neighbors.pt')
            if neighbors.shape[0] > 2:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if neighbors[i] != int_node and neighbors[j] != int_node:
                            # int_link.append(torch.Tensor([neighbors[i], neighbors[j]]).to(device))
                            int_link[neighbors[i], neighbors[j]] = 1
                            int_link[neighbors[j], neighbors[i]] = 1

        neg_link = ((structure[1] == 0) & (int_link == 0)).nonzero()
        int_link = int_link.nonzero()
        neg_link = random.sample(neg_link.tolist(), len(int_link))
        neg_score = []
        for link in neg_link:
            neg_score.append(HSIC(Embeddings[link[0]].cpu().detach().numpy(),
                                  Embedding_intervened[link[1]].cpu().detach().numpy()))
        neg_mean, neg_std = np.mean(neg_score), np.std(neg_score)
        pos_score = []
        for link in int_link:
            p_score = HSIC(Embeddings[link[0]].cpu().detach().numpy(),
                           Embedding_intervened[link[1]].cpu().detach().numpy())
            if p_score < neg_mean + neg_std:
                structure[idx, link[0], int_node] = 0.5
                structure[idx, link[1], int_node] = 0.5
                structure[idx, int_node, link[0]] = 0
                structure[idx, int_node, link[0]] = 0
            pos_score.append(p_score)

        # parent link
        int_link = torch.zeros(adj[idx].shape).to(device)
        for int_node in int_nodes:
            neighbors = structure[idx][int_node].nonzero()
            for neighbor in neighbors:
                if neighbor != int_node:
                    int_link[int_node, neighbor] = 1
        int_link = int_link.nonzero()
        orig_score = []
        for link in int_link:
            orig_score.append(HSIC(Embeddings[link[0]].cpu().detach().numpy(),
                                   Embeddings[link[1]].cpu().detach().numpy()))
        orig_mean, orig_std = np.mean(orig_score), np.std(orig_score)
        for link in int_link:
            int_score = HSIC(Embeddings[link[0]].cpu().detach().numpy(),
                             Embedding_intervened[link[1]].cpu().detach().numpy())
            if int_score > orig_mean + orig_std:
                structure[idx, link[0], link[1]] /= 2



        '''
        kernel_1 = RBFkernel(sigma=0.2)
        kernel_2 = RBFkernel(sigma=0.2)
        hisc_value = hsic(kernel_1, kernel_2, args.aug_num,
                          Embeddings[:, node_idx + i , :].cpu().squeeze().detach().numpy(),
                          Embedding_intervened[:, node_idx + j :].cpu().squeeze().detach().numpy())
        '''
        # print(f"#{idx}'s graph")
    end_time = time.time()
    print(f"Structure learning time: {end_time - start_time}")
    return structure


