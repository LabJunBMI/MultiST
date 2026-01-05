
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from functools import partial


def sce_loss(x, y, alpha=3):
    """Self-construction error loss."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


def mmd_loss_jingxiangkernel(p, q, sigma=1.0):
    """Maximum Mean Discrepancy (MMD) loss for MMD-GAN."""
    xx = torch.matmul(p, p.t())
    yy = torch.matmul(q, q.t())
    xy = torch.matmul(p, q.t())
    rx = xx.diagonal().unsqueeze(0).expand_as(xx)
    ry = yy.diagonal().unsqueeze(0).expand_as(yy)

    K = torch.exp(-(rx.t() + rx - 2 * xx) / (2 * sigma**2))
    L = torch.exp(-(ry.t() + ry - 2 * yy) / (2 * sigma**2))
    P = torch.exp(-(rx.t() + ry - 2 * xy) / (2 * sigma**2))

    return K.mean() + L.mean() - 2 * P.mean()

def fisher_kernel_matrix_gan(X, Y, generator):
    """
    Compute Fisher Kernel matrix using GAN generator.
    
    Args:
        X (torch.Tensor): Sample set X with shape (n_samples_X, n_features).
        Y (torch.Tensor): Sample set Y with shape (n_samples_Y, n_features).
        generator (torch.nn.Module): GAN generator model.
        
    Returns:
        torch.Tensor: Fisher Kernel matrix (n_samples_X, n_samples_Y).
    """
    fisher_vectors_X = []
    fisher_vectors_Y = []

    # Compute Fisher vectors for X
    for x in X:
        x = x.unsqueeze(0).requires_grad_(True)  # Ensure gradients can flow
        gen_x = generator(x)
        grad_x = torch.autograd.grad(gen_x, generator.parameters(), retain_graph=True, create_graph=True)
        fisher_vectors_X.append(torch.cat([g.flatten() for g in grad_x]))

    # Compute Fisher vectors for Y
    for y in Y:
        y = y.unsqueeze(0).requires_grad_(True)
        gen_y = generator(y)
        grad_y = torch.autograd.grad(gen_y, generator.parameters(), retain_graph=True, create_graph=True)
        fisher_vectors_Y.append(torch.cat([g.flatten() for g in grad_y]))

    fisher_vectors_X = torch.stack(fisher_vectors_X)
    fisher_vectors_Y = torch.stack(fisher_vectors_Y)

    # Compute kernel matrix
    fisher_kernel = torch.mm(fisher_vectors_X, fisher_vectors_Y.t())
    return fisher_kernel

def mmd_loss(X, Y, generator):
    """
    Compute MMD loss using Fisher Kernel with GAN generator.
    
    Args:
        X (torch.Tensor): Real samples.
        Y (torch.Tensor): Generated samples.
        generator (torch.nn.Module): GAN generator model.
        
    Returns:
        torch.Tensor: MMD loss.
    """
    K_XX = fisher_kernel_matrix_gan(X, X, generator)
    K_YY = fisher_kernel_matrix_gan(Y, Y, generator)
    K_XY = fisher_kernel_matrix_gan(X, Y, generator)

    # Compute MMD
    mmd = torch.mean(K_XX) + torch.mean(K_YY) - 2 * torch.mean(K_XY)
    return mmd


def full_block(in_features, out_features, p_drop):
    """A fully connected block with BatchNorm, ELU activation, and Dropout."""
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class GraphConvolution(nn.Module):
    """Graph Convolutional Layer."""
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return self.act(output)

from torch_geometric.nn import GATConv
class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super(GATEncoder, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, mask):
        if mask.layout != torch.sparse_coo:
            mask = mask.to_sparse()
        col = mask.coalesce().indices()[0]
        row = mask.coalesce().indices()[1]
        result = self.act(torch.sum(z[col] * z[row], axis=1))
        return result


class MMD_Generator(nn.Module):
    """Generator for MMD-GAN."""
    def __init__(self, input_dim, output_dim):
        super(MMD_Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.fc(z)


class MMD_Discriminator(nn.Module):
    """Discriminator for MMD-GAN."""
    def __init__(self, input_dim):
        super(MMD_Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)


class MultiST_module(nn.Module):
    """MultiST core module."""
    def __init__(
            self,
            input_dim,
            feat_hidden1=64,
            feat_hidden2=16,
            gcn_hidden1=64,
            gcn_hidden2=16,
            p_drop=0.2,
            alpha=1.0,
            dec_cluster_n=10,
    ):
        super(MultiST_module, self).__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_cluster_n
        self.latent_dim = self.gcn_hidden2 + self.feat_hidden2

        # Feature autoencoder
        self.encoder = nn.Sequential(
            full_block(self.input_dim, self.feat_hidden1, self.p_drop),
            full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop)
        )
        self.decoder = GraphConvolution(self.latent_dim, self.input_dim, self.p_drop, act=lambda x: x)

        # GCN layers
        self.gc1 = GraphConvolution(self.feat_hidden2, self.gcn_hidden1, self.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(self.p_drop, act=lambda x: x)

        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.latent_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # Masking token
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self._mask_rate = 0.8
        self.criterion = self.setup_loss_fn(loss_fn='sce')

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        """Set up the loss function."""
        if loss_fn == "mse":
            return nn.MSELoss()
        elif loss_fn == "sce":
            return partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError

    # def encode(self, x, adj):
    #     """Encoder forward pass."""
    #     feat_x = self.encoder(x)
    #     hidden1 = self.gc1(feat_x, adj)
    #     return self.gc2(hidden1, adj), feat_x
    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        mu = self.gc2(hidden1, adj)
        logvar = torch.zeros_like(mu)  
        return mu, logvar, feat_x



    def forward(self, x, adj):
        mu, logvar, feat_x = self.encode(x, adj)
        z = torch.cat((feat_x, mu), 1)
        de_feat = self.decoder(z, adj)

        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()


        loss = self.criterion(de_feat, x)


        return z, mu, logvar, de_feat, q, feat_x, mu, loss
        # return z, mu, de_feat, q, feat_x, loss

    # def forward(self, x, adj):
    #     adj, x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj, x, self._mask_rate)

    #     mu, feat_x = self.encode(x, adj)
    #     z = torch.cat((feat_x, mu), 1)
    #     de_feat = self.decoder(z, adj)

    #     # DEC clustering
    #     q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
    #     q = q.pow((self.alpha + 1.0) / 2.0)
    #     q = (q.t() / torch.sum(q, 1)).t()

    #     # Self-construction loss
    #     x_init = x[mask_nodes]
    #     x_rec = de_feat[mask_nodes]
    #     loss = self.criterion(x_rec, x_init)

    #     return z, mu, de_feat, q, feat_x, loss

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        """Apply masking noise to the input data."""
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[token_nodes] += self.enc_mask_token
        return adj.clone(), out_x, (mask_nodes, keep_nodes)


class MultiST_impute_module(nn.Module):
    """MultiST module for imputation."""
    def __init__(
            self,
            input_dim,
            feat_hidden1=64,
            feat_hidden2=16,
            gcn_hidden1=64,
            gcn_hidden2=16,
            p_drop=0.2,
            alpha=1.0,
            dec_cluster_n=10,
    ):
        super(MultiST_impute_module, self).__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_cluster_n
        self.latent_dim = self.gcn_hidden2 + self.feat_hidden2

        # Feature autoencoder
        self.encoder = nn.Sequential(
            full_block(self.input_dim, self.feat_hidden1, self.p_drop),
            full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop)
        )
        self.decoder = nn.Sequential(
            full_block(self.latent_dim, self.input_dim, self.p_drop)
        )

        # GCN layers
        self.gc1 = GraphConvolution(self.feat_hidden2, self.gcn_hidden1, self.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(self.p_drop, act=lambda x: x)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.latent_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # Masking token
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self._mask_rate = 0.8
        self.criterion = self.setup_loss_fn(loss_fn='sce')

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        """Set up the loss function."""
        if loss_fn == "mse":
            return nn.MSELoss()
        elif loss_fn == "sce":
            return partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError


    # def encode(self, x, adj):
    #     """Encoder forward pass."""
    #     feat_x = self.encoder(x)
        
    #     hidden1 = self.gc1(feat_x, adj)
    #     return self.gc2(hidden1, adj), feat_x
    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        mu = self.gc2(hidden1, adj)
        logvar = torch.zeros_like(mu)  # 占位值
        return mu, logvar, feat_x

    def forward(self, x, adj):
        """Forward pass for imputation."""
        adj, x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj, x, self._mask_rate)

        mu, logvar, feat_x = self.encode(x, adj)
        z = torch.cat((feat_x, mu), 1)
        de_feat = self.decoder(z)

        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        # Self-construction loss
        x_init = x[mask_nodes]
        x_rec = de_feat[mask_nodes]
        loss = self.criterion(x_rec, x_init)
        return z, mu, logvar, de_feat, q, feat_x, mu, loss
        # return z, mu, de_feat, q, feat_x, loss

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        """Apply masking noise to the input data."""
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[token_nodes] += self.enc_mask_token
        return adj.clone(), out_x, (mask_nodes, keep_nodes)

