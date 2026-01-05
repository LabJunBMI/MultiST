
import time
import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .MultiST_module import MultiST_module, MultiST_impute_module, MMD_Generator, MMD_Discriminator, mmd_loss
from tqdm import tqdm


def efficient_fisher_kernel_matrix(X, Y, generator, batch_size=32):

    device = X.device
    
    def compute_fisher_vectors_batch(data, batch_size):
        fisher_vectors = []
        n_samples = data.shape[0]
        
        for i in range(0, n_samples, batch_size):
            batch = data[i:i+batch_size].clone().detach().requires_grad_(True)
            
            gen_output = generator(batch)

            if gen_output.requires_grad:
                grad_input = torch.autograd.grad(
                    gen_output.sum(), batch, 
                    retain_graph=True, create_graph=False
                )[0]
                fisher_vectors.append(grad_input.flatten(1))
            else:

                fisher_vectors.append(torch.zeros(batch.shape[0], batch.shape[1]).to(device))
                
        return torch.cat(fisher_vectors, dim=0)
    

    fisher_X = compute_fisher_vectors_batch(X, batch_size)
    fisher_Y = compute_fisher_vectors_batch(Y, batch_size)
    

    kernel_matrix = torch.mm(fisher_X, fisher_Y.t())
    return kernel_matrix

def efficient_mmd_loss(X, Y, generator, batch_size=32):
    try:
        K_XX = efficient_fisher_kernel_matrix(X, X, generator, batch_size)
        K_YY = efficient_fisher_kernel_matrix(Y, Y, generator, batch_size)
        K_XY = efficient_fisher_kernel_matrix(X, Y, generator, batch_size)
        
        mmd = torch.mean(K_XX) + torch.mean(K_YY) - 2 * torch.mean(K_XY)
        return mmd
    except:
        mean_X = torch.mean(X, dim=0)
        mean_Y = torch.mean(Y, dim=0)
        return torch.mean((mean_X - mean_Y) ** 2)


class MultiST:
    def __init__(
            self,
            X,
            graph_dict,
            rec_w=10,
            gcn_w=0.1,
            self_w=1,
            dec_kl_w=1,
            use_gan=False,          
            gan_w=0.5,           
            mmd_w=0.3,              
            fisher_batch_size=32,    
            mode='clustering',
            device='cuda:0',
    ):
        self.rec_w = rec_w
        self.gcn_w = gcn_w
        self.self_w = self_w
        self.dec_kl_w = dec_kl_w
        
 
        self.use_gan = use_gan
        self.gan_w = gan_w
        self.mmd_w = mmd_w
        self.fisher_batch_size = fisher_batch_size
        self.gan_update_interval = 5
        
        self.device = device
        self.mode = mode

        if 'mask' in graph_dict:
            self.mask = True
            self.adj_mask = graph_dict['mask'].to(self.device)
        else:
            self.adj_mask = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0),
                size=(len(X), len(X))
            ).to(self.device)
            
        self.cell_num = len(X)
        self.X = torch.FloatTensor(X.copy()).to(self.device)
        self.input_dim = self.X.shape[1]
        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.adj_label = graph_dict["adj_label"].to(self.device)
        self.norm_value = graph_dict["norm_value"]
       
        if self.mode == 'clustering':
            self.model = MultiST_module(self.input_dim).to(self.device)
        elif self.mode == 'imputation':
            self.model = MultiST_impute_module(self.input_dim).to(self.device)
        else:
            raise ValueError(f'{self.mode} is not currently supported!')

 
        if self.use_gan:
            self._init_gan_components()

    def _init_gan_components(self):
        self.generator = MMD_Generator(self.input_dim, self.input_dim).to(self.device)
        self.discriminator = MMD_Discriminator(self.input_dim).to(self.device)
        self.optimizer_g = torch.optim.RMSprop(self.generator.parameters(), lr=0.0001)
        self.optimizer_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.0001)
        self.clipping_param = 0.01

    def generate_noise_data(self):
        return torch.randn((self.cell_num, self.input_dim)).to(self.device)

    def update_gan_components(self, real_data, epoch):
        if not self.use_gan or epoch % self.gan_update_interval != 0:
            return 0.0, 0.0
            
        noise_data = self.generate_noise_data()
        
        try:

            self.optimizer_d.zero_grad()
            real_output = self.discriminator(real_data)
            
            with torch.no_grad():
                fake_data = self.generator(noise_data)
            fake_output = self.discriminator(fake_data)
            

            mmd_d_loss = efficient_mmd_loss(real_data, fake_data, self.generator, self.fisher_batch_size)
            
            d_loss = -torch.mean(real_output) + torch.mean(fake_output) + self.mmd_w * mmd_d_loss
            d_loss.backward()
            self.optimizer_d.step()
            

            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clipping_param, self.clipping_param)

            self.optimizer_g.zero_grad()
            fake_data_grad = self.generator(noise_data)
            fake_output_grad = self.discriminator(fake_data_grad)
            
            mmd_g_loss = efficient_mmd_loss(real_data, fake_data_grad, self.generator, self.fisher_batch_size)
            g_loss = -torch.mean(fake_output_grad) + self.mmd_w * mmd_g_loss
            g_loss.backward()
            self.optimizer_g.step()
            
            return d_loss.item(), g_loss.item()
            
        except Exception as e:
            print(f"GAN update failed: {e}")
            return 0.0, 0.0

    def train_without_dec(self, epochs=200, lr=0.01, decay=0.01, N=1):
      
        self.optimizer = torch.optim.Adam(params=list(self.model.parameters()), lr=lr, weight_decay=decay)
        self.model.train()
        
        print(f" (GAN: {self.use_gan})...")
        
        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            
            latent_z, mu, logvar, de_feat, q, feat_x, gnn_z, loss_self = self.model(self.X, self.adj_norm)


            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask),
                labels=self.adj_mask.coalesce().values(),
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )
            loss_rec = reconstruction_loss(de_feat, self.X)

            gan_loss = 0.0
            if self.use_gan and epoch >= 10:
                try:
                    noise_data = self.generate_noise_data()
                    fake_data = self.generator(noise_data)
                    fake_latent_z, _, _, fake_de_feat, _, _, _, _ = self.model(fake_data, self.adj_norm)
                    
                    mmd_latent_loss = efficient_mmd_loss(
                        latent_z.detach(), fake_latent_z, self.generator, self.fisher_batch_size
                    )
                    gan_loss = self.gan_w * mmd_latent_loss
                except Exception as e:
                    print(f"GAN loss calculation failed: {e}")
                    gan_loss = 0.0

            total_loss = (self.rec_w * loss_rec + 
                         self.gcn_w * loss_gcn + 
                         self.self_w * loss_self + 
                         gan_loss)
            
            total_loss.backward()
            self.optimizer.step()
            
      
            d_loss, g_loss = self.update_gan_components(self.X, epoch)
            
     
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Total={total_loss.item():.4f}, "
                      f"Rec={loss_rec.item():.4f}, GCN={loss_gcn.item():.4f}, "
                      f"Self={loss_self.item():.4f}, GAN={gan_loss:.4f}")
                if self.use_gan:
                    print(f"  D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")

    def train_with_dec(self, epochs=200, preepoch=200, dec_interval=20, dec_tol=0.00, N=1):
      
        self.train_without_dec(epochs=preepoch)
        
        from sklearn.cluster import KMeans
        
       
        test_z, _, _, _ = self.process()
        kmeans = KMeans(n_clusters=self.model.dec_cluster_n, n_init=self.model.dec_cluster_n * 2, random_state=42)
        # kmeans = KMeans(n_clusters=self.model.dec_cluster_n, init='k-means++', random_state=42)
        y_pred_last = np.copy(kmeans.fit_predict(test_z))
        
        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()
        
        print(f"DEC (GAN: {self.use_gan})...")

        for epoch_id in tqdm(range(epochs)):
            if epoch_id % dec_interval == 0:
                _, tmp_q, _, _ = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                
                if epoch_id > 0 and delta_label < dec_tol:
                    print(f'delta_label {delta_label:.4f} < tol {dec_tol}')
                    print('Reached tolerance threshold. Stopping training.')
                    break

            self.optimizer.zero_grad()
            latent_z, mu, logvar, de_feat, out_q, feat_x, gnn_z, loss_self = self.model(self.X, self.adj_norm)
            
         
            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask),
                labels=self.adj_mask.coalesce().values(),
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )
            loss_rec = reconstruction_loss(de_feat, self.X)
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device))

            gan_loss = 0.0
            if self.use_gan:
                try:
                    noise_data = self.generate_noise_data()
                    fake_data = self.generator(noise_data)
                    fake_latent_z, _, _, _, fake_q, _, _, _ = self.model(fake_data, self.adj_norm)
                    
                  
                    cluster_consistency_loss = F.kl_div(fake_q.log(), out_q.detach())
                    
                 
                    mmd_cluster_loss = efficient_mmd_loss(
                        latent_z.detach(), fake_latent_z, self.generator, self.fisher_batch_size
                    )
                    
                    gan_loss = self.gan_w * (cluster_consistency_loss + mmd_cluster_loss)
                except Exception as e:
                    print(f"GAN loss calculation failed: {e}")
                    gan_loss = 0.0
            
   
            total_loss = (self.gcn_w * loss_gcn + 
                         self.dec_kl_w * loss_kl + 
                         self.rec_w * loss_rec + 
                         gan_loss)
            
            total_loss.backward()
            self.optimizer.step()
            
       
            d_loss, g_loss = self.update_gan_components(self.X, epoch_id)
            
            if epoch_id % 20 == 0:
                print(f"DEC Epoch {epoch_id}: Total={total_loss.item():.4f}, "
                      f"KL={loss_kl.item():.4f}, Rec={loss_rec.item():.4f}, GAN={gan_loss:.4f}")


    def save_model(self, save_model_file):
        save_dict = {'state_dict': self.model.state_dict()}
        if self.use_gan:
            save_dict['generator_state_dict'] = self.generator.state_dict()
            save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
        torch.save(save_dict, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        if self.use_gan and 'generator_state_dict' in saved_state_dict:
            if not hasattr(self, 'generator'):
                self._init_gan_components()
            self.generator.load_state_dict(saved_state_dict['generator_state_dict'])
            self.discriminator.load_state_dict(saved_state_dict['discriminator_state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self):
        self.model.eval()
        latent_z, _, _, _, q, feat_x, gnn_z, _ = self.model(self.X, self.adj_norm)
        return latent_z.data.cpu().numpy(), q.data.cpu().numpy(), feat_x.data.cpu().numpy(), gnn_z.data.cpu().numpy()

    def recon(self):
        self.model.eval()
        latent_z, _, _, de_feat, q, feat_x, gnn_z, _ = self.model(self.X, self.adj_norm)
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(de_feat.data.cpu().numpy())


def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()

def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn

def gcn_loss(preds, labels, mu, logvar, n_nodes, norm):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD