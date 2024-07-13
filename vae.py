import torch
from torch import nn
from typing import Dict
from einops import rearrange

from models import Encoder, Decoder 


def guaasian_kl_div_loss(mean, log_var):
    """cal kl div of p(x) and q(x), p(x) is the model's output
    q(x) ~ N(0, 1),
    the `res = - log_std - 0.5 + (\mu ** 2  + \std ** 2) / 2 `"""
    return torch.mean(- 0.5 * log_var - 0.5 + (mean ** 2 + torch.exp(log_var)) * 0.5)


class VAE(nn.Module):
    def __init__(self, 
                 encoder_config: Dict,
                 laten_dim: int,
                 decoder_config: Dict,
                 img_size: int,
                 device,
                 reg_loss_w: float
                 ) -> None:
        super().__init__()
        self.img_size = img_size
        self.device = device
        self.reg_loss_w = reg_loss_w
        self.laten_dim = laten_dim

        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        laten_w = int(img_size / 2**(len(encoder_config['channels_lst']) - 1))
        self.laten_w = laten_w

        self.reconstruct_loss_fn = nn.MSELoss()
        self.reg_loss = guaasian_kl_div_loss
        
        
        self.to_laten = nn.Conv2d(encoder_config['channels_lst'][-1], laten_dim, 1, 1, 0)
        self.to_mean_and_log_var = nn.Linear(laten_dim * laten_w * laten_w, laten_dim * 2)
        self.noise_to_dec_inp = nn.Linear(laten_dim, laten_dim * laten_w * laten_w)
        self.to_dec_inp = nn.Conv2d(laten_dim, decoder_config['channels_lst'][0], 1, 1, 0)
        
        
    def train_loss(self, img):
        img_laten = self.to_laten(self.encoder(img)) # (b, laten_dim, laten_w, laten_w)
        b, c, h, w = img_laten.shape
        img_laten = rearrange(img_laten, 'b c h w -> b (c h w)') # (b, laten_dim*laten_w*laten_w)
        mean_log_var = self.to_mean_and_log_var(img_laten)  # (b, laten_dim * 2)
        mean, log_var = torch.chunk(mean_log_var, chunks=2, dim=1)
        # var always > 0, so log var can be negative or postive
        noise = torch.randn_like(mean, device=img.device)
        
        std = torch.exp(log_var * 0.5)
        sampled_noise = mean +  std * noise # (b, laten_dim)
        
        reconstruct_inp = self.noise_to_dec_inp(sampled_noise) # (b, laten_dim*laten_w*laten_w)
        reconstruct_inp = rearrange(reconstruct_inp, 'b (c h w) -> b c h w', c=c, h=h, w=w) # (b, laten_dim, laten_w, laten_w)
        reconstruct_inp = self.to_dec_inp(reconstruct_inp)
        reconstructed = self.decoder(reconstruct_inp)
        
        reconst_loss = self.reconstruct_loss_fn(reconstructed, img)
        reg_loss = self.reg_loss(mean, log_var)
        loss = reconst_loss + self.reg_loss_w * reg_loss
        return dict(reconst_loss=reconst_loss, 
                    reg_loss=reg_loss,
                    loss=loss)

    
    @torch.no_grad()
    def sample_img(self, b_s, device='cpu'):
        self.eval()
        noise = torch.randn([b_s] + [self.laten_dim]).to(device)
        sampled = self.noise_to_dec_inp(noise) # (b, laten_dim*laten_w*laten_w)
        sampled = rearrange(sampled, 
                            'b (c h w) -> b c h w', 
                            c=self.laten_dim, 
                            h=self.laten_w, 
                            w=self.laten_w) # (b, laten_dim, laten_w, laten_w)
        sampled = self.to_dec_inp(sampled)
        sampled = self.decoder(sampled)
        assert sampled.shape[-1] == self.img_size
        return sampled
    
    @torch.no_grad()
    def enc_dec_img(self, img):
        self.eval()
        return self(img)

    def forward(self, img):
        img_laten = self.to_laten(self.encoder(img)) # (b, laten_dim, laten_w, laten_w)
        b, c, h, w = img_laten.shape
        img_laten = rearrange(img_laten, 'b c h w -> b (c h w)') # (b, laten_dim*laten_w*laten_w)
        mean_log_var = self.to_mean_and_log_var(img_laten)  # (b, laten_dim * 2)
        mean, log_var = torch.chunk(mean_log_var, chunks=2, dim=1)
        # var always > 0, so log var can be negative or postive
        noise = torch.randn_like(mean, device=img.device)
        
        std = torch.exp(log_var * 0.5)
        sampled_noise = mean +  std * noise # (b, laten_dim)
        
        reconstruct_inp = self.noise_to_dec_inp(sampled_noise) # (b, laten_dim*laten_w*laten_w)
        reconstruct_inp = rearrange(reconstruct_inp, 'b (c h w) -> b c h w', c=c, h=h, w=w) # (b, laten_dim, laten_w, laten_w)
        reconstruct_inp = self.to_dec_inp(reconstruct_inp)
        reconstructed = self.decoder(reconstruct_inp)
        return reconstructed