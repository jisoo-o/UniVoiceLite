import torch
from torch import nn
import torch.nn.functional as F


class AV_WAE(nn.Module):
    def __init__(self, input_dim=None, latent_dim=None,
                 hidden_dim_encoder=None, activation=None, activationV=None,
                 mode="train"):
        """
        mode: "train" or "enhance"
        """
        super(AV_WAE, self).__init__()
        
        self.input_dim = input_dim          # Dimension of the audio input
        self.latent_dim = latent_dim        # Dimension of the latent space
        self.landmarks_dim = 768            # Dimension of the visual feature
        self.hidden_dim_encoder = hidden_dim_encoder  # Hidden layer dimensions for the encoder
        self.activation = activation        # Activation function for the audio network
        self.activationV = activationV      # Activation function for the visual network
        self.mode = mode                    # Mode flag: "train" or "enhance"
           
        # Visual feature encoding layers
        self.vfeats = nn.Sequential(
            nn.Linear(self.landmarks_dim * 2 , 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )

        # Decoder layers
        self.decoder_layerZ = nn.Linear(self.latent_dim, self.hidden_dim_encoder[0])
        self.decoder_layerV = nn.Linear(128, self.hidden_dim_encoder[0])     

        # Audio and visual encoding layers
        self.encoder_layerX = nn.Linear(self.input_dim, self.hidden_dim_encoder[0])
        self.encoder_layerV = nn.Linear(128, self.hidden_dim_encoder[0])      
        
        # z_prior layers (define prior distribution from visual features)
        self.zprior_layer = nn.Linear(128, self.hidden_dim_encoder[0])
        self.zprior_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.zprior_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim_encoder[0], self.input_dim)

        # Layers for the mean and variance of latent variables
        self.latent_mean_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        self.latent_logvar_layer = nn.Linear(self.hidden_dim_encoder[0], self.latent_dim)
        
    # Compute audio embeddings
    def encode_audio(self, x):
        audio_embedding = self.encoder_layerX(x)
        return self.activation(audio_embedding)

    # Compute visual embeddings
    def encode_visual(self, v, i):
        v_i = torch.cat((v, i), dim=1)  # Concatenate visual features and facial attributes
        visual_embedding = self.vfeats(v_i)
        return self.activationV(visual_embedding)

    # Full encoding function to compute mean and variance of latent variables
    def encode(self, x, v, i):
        print(v.shape)
        print(i.shape)

        # Apply transpose only in enhancement mode
        if self.mode == "enhance":
            v = v.transpose(0, 1)

        # Encode visual features
        v_i = torch.cat((v, i), dim=1)
        ve = self.vfeats(v_i)
        
        # Combine audio and visual embeddings
        print(ve.shape)
        xv = self.encoder_layerX(x) + self.encoder_layerV(ve)
        he = self.activation(xv)
        
        # Compute mean and variance
        return self.latent_mean_layer(he), self.latent_logvar_layer(he)

    # Define prior distribution in latent space
    def zprior(self, v, i):
        v_i = torch.cat((v, i), dim=1)
        zp1 = self.vfeats(v_i)
        zp = self.zprior_layer(zp1)
        zp = self.activationV(zp)
        
        return self.zprior_mean_layer(zp), self.zprior_logvar_layer(zp)

    # Reparameterization trick to sample latent variables
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    # Decoding function
    def decode(self, z, v, i):
        v_i = torch.cat((v, i), dim=1)
        vd = self.vfeats(v_i)
        zv = self.decoder_layerZ(z) + self.decoder_layerV(vd)
        hd = self.activation(zv)
        return torch.exp(self.output_layer(hd))

    # Forward pass of the full network
    def forward(self, x, v, i):
        # Encode audio and visual features
        mu, logvar = self.encode(x, v, i)
            
        # Sample latent variable z
        z = self.reparameterize(mu, logvar)
        
        # Sample from the prior distribution
        mu_zp, logvar_zp = self.zprior(v, i)
        z_p = self.reparameterize(mu_zp, logvar_zp)
        
        # Audio reconstruction
        return self.decode(z, v, i), mu, logvar, mu_zp, logvar_zp, self.decode(z_p, v, i)
        
        
class AV_WAE_Decoder(nn.Module):
    
    def __init__(self, cwae):
        
        super(AV_WAE_Decoder, self).__init__()
        self.latent_dim = cwae.latent_dim
        self.activation = cwae.activation 
        self.activationV = cwae.activationV 
        self.output_layer = None
        self.build(cwae)
        
    def build(self, cwae):
        self.vfeats = cwae.vfeats                
        self.output_layer = cwae.output_layer
        self.decoder_layerZ = cwae.decoder_layerZ
        self.decoder_layerV = cwae.decoder_layerV
        
    def forward(self, z, v, i):
        if v.shape[1] == 1:
            v_i = torch.cat((v, i), dim=2)
        else:
            v_i = torch.cat((v, i), dim=1)
        vd = self.vfeats(v_i) 
        zv = self.decoder_layerZ(z) + self.decoder_layerV(vd)
        hdd = self.activation(zv)
            
        return torch.exp(self.output_layer(hdd))
