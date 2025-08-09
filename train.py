import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch import optim
from GRID_corpus import GRID
from AV_WAE import AV_WAE
import os
from pytorchtools import EarlyStopping
import argparse

#%% Torch seed:
torch.manual_seed(23)

#%% network parameters
def main(args):
    
    input_dim = 513
    latent_dim = 32 
    hidden_dim_encoder = [128]
    activation = torch.tanh  # activation for audio nets
    activationV = nn.ReLU() # activation for video nets
    
    #%% STFT parameters
    wlen_sec=0.04096
    hop_percent= 0.335  
    fs=25000
    zp_percent=0
    trim=False
    verbose=False
    
    #%% training parameters
    data_dir_tr = '/mnt/storage1/dataset/gridcorpus_avhubert_resnet/train' # Path to your preprocessed training data
    data_dir_val = '/mnt/storage1/dataset/gridcorpus_avhubert_resnet/dev'  # Path to your preprocessed validation data

    file_list_tr = []
    for root, dirs, files in os.walk(data_dir_tr):
        for name in files:
            if name.endswith('.wav'):
                pt_path = os.path.join(root.replace('audio', 'video'), name[:-4] + '.pt')
                npy_path = os.path.join(root.replace('audio', 'video'),'output', name[:-4] + '.npy')
                if os.path.exists(pt_path) and os.path.exists(npy_path):
                    file_list_tr.append(os.path.join(root, name))

    file_list_val = []
    for root, dirs, files in os.walk(data_dir_val):
        for name in files:
            if name.endswith('.wav'):
                pt_path = os.path.join(root.replace('audio', 'video'), name[:-4] + '.pt')
                npy_path = os.path.join(root.replace('audio', 'video'),'output', name[:-4] + '.npy')
                if os.path.exists(pt_path)and os.path.exists(npy_path):
                    file_list_val.append(os.path.join(root, name))

    print(file_list_tr[0])
    print(file_list_val[0])
    
    print('Number of training samples: ', len(file_list_tr))
    print('Number of validation samples: ', len(file_list_val)) 
    print('done')
    
    epoches = 200
    lr = args.lr
    batch_size = args.batch_size
    
    save_dir = args.save_dir
    if not os.path.isdir(save_dir): 
        os.makedirs(save_dir)
    
    num_workers = 1
    shuffle_file_list = True
    shuffle_samples_in_batch = True

    
    device =  args.device
    pgrad = args.pgrad 
    beta = args.beta 
    
    #%%
    
    train_dataset = GRID('training', file_list=file_list_tr, wlen_sec=wlen_sec, 
                     hop_percent=hop_percent, fs=fs, zp_percent=zp_percent, 
                     trim=trim, verbose=verbose, batch_size=batch_size, 
                     shuffle_file_list=shuffle_file_list, video_part=True, facial_attributes=True)
    
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, 
                                       shuffle=shuffle_samples_in_batch, 
                                       num_workers=num_workers)

    val_dataset = GRID('validation', file_list=file_list_val, wlen_sec=wlen_sec, 
                     hop_percent=hop_percent, fs=fs, zp_percent=zp_percent, 
                     trim=trim, verbose=verbose, batch_size=batch_size, 
                     shuffle_file_list=shuffle_file_list, video_part=True, facial_attributes=True)
    
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, 
                                       shuffle=shuffle_samples_in_batch, 
                                       num_workers=num_workers)
    
    print('data loader')
    print('len(train_dataloader.dataset)', len(train_dataloader.dataset))
    print('len(val_dataloader.dataset)', len(val_dataloader.dataset))
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model = AV_WAE(input_dim=input_dim, latent_dim=latent_dim, 
            hidden_dim_encoder=hidden_dim_encoder, 
            activation=activation, activationV = activationV,mode=args.mode).to(device)
    
    print(f"UniVoice Total Parameters: {count_parameters(model)}")


    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # AV-WAE loss function
    def loss_function(recon_xi, xi, mui, logvari, muzpi, logvarzpi, recon_xi_zp, beta=1.0):
        # Reconstruction loss for xi
        recon_loss_xi = torch.mean((recon_xi - xi) ** 2)
        # Reconstruction loss for zpi
        recon_loss_zpi = torch.mean((recon_xi_zp - xi) ** 2)
        # Euclidean distance in latent space as a simple approximation of Wasserstein distance
        latent_euclidean_distance = torch.mean((mui - muzpi) ** 2)    
        # WAE loss
        loss = beta * (recon_loss_xi + recon_loss_zpi) + (1 - beta) * latent_euclidean_distance

        return loss
    
    def SwitchGradParam(model, gmode = False):
                
        model.latent_logvar_layer.bias.requires_grad = gmode
        model.latent_logvar_layer.weight.requires_grad = gmode
        model.latent_mean_layer.bias.requires_grad = gmode
        model.latent_mean_layer.weight.requires_grad = gmode
        model.decoder_layerZ.bias.requires_grad = gmode
        model.decoder_layerZ.weight.requires_grad = gmode
        model.encoder_layerX.bias.requires_grad = gmode
        model.encoder_layerX.weight.requires_grad = gmode
        
        return model
    

    #%% main loop for training
        
    save_loss_dir_tr = os.path.join(save_dir, 'Train_loss')  
    save_loss_dir_val = os.path.join(save_dir, 'Validation_loss') 
            
    # initialize the early_stopping object
    if args.early_stop:
        checkpoint_path = os.path.join(save_dir,'_checkpoint.pt')
        
        early_stopping = EarlyStopping(save_dir = checkpoint_path)
    
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = [] 
        
    skip_epoch = 1
    

    epoch0 = 0
    
    if os.path.isfile(checkpoint_path) and args.early_stop == 'True':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch0 = checkpoint['epoch']
        
        print("=> loaded checkpoint '{}' (epoch {})"
          .format(checkpoint_path, checkpoint['epoch']))
    
    if os.path.isfile(save_loss_dir_tr+'.npy'):
        avg_train_losses = np.load(save_loss_dir_tr+'.npy')
    
    if os.path.isfile(save_loss_dir_val+'.npy'):
        avg_valid_losses = np.load(save_loss_dir_val+'.npy')
    
    for epoch in range(epoch0, epoches):
        
        model.train()
    
        for batch_idx, (batch_audio, batch_video, batch_facial_features) in enumerate(train_dataloader):
            # toggle requires.grad occasionally:
            rn = np.random.rand()
            if rn < pgrad:
                model = SwitchGradParam(model, gmode = False)
                
            batch_audio = batch_audio.to(device)
            #print('batch_audio', batch_audio.size())
            batch_video = batch_video.view(batch_video.size(0), -1).to(device)
            #print('batch_video', batch_video.size())
            batch_facial_features = batch_facial_features.to(device)
            
            recon_batch, mu, logvar, muz, logvarz, recon_batch_zp = model(batch_audio, batch_video, batch_facial_features)
            loss = loss_function(recon_batch, batch_audio, mu, logvar, muz, logvarz, recon_batch_zp, beta=beta)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            model = SwitchGradParam(model, gmode = True)
        
        if epoch % skip_epoch == 0:
    
            # validation loss:
            model.eval()
            with torch.no_grad():
                for batch_idx, (batch_audio, batch_video, batch_facial_features) in enumerate(val_dataloader):
                    batch_audio = batch_audio.to(device)
                    batch_video = batch_video.view(batch_video.size(0), -1).to(device)
                    batch_facial_features = batch_facial_features.to(device)
                    recon_batch, mu, logvar, muz, logvarz, recon_batch_zp = model(batch_audio, batch_video, batch_facial_features)
                    loss = loss_function(recon_batch, batch_audio, mu, logvar, muz, logvarz, recon_batch_zp, beta=beta)
                    valid_losses.append(loss.item())      
                    
        train_loss = np.sum(train_losses) / len(train_dataloader.dataset)
        valid_loss = np.sum(valid_losses) / len(val_dataloader.dataset)
        
        avg_train_losses = np.append(avg_train_losses, train_loss)
        avg_valid_losses = np.append(avg_valid_losses, valid_loss)
           
        epoch_len = len(str(epoches))
        
        print_msg = (f'====> Epoch: [{epoch:>{epoch_len}}/{epoches:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
             
        np.save(save_loss_dir_tr, avg_train_losses)
        np.save(save_loss_dir_val, avg_valid_losses)
            
        train_losses = []
        valid_losses = []
            
    
        if args.early_stop == 'True':
            early_stopping(train_loss, valid_loss, model, epoch, optimizer)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    
    save_file = os.path.join(save_dir, 'final_model_'+'.pt')
    torch.save(model.state_dict(), save_file)

#%%

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    
    parser.add_argument('--save_dir', type=str, default='./experiments/1', help='directory to save the model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'enhance'],
                    help='AV_WAE internal mode: train or enhance')
    parser.add_argument("--pgrad", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.9, help='weight between log p(x|z) and log p(x|zp)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument ('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--early_stop', type=str, default="True", help='early stopping')
    
    args = parser.parse_args()
    print(args)

    main(args)