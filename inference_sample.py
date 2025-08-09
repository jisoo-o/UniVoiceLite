import torch
import numpy as np
import soundfile as sf 
import librosa
import torch.nn as nn
from MCEM_algo import VI_MCEM_algo
from AV_WAE import AV_WAE_Decoder, AV_WAE
import os


#%% network parameters

input_dim = 513
latent_dim = 32
device = 'cpu' # 'cuda' 
hidden_dim_encoder = [128]
activation = torch.tanh
activationv = nn.ReLU()
landmarks_dim = 768 # if you use raw video data, this dimension should be 67*67. Otherwise, if you use the
#                       pre-trained ASR feature extractor, this dimension is 1280

#%% MCEM algorithm parameters

niter_MCEM = 100 # number of iterations for the MCEM algorithm
niter_MH = 40 # total number of samples for the Metropolis-Hastings algorithm
burnin = 30 # number of initial samples to be discarded
var_MH = 0.01 # variance of the proposal distribution
tol = 1e-5 # tolerance for stopping the MCEM iterations
    
    
#%% STFT parameters

#wlen_sec=64e-3
wlen_sec=0.04096
hop_percent= 0.435
   
  
fs=25000
wlen = int(wlen_sec*fs) # window length of 64 ms
wlen = np.int32(np.power(2, np.ceil(np.log2(wlen)))) # next power of 2

nfft = wlen
hop = np.int32(hop_percent*wlen) # hop size
win = np.sin(np.arange(.5,wlen-.5+1)/wlen*np.pi); # sine analysis window

speaker_name = 'bbas2p'
noise_names = ['station', 'kitchen_01', 'metro_01', 'cafeteria_01']
noise_level = ['9']

speaker_noises = []
save_dirs = []

saved_models = 'Wasserstein_ckpt/saved_model_batch_256_lr_1e-6'

_, ckpt_name = os.path.split(saved_models)


for noise_name in noise_names:
    save_dirs.append(f'./output_dir') # directory to save results
    for noise in noise_level:
        speaker_noises.append(f'./snr_data/{noise_name}/{speaker_name}_with_{noise_name}_{noise}dB.wav')
     
file_list_npy = f'/mnt/storage1/dataset/gridcorpus_avhubert_resnet/test/video/output/{speaker_name}.npy'
file_list_pt = f'/mnt/storage1/dataset/gridcorpus_avhubert_resnet/test/video/{speaker_name}.pt'

for idx, speaker_noise in enumerate(speaker_noises):
    save_dir = save_dirs[idx // len(noise_level)]
    print('Save dir : ', save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    video_file = file_list_npy
    pt_file = file_list_pt
    
    print('Processing file: ', speaker_noise)

    path, file_name = os.path.split(speaker_noise)
    file_name = file_name[:-4]
    print('file_name : ',file_name)
    
    #%% Read input audio and video observations:
    
    x, fs = librosa.load(speaker_noise, sr=fs)
    x = x/np.max(np.abs(x)) # normalize input mixture
    v = np.load(video_file)     
    T_orig = len(x)

    i = torch.load(pt_file)
    
    v = v.transpose(0, 1)
    i = i.expand_as(torch.tensor(v))
    
    
    K_b = 10 # NMF rank for noise model
    
    X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)
    
    
    # Observed power spectrogram of the noisy mixture signal
    X_abs_2 = np.abs(X)**2
    X_abs_2_tensor = torch.from_numpy(X_abs_2.astype(np.float32))
    
    F, N = X.shape
    
    # check if the number of video frames is equal to the number of spectrogram frames. If not, augment video by repeating the last frame:
    Nl = np.maximum(N, v.shape[1])
    
    if v.shape[1] < Nl:
        v = np.hstack((v, np.tile(v[:, [-1]], Nl-v.shape[1])))
      
    v = v.T
    v = torch.from_numpy(v.astype(np.float32))
    v.requires_grad = False
    
    # Random initialization of NMF parameters
    eps = np.finfo(float).eps
    np.random.seed(0)
    W0 = np.maximum(np.random.rand(F,K_b), eps)
    H0 = np.maximum(np.random.rand(K_b,N), eps)
    
    
    V_b0 = W0@H0
    V_b_tensor0 = torch.from_numpy(V_b0.astype(np.float32))
    
    g0 = np.ones((1,N))
    g_tensor0 = torch.from_numpy(g0.astype(np.float32))
    
    if idx == 0:
        saved_model_av_cvae = os.path.join(saved_models, 'final_model_AV-WAE.pt')  

        # Loading the pre-trained model
        vae = AV_WAE(input_dim = input_dim, latent_dim = latent_dim, hidden_dim_encoder = hidden_dim_encoder,
                    activation = activation, activationV = activationv).to(device)

        checkpoint = torch.load(saved_model_av_cvae, map_location = 'cpu')
        print(checkpoint.keys())
        vae.load_state_dict(checkpoint, strict = False)

        decoder = AV_WAE_Decoder(vae)

        vae.eval()
        decoder.eval()
        
        # Freeze the decoder parameters
        for param in decoder.parameters():
            param.requires_grad = False

    # Initialize the latent variables by encoding the noisy mixture
    with torch.no_grad():
        data_orig = X_abs_2
        data = data_orig.T
        data = torch.from_numpy(data.astype(np.float32))
        data = data.to(device)
        vae.eval()
        

        z, _ = vae.encode(data, v,i)
        z = torch.t(z)
        v = v.transpose(0, 1)
        mu_z, logvar_z = vae.zprior(v,i)
        
    Z_init = z.numpy() 
    mu_z = mu_z.numpy()
    logvar_z = logvar_z.numpy()
    
    # MCEM algorithm for estimating the noise model parameters:
    vi_mcem_algo = VI_MCEM_algo(mu_z, logvar_z, X=X, W=W0, H=H0, Z=Z_init, v = v, decoder=decoder,
                        niter_VI_MCEM=niter_MCEM, niter_MH=niter_MH, burnin=burnin, var_MH=var_MH,i=i)

    S_hat, N_hat = vi_mcem_algo.run()

    s_hat = librosa.istft(stft_matrix=S_hat, hop_length=hop,
                        win_length=wlen, window=win, length=T_orig)
    b_hat = librosa.istft(stft_matrix=N_hat, hop_length=hop,
                        win_length=wlen, window=win, length=T_orig)

    # save the results:
    save_vae = os.path.join(save_dir, ckpt_name)
    if not os.path.isdir(save_vae):
        os.makedirs(save_vae)
        speaker_noise
    speech_name = 'est_'+ file_name+ '_speech.wav'
    noise_name = 'est_'+ file_name+ '_noise.wav'
    sf.write(os.path.join(save_vae, speech_name), s_hat, fs)
    sf.write(os.path.join(save_vae, noise_name), b_hat, fs)
                
    print(f'{speaker_noise[:-4]} done')        
                            