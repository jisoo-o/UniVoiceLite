import torch
from torch.utils import data
import os
import numpy as np
import librosa
import random
  
class GRID(data.Dataset):
    
    def __init__(self, data_mode, file_list, wlen_sec=0.04096, 
                 hop_percent=0.25, fs=25000, zp_percent=0, trim=False,
                 verbose=False, batch_size=128, shuffle_file_list=True, video_part = True, facial_attributes = False):
        """
        Initialization of class GRID
        """
        super(GRID, self).__init__()
        self.batch_size = batch_size
        self.file_list = file_list
        self.data_mode = data_mode
        self.wlen_sec = wlen_sec # STFT window length in seconds
        self.hop_percent = hop_percent  # hop size as a percentage of the window length
        self.fs = fs
        self.zp_percent = zp_percent
        self.wlen = self.wlen_sec*self.fs # window length in samples -> 1024
        self.wlen = np.int64(np.power(2, np.ceil(np.log2(self.wlen)))) # next power of 2
        self.hop = np.int64(self.hop_percent*self.wlen) # hop size in samples
        self.nfft = self.wlen + self.zp_percent*self.wlen # number of points of the discrete Fourier transform
        self.win = np.sin(np.arange(.5,self.wlen-.5+1)/self.wlen*np.pi); # sine analysis window
        self.video_part = video_part
        self.facial_attributes = facial_attributes
        
        self.cpt_file = 0
        self.trim = trim
        self.current_frame = 0
        self.tot_num_frame = 0
        self.verbose = verbose
        self.shuffle_file_list = shuffle_file_list
        self.compute_len()
        
    def compute_len(self):
        self.num_samples = 0
        
        for cpt_file, wavfile in enumerate(self.file_list):
            path, file_name = os.path.split(wavfile) # file_name : sx374.wav
            path, speaker = os.path.split(path) # speaker : 01M
            path, set_type = os.path.split(path) # set_type : train
            path, dialect = os.path.split(path) # dialect : volunteers
            # print(f'wavfile : {wavfile}, file_name : {file_name}, speaker : {speaker}, set_type : {set_type}, dialect : {dialect}')
            
            x, fs_x = librosa.load(wavfile, sr = self.fs)
            
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')        
                
            if self.trim: # remove beginning and ending silence
                
                xt, index = librosa.effects.trim(x, top_db=30)
                
                x = np.pad(xt, int(self.nfft // 2), mode='reflect') # (cf. librosa.core.stft)
                
            else:
                
                x = np.pad(x, int(self.nfft // 2), mode='reflect') # (cf. librosa.core.stft)
            
            n_frames = 1 + int((len(x) - self.wlen) / self.hop)

            
            self.num_samples += n_frames

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if self.current_frame == self.tot_num_frame:
        
            if self.cpt_file == len(self.file_list):
                self.cpt_file = 0
                if self.shuffle_file_list:
                    random.shuffle(self.file_list)
            
            wavfile = self.file_list[self.cpt_file]
            self.cpt_file += 1
            
            path, file_name = os.path.split(wavfile)
            path, audio = os.path.split(path)
            path, speaker = os.path.split(path)
            path, set_type = os.path.split(path)
            path_gridTr = path
            path, dialect = os.path.split(path)
            path_gridVal = path

            x, fs_x = librosa.load(wavfile, sr = self.fs)
            
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')  
            
            x = x/np.max(np.abs(x))
            
            if self.video_part:
                if  self.data_mode == 'training':           
                    # Read video data
                    v = np.load(os.path.join(path_gridTr, str(set_type), str(speaker), 'video', 'output', file_name[:-4]+'.npy')) 
                elif self.data_mode == 'validation':
                    v = np.load(os.path.join(path_gridTr, str(set_type), str(speaker), 'video', 'output', file_name[:-4]+'.npy')) 
                elif self.data_mode == 'test':
                    v = np.load(os.path.join(path_gridTr, str(set_type), str(speaker), 'video', 'output', file_name[:-4]+'.npy')) 
                else:
                    raise NameError('Wrong "training" or "validation" mode specificed.')
            
            if self.facial_attributes:
                if self.data_mode == 'training':
                    i_tensor = torch.load(os.path.join(path_gridTr, str(set_type), str(speaker), 'video', file_name[:-4]+'.pt')) 
                    i = i_tensor.numpy()
                elif self.data_mode == 'validation':
                    i_tensor = torch.load(os.path.join(path_gridTr, str(set_type), str(speaker), 'video', file_name[:-4]+'.pt')) 
                    i = i_tensor.numpy()
                elif self.data_mode == 'test':
                    i_tensor = torch.load(os.path.join(path_gridTr, str(set_type), str(speaker), 'video', file_name[:-4]+'.pt')) 
                    i = i_tensor.numpy()    
                else:
                    raise NameError('Wrong "training" or "validation" mode specificed.')


            if self.trim: # remove beginning and ending silence
                
                xt, index = librosa.effects.trim(x, top_db=30)
                            
                X = librosa.stft(xt, n_fft=self.nfft, hop_length=self.hop, 
                                 win_length=self.wlen,
                                 window=self.win) # STFT
                
            else:
                
                X = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, 
                                 win_length=self.wlen,
                                 window=self.win) # STFT                
            
            self.audio_data = np.abs(X)**2 # num_freq x num_frames
            
            if self.video_part:
                self.video_data = v 
                
            else:
                self.video_data = self.audio_data.copy()
            
            if self.facial_attributes:
                self.facial_data = i
            else:
                self.facial_data = self.audio_data.copy()

            # check if num of frames equal
            self.current_frame = 0
            self.tot_num_frame = np.minimum(self.audio_data.shape[1],self.video_data.shape[0])
         
        audio_frame = self.audio_data[:,self.current_frame]
        
        if self.video_part:
            video_frame = self.video_data[self.current_frame,:]
        else:
            video_frame = self.video_data[:,self.current_frame]

        if self.facial_attributes:
            facial_attributes = self.facial_data[0,:]
        else:
            facial_attributes = self.facial_data[:,self.current_frame]
        
        self.current_frame += 1
        
        audio_frame = torch.from_numpy(audio_frame.astype(np.float32))
        video_frame = torch.from_numpy(video_frame.astype(np.float32))
        
            
        return audio_frame, video_frame, facial_attributes
