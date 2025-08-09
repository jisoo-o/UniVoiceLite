#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:14:02 2019

@author: smostafa
"""

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_dir, encoder_layer_sizes = None, decoder_layer_sizes = None, zprior_layer_sizes = None, patience=20, verbose=True, val_loss_min = np.Inf):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = val_loss_min
        if not np.isinf(val_loss_min):
            self.best_score = -val_loss_min
            
        self.save_dir = save_dir

        self.encoder_layer_sizes = encoder_layer_sizes
        self.decoder_layer_sizes = decoder_layer_sizes
        self.zprior_layer_sizes = zprior_layer_sizes
        
    def __call__(self, train_loss, valid_loss, model, epoch, optimizer):

        score = -valid_loss
        

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, valid_loss, model, epoch, optimizer)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, valid_loss, model, epoch, optimizer)
            self.counter = 0

    def save_checkpoint(self, tr_loss, val_loss, model, epoch, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tr_loss': tr_loss,
                'val_loss': val_loss,
                'encoder_layer_sizes': self.encoder_layer_sizes,
                'decoder_layer_sizes': self.decoder_layer_sizes,
                'zprior_layer_sizes': self.zprior_layer_sizes}, self.save_dir)
    
#        torch.save(model.state_dict(), str(self.model_name)+'_checkpoint.pt')
        self.val_loss_min = val_loss