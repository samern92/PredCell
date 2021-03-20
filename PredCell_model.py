# Authors - Samer Nour Eddine (snoure01@tufts.edu), Apurva Kalia (apurva.kalia@tufts.edu)
### IN PROGRESS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StateUnit(nn.Module):
    def __init__(self, layer_level, timestep, initializer,isTopLayer = False):
        super().__init__()
        self.layer_level = layer_level
        self.timestep = timestep
        self.state_ = st_initializer.sample() # the idea is to sample from a distribution
        self.recon_ = 0 # reconstructions at all other time points will be determined by the state
        self.V = np.random.uniform(-1,1,size = (3,3)) # figure out size?
        
    def forward(self, BU_err, TD_err):
        self.timestep += 1
        if isTopLayer:
            self.state_ = nn.LSTM(self.state_, BU_err)
        else:
            self.state_ = nn.LSTM(self.state_, BU_err, TD_err)
        self.recon_ = self.V * self.state_ # this is a linear map. Should we use nn.Linear?

        
class ErrorUnit(nn.Module):
    def __init__(self, layer_level, timestep, err_initializer, W_initializer):
        super().__init__()
        self.layer_level = layer_level
        self.timestep = timestep
        self.TD_err = err_initializer.sample()
        self.BU_err = 0
        self.W = W_initializer.sample() 
    def forward(self, state_, recon_):
        self.timestep += 1
        self.TD_err = np.abs(state_ - recon_)
        self.BU_err = self.W * self.TD_err # this is a linear map. Should we use nn.Linear?
    def get_timestep():
        return self.timestep
        
class PredCells(nn.Module):
    def __init__(self, num_layers, total_timesteps):
        self.num_layers = num_layers
        self.total_timesteps = total_timesteps
        self.st_units = []
        self.err_units = []
        self.state_initval = 0
        self.error_initval = 0
        for lyr in range(num_layers):
            self.st_units.append(StateUnit(lyr,0,initializer))
            self.err_units.append(ErrorUnit(lyr,0,initializer))

    def forward(self, input_sentence):
        for t in range(self.total_timesteps):
            # input_char at each t is a one-hot character encoding
            input_char = input_sentence[t]
            for lyr in range(num_layers):
                if lyr == 0:
                    # override the value 
                    self.st_units[lyr] = input_char
                else:
                    self.st_units[lyr] = self.st_units[lyr].forward(self.err_units[lyr-1].BU_err, self.err_units[lyr].TD_err)
                if lyr < num_layers:
                    self.err_units[lyr].forward(self.st_units[lyr].state_, self.st_units[lyr+1].recon_)
                else:
                    self.err_units[lyr].forward()
                
          




            
            
