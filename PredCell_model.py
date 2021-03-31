''' Authors - Samer Nour Eddine (snoure01@tufts.edu), Apurva Kalia (apurva.kalia@tufts.edu)
IN PROGRESS

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing
import keras
import string
import io


class StateUnit(nn.Module):
    def __init__(self, layer_level, timestep,thislayer_dim,lowerlayer_dim,isTopLayer = False):
        super().__init__()
        self.layer_level = layer_level
        self.timestep = timestep
        self.isTopLayer = isTopLayer
        if self.isTopLayer:
            self.LSTM_ = nn.LSTM(thislayer_dim, thislayer_dim, 1)
        else:
            self.LSTM_ = nn.LSTM((lowerlayer_dim + thislayer_dim), thislayer_dim, 1)
        self.state_ = np.zeros(shape = (thislayer_dim, 1))
        self.recon_ = np.zeros(shape = (lowerlayer_dim, 1)) # reconstructions at all other time points will be determined by the state
        self.V = nn.Linear(thislayer_dim,lowerlayer_dim) # maps from this layer to the lower layer
        
    def forward(self, BU_err, TD_err):
        self.timestep += 1
        if self.isTopLayer:
            self.state_ = self.LSTM_(BU_err)
        else:
            self.state_ = self.LSTM_(np.concatenate((BU_err, TD_err), axis = 0)) # not sure about syntax; should we be concatenating these?
        self.recon_ = self.V(self.state_)

        
class ErrorUnit(nn.Module):
    def __init__(self, layer_level, timestep, thislayer_dim, higherlayer_dim):
        super().__init__()
        self.layer_level = layer_level
        self.timestep = timestep
        self.TD_err = np.zeros(shape = (thislayer_dim, 1)) 
        self.BU_err = np.zeros(shape = (higherlayer_dim, 1)) # it shouldn't matter what we initialize this to; it will be determined by TD_err in all other iterations
        self.W = nn.Linear(thislayer_dim,higherlayer_dim)# maps up to the next layer
    def forward(self, state_, recon_):
        self.timestep += 1
        self.TD_err = np.abs(state_ - recon_)
        self.BU_err = self.W(self.TD_err)
    def get_timestep():
        return self.timestep
        
class PredCells(nn.Module): # does this need to be an nn.Module?
    def __init__(self, num_layers, total_timesteps, hidden_dim):
        self.num_layers = num_layers
        self.total_timesteps = total_timesteps
        self.st_units = []
        self.err_units = []
        for lyr in range(num_layers):
            if lyr == 0:
                self.st_units.append(StateUnit(lyr, 0, hidden_dim,numchars))
                self.err_units.append(ErrorUnit(lyr, 0, numchars, hidden_dim))
            elif lyr < num_layers - 1:
                self.st_units.append(StateUnit(lyr, 0, hidden_dim,hidden_dim))
                self.err_units.append(ErrorUnit(lyr, 0, hidden_dim, hidden_dim))
            else:
                self.st_units.append(StateUnit(lyr,0,hidden_dim,hidden_dim,isTopLayer = True))
                self.err_units.append(ErrorUnit(lyr, 0, hidden_dim, hidden_dim))
            

    def forward(self, input_sentence):
        for t in range(self.total_timesteps):
            # input_char at each t is a one-hot character encoding
            input_char = input_sentence[t] # 56 dim one hot vector
            for lyr in range(num_layers):
                if lyr == 0:
                    # set the lowest state unit value to the current character 
                    self.st_units[lyr] = input_char
                else:
                    self.st_units[lyr] = self.st_units[lyr]\
                                         .forward(self.err_units[lyr-1].BU_err\
                                                  , self.err_units[lyr].TD_err)
                if lyr < num_layers - 1:
                    self.err_units[lyr].forward(self.st_units[lyr].state_, self.st_units[lyr+1].recon_)
                else:
                    pass
    

                
            


path = keras.utils.get_file(
    "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display
print("Corpus length:", len(text))

chars = sorted(list(set(text)))
print("Total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

#note that this means that y[i] == x[i+1][-3]

            
            
