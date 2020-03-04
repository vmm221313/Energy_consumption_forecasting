# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class baseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_output_cols, window_size, dropout_prob):
        super(baseLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.dropout_prob = dropout_prob
        self.num_output_cols = num_output_cols

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p = dropout_prob)
        
        self.lstm_layer1 = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim)
        self.lstm_layer2 = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim)
        self.lstm_layer3 = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, num_output_cols)

    def attention_net(self, all_hidden_states, next_hidden_state):

        attn_weights = torch.bmm(all_hidden_states.reshape(1, self.window_size, self.hidden_dim), next_hidden_state.reshape(1,self.hidden_dim,1))
        soft_attn_weights = F.softmax(attn_weights, dim = 1).reshape(1, 1, self.window_size)
        new_hidden_state = torch.bmm(soft_attn_weights, torch.transpose(all_hidden_states,0,1))
        
        return new_hidden_state
        
    def forward(self, input_, hidden_state, cell_state):
        
        all_hidden_states, (next_hidden_state, next_cell_state) = self.lstm_layer1(input_, (hidden_state, cell_state))
        next_hidden_state = self.tanh(next_hidden_state).to(device)
        next_cell_state = self.tanh(next_cell_state).to(device)
        
        #all_hidden_states, (next_hidden_state, next_cell_state) = self.lstm_layer2(input_, (next_hidden_state, next_cell_state))
        #next_hidden_state = self.tanh(self.dropout(next_hidden_state)).to(device)
        #next_cell_state = self.tanh(self.dropout(next_cell_state)).to(device)
        
        
        #all_hidden_states, (next_hidden_state, next_cell_state) = self.lstm_layer3(input_, (next_hidden_state, next_cell_state))
        #next_hidden_state = self.tanh(self.dropout(next_hidden_state)).to(device)
        #next_cell_state = self.tanh(self.dropout(next_cell_state)).to(device)
        
        #next_hidden_state = self.attention_net(all_hidden_states, next_hidden_state)
        
        out = self.fc4(next_hidden_state).to(device)
        
        return out, (next_hidden_state, next_cell_state)
