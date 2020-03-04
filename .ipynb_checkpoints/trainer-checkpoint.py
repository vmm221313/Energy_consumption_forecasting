import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from tqdm import tqdm_notebook
import pandas as po
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(train_dataloader, window_size, model, loss_function, optimizer, num_epochs, input_dim, hidden_dim, dropout_prob):
    
    hidden_state = torch.randn(1, 1, hidden_dim)
    cell_state = torch.randn(1, 1, hidden_dim)
    
    losses = []
    
    model.train()
    for epoch in range(int(num_epochs)):
        print('Training epoch {}'.format(epoch+1))
        for step, batch in enumerate(train_dataloader):
            b_inputs = batch[0].tolist()
            b_target = batch[1].tolist()

            for i in range(window_size, len(b_inputs)):
                model.zero_grad()
                optimizer.zero_grad()

                input_ = torch.tensor(np.array(b_inputs[i-window_size:i]), dtype = torch.float).view(window_size, 1, input_dim).to(device)
                prediction, (hidden_state, cell_state) = model(input_, hidden_state, cell_state)
                prediction = prediction.view(1).to(device) 

                target = torch.tensor([b_target[i]]).to(device)
                hidden_state.detach_()
                cell_state.detach_()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                loss = loss_function(prediction, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
        print('Loss after {} epochs = {}'.format(epoch + 1, loss))
        
    return model, hidden_state, cell_state, losses



