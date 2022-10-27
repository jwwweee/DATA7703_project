import torch
from torch import nn

from data_preprosessing import *

class BaggingRNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers, output_size, K):
        super(BaggingRNN, self).__init__()

        self.rnns = []
        self.fcs = []
        self.paras = []
        self.K = K
        
        self.relu = nn.ReLU(inplace=True)
        for _ in range(self.K):
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,     # rnn hidden unit
                num_layers=num_layers,       # number of rnn layer
                batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            )

            for name, param in self.rnn.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)
            
            self.fc = nn.Linear(hidden_size, output_size)
            para = torch.nn.Parameter(torch.FloatTensor([1/self.K]), requires_grad=True)
            
            self.rnns.append(self.rnn)
            self.fcs.append(self.fc)
            self.paras.append(para)

    def forward(self, x):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        y_out = 0
        for i in range(self.K):
            r_out, _ = self.rnns[i](x)
            r_out = r_out[:, -1, :]
            y = self.fcs[i](r_out)
            y_out += y*self.paras[i]

        return y_out

def train(train_loader, rnn, optimizer, loss_func, X_test, y_test, epoch):
    train_losses = []
    test_losses = []
    for i in range(epoch):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            prediction = rnn(batch_x)   # rnn output
            train_loss = loss_func(prediction, batch_y)         # calculate loss
            train_losses.append(train_loss.detach().numpy())

            optimizer.zero_grad()                   # clear gradients for this training step
            train_loss.backward()                         # backpropagation, compute gradients
            optimizer.step()                        # apply gradients

            prediction = rnn(X_test)   # rnn output
            test_loss = loss_func(prediction, y_test)         # calculate loss
            test_losses.append(test_loss.detach().numpy())

    # print(train_losses[-1])

    return rnn, train_losses, test_losses



