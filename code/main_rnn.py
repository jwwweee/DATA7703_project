from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os

from RNN import *
from Bagging_RNN import *
from log import *

def rnn_train(data_path, time_step, batch_size, epoch, lr, input_size, hidden_size, num_layer, output_size):
    train_loader, X_test_set, y_test_set = data_preprocessing(data_path, time_step, batch_size)

    rnn = RNN(input_size, hidden_size, num_layer, output_size)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    
    rnn, train_losses, test_losses = train(train_loader, rnn, optimizer, loss_func, X_test_set, y_test_set, epoch)
    prediction = rnn(X_test_set)   # rnn output

    r2, mae, rmse = evaluate(prediction.data.numpy(), y_test_set)
    # print('rnn: ', r2)
    # print('rnn MAE: ', mae)
    # print('rnn RMSE: ', rmse)

    info = [time_step, hidden_size, num_layer, batch_size, r2, mae, rmse, train_losses[-1], test_losses[-1]]
    losses = [train_losses, test_losses]
    return info, losses, rnn

def evaluate(y_pred, y_truth):
    r2 = r2_score(y_pred, y_truth)
    mae = mean_absolute_error(y_pred, y_truth)
    rmse = np.sqrt(mean_squared_error(y_pred, y_truth))
    return r2, mae, rmse

def plot_loss(train_losses, test_losses):
    plt.plot(range(len(train_losses)), train_losses, label='train')
    plt.plot(range(len(test_losses)), test_losses, label='test')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data_path = 'data/Brisbanecbd_24.csv'
    rnn_log_path = 'rnn_results/rnn_log_24.csv'

    if os.path.exists(rnn_log_path):
        os.remove(rnn_log_path)
    
    # Hyper Parameters
    INPUT_SIZE = 7       # rnn input size
    OUTPUT_SIZE = 1      # rnn output size
    EPOCH = 15
    LR = 0.01            # learning rate
    TIME_STEP = 24       # rnn time step

    HIDDEN_SIZE = 128
    NUM_LAYER = 2
    BATCH_SIZE = 512
    K=3

    rnn_count = 0

    # write header
    rnn_log_header = ['index', 'time_step', 'hidden_size', 'num_layer', 'batch_size', 'r2', 'mae', 'rmse', 'train_loss', 'test_loss']
    record_log(rnn_log_path, rnn_log_header)

    min_rnn_loss = 10
    min_rnn_info = []
    min_rnn_losses = []
    min_rnn_model = []


    for i in range(30):
        print("Training Times:", i)
        info, losses, rnn = rnn_train(data_path, TIME_STEP, BATCH_SIZE, EPOCH, LR, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, OUTPUT_SIZE)
        rnn_loss = info[8]
        print('rnn_loss:', rnn_loss)

        min_rnn_info = [rnn_count] + info
        
        record_log(rnn_log_path, min_rnn_info)

        if min_rnn_loss > rnn_loss:
            min_rnn_loss = rnn_loss
            min_rnn_losses = losses
            min_rnn_model = rnn
            print(min_rnn_loss)
            save_loss('rnn_results/pickles/rnn_24_'+'.pkl', min_rnn_losses)
            save_net('rnn_results/net/rnn_24_'+'.pt', min_rnn_model)

            

        rnn_count+=1 