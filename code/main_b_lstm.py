from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os

from RNN import *
from Bagging_RNN import *
from log import *

def bagging_train(data_path, time_step, batch_size, epoch, lr, input_size, hidden_size, num_layer, output_size, K):
    train_loader, X_test_set, y_test_set = data_preprocessing(data_path, time_step, batch_size)

    bagging_rnn = BaggingRNN(input_size, hidden_size, num_layer, output_size, K)
    optimizer = torch.optim.Adam(bagging_rnn.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    bagging_rnn, train_losses, test_losses = train(train_loader, bagging_rnn, optimizer, loss_func, X_test_set, y_test_set, epoch)
    prediction = bagging_rnn(X_test_set)   # rnn output

    r2, mae, rmse = evaluate(prediction.data.numpy(), y_test_set)

    info = [time_step, hidden_size, num_layer, batch_size, K, r2, mae, rmse, train_losses[-1], test_losses[-1]]
    losses = [train_losses, test_losses]
    return info, losses, bagging_rnn

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

    # Hyper Parameters
    INPUT_SIZE = 7       # rnn input size
    OUTPUT_SIZE = 1      # rnn output size
    EPOCH = 15
    LR = 0.01            # learning rate
    TIME_STEP = 24       # rnn time step

    HIDDEN_SIZE = 128
    NUM_LAYER = 2
    BATCH_SIZE = 512
    
    Ks = [3]
    for K in Ks:
        bagging_rnn_count = 0

        # write header
        bagging_rnn_log_path = 'b_lstm_results/K_'+str(K)+'_bagging_lstm_log_24.csv'

        if os.path.exists(bagging_rnn_log_path):
            os.remove(bagging_rnn_log_path)

        bagging_rnn_log_header = ['index', 'time_step', 'hidden_size', 'num_layer', 'batch_size', 'k', 'r2', 'mae', 'rmse', 'train_loss', 'test_loss']
        record_log(bagging_rnn_log_path, bagging_rnn_log_header)

        min_b_rnn_loss = 10
        min_bagging_rnn_info = []
        min_bagging_rnn_losses = []
        min_bagging_rnn_model = []


        for i in range(10):
            print("Training K with " + str(K) + " at " + str(i) + "times")
    
            b_info, b_losses, b_rnn = bagging_train(data_path, TIME_STEP, BATCH_SIZE, EPOCH, LR, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, OUTPUT_SIZE, K)
            b_rnn_loss= b_info[9]
            print('b_rnn_loss:', b_rnn_loss)

            min_bagging_rnn_info = [bagging_rnn_count]+b_info
            
            record_log(bagging_rnn_log_path, min_bagging_rnn_info)

            if min_b_rnn_loss > b_rnn_loss:
                min_b_rnn_loss = b_rnn_loss
                print(min_b_rnn_loss)
                min_bagging_rnn_losses = b_losses
                min_bagging_rnn_model = b_rnn
                save_loss('b_lstm_results/pickles/K_'+str(K)+'_b_rnn_'+'.pkl', min_bagging_rnn_losses)
                save_net('b_lstm_results/net/K_'+str(K)+'_b_rnn_'+'.pt', min_bagging_rnn_model)
        
            bagging_rnn_count+=1