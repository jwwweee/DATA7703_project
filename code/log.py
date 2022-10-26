import csv
import torch
import pickle
import os
from data_preprosessing import *
# from sklearn.metrics import r2_score

def record_log(path, info):
    with open(path,'a+', newline='') as file:
        csv_write = csv.writer(file)
        csv_write.writerow(info)

def save_net(path, net):
    if os.path.exists(path):
        os.remove(path)
    torch.save(net, path)

def load_net(path):
    net = torch.load(path)
    return net

def save_loss(path, losses):
    if os.path.exists(path):
        os.remove(path)
    with open(path,'wb') as file:
        pickle.dump(losses, file)

def load_loss(path):
    with open(path,'rb') as out_data:
        losses = pickle.load(out_data)
        return losses

def load_prediction_result(load_path):
    _, X_test_set, y_test_set = data_preprocessing('data/Brisbanecbd_24.csv', 24, 512)
    net = load_net(load_path)
    net.eval()
    prediction = net(X_test_set)   # rnn output
    return prediction, y_test_set