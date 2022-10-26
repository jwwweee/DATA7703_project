import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch
import os 

def read_all_file(data_path):
    file_paths = []
    steps = []
    directory = os.walk(data_path)  
    for path, _, file_list in directory:  
        for file_name in file_list:  
            file_path = os.path.join(path, file_name)
            step = file_name.split('.')[0].split('_')[1]
            file_paths.append(file_path)
            steps.append(step)
    return file_paths, steps

def data_preprocessing(data_path, step, BATCH_SIZE):
    data = pd.read_csv(data_path,index_col=0)

    X = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]

    X_scaled = StandardScaler().fit_transform(X)

    # print(X_scaled.shape)

    X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X_scaled, y, test_size=0.1, shuffle=False)
    X_train_set = torch.tensor(X_train_set, dtype=torch.float).view(-1, step, 7)
    X_test_set = torch.tensor(X_test_set, dtype=torch.float).view(-1, step, 7)
    y_train_set = torch.tensor(y_train_set, dtype=torch.float).view(-1,1)
    y_test_set = torch.tensor(y_test_set, dtype=torch.float).view(-1,1)

    torch_dataset = Data.TensorDataset(X_train_set, y_train_set)

    train_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
    )

    return train_loader, X_test_set, y_test_set

# data_path = r'..\result'
# file_paths, steps = read_all_file(data_path)

# print(file_paths[0])
# train_loader, X_test_set, y_test_set = data_preprocessing(file_paths[0], steps[0], 500)