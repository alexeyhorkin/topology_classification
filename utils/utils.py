'''All useful utils here.'''

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join as p_join

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("[ Using Seed : ", seed, " ]")



#########################################
##      DATA PROCESSING FUNCTIONS      ##
#########################################

def create_dataset(folders, verbose=False, feature_nums=16, n='both'):
    X_merged, Y_merged = [], []
    for folder_path in tqdm(folders):
        X_path = p_join(folder_path, '2nd_exp_Input.txt')
        Y_path_down = p_join(folder_path, '2nd_exp_Topology_down.txt')
        Y_path_up = p_join(folder_path, '2nd_exp_Topology_up.txt')
        _ = p_join(folder_path, '2nd_exp_Parameters.txt')


        values = pd.read_csv(X_path).values
        X = select_n_center_features(values, feature_nums)

        Y_down = pd.read_csv(Y_path_down).values
        Y_up = pd.read_csv(Y_path_up).values
        Y = np.array([map_classes(Y_up[i], Y_down[i]) for i in range(Y_up.shape[0])])

        if n == 'n':
            mask = values[:,  -1] == -1
        elif n == 'n+1':
            mask = ~(values[:,  -1] == -1)
        else:
            mask = np.array([True] * len(values))


        X = X[mask]
        Y = Y[mask]

        X_merged.append(X)
        Y_merged.append(Y)

    # Merge and shuffle it!
    X, Y = np.concatenate(X_merged, axis=0), np.concatenate(Y_merged, axis=0)
    indexes_for_shuffle = np.random.permutation(np.arange(X.shape[0]))
    X = X[indexes_for_shuffle]
    Y = Y[indexes_for_shuffle]
    if verbose:
        print('Dataset cteated!')
    return X, Y


def helper(num):
    n_needed = 'both'
    if num % 2 == 0:
        n_needed = 'n+1'
        N = f'N2={num - 1}'
    else:
        n_needed = 'n'
        N = f'N2={num}'
    return N, n_needed


def make_merged_dataset_by_N(data_path, nums, n_feat=16, normalize=True):
    seed_all()
    X_merged, Y_merged = [], []
    for num in nums:
        N, n = helper(num)
        FOLDERS = [p_join(os.path.abspath(data_path), item) for item in os.listdir(data_path) if N in item]
        X, Y = create_dataset(FOLDERS, feature_nums=n_feat, n=n, verbose=False)
#         X = normalize_data(X)   ### Attention! may provide better results ???
        X_merged.append(X)
        Y_merged.append(Y)

    X, Y = np.concatenate(X_merged, axis=0), np.concatenate(Y_merged, axis=0)
    indexes_for_shuffle = np.random.permutation(np.arange(X.shape[0]))
    X = X[indexes_for_shuffle]
    Y = Y[indexes_for_shuffle]
    
    print(f'Dataset for nums:{nums} was created')
    if normalize:
        X = normalize_data(X)
    return X, Y


def create_dataloaders(X, Y, cpu_workers=2, train_bs=64, test_bs=64):
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)


    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  num_workers=cpu_workers,
                                  batch_size=train_bs,
                                  drop_last=True)

    test_dataloader = DataLoader(test_dataset,
                                 shuffle=False,
                                 num_workers=cpu_workers,
                                 batch_size=test_bs,
                                 drop_last=False)

    return train_dataloader, test_dataloader


def normalize_data(X):
    mean = X.mean(0)
    std = X.std(0)
    return (X - mean) / std


def select_n_center_features(data: np.ndarray, n_features: int, verbose: bool = False) -> np.ndarray:
    from copy import deepcopy

    total_components = data.shape[1]
    start = int((total_components - n_features)/2)
    res = deepcopy(data)[:, start: start + n_features]
    if verbose:
        print(f'Selected features from indexes:  [{start}, {start + n_features})')
    return res

def map_classes(i, j):
    if i == 0 and j == 0:
        return 0
    elif i == 0 and j == 1:
        return 1
    elif i == 1 and j == 0:
        return 2
    elif i == 1 and j == 1:
        return 3



#########################################
## TRAINING AND TESTING FUNCTIONS (DL) ##
#########################################

def train_epoch(net, optimizer, dataloader, criterion, device):
    """Perform one traing epoch"""
    running_loss = 0.0
    net.train(True)
    for X, y in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device).long()
        optimizer.zero_grad()
        y_hat = net(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def test_model(net, dataloader, device):
    """Return accuracy"""
    correct, count = 0, 0
    net.eval()
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device).long()
            logits = net(X)
            _, y_hat = torch.max(logits, dim=1)
            correct += (y_hat == y).sum().item()
            count += y_hat.shape[0]
    
    return 100 * correct / count


def run_training(net, optimizer, config, train_loader, test_loader=None):
    ### Get hyperparams from config
    epochs = config.get('ephs', 10)
    device = config.get('device', 'cpu')
    criterion = config.get('criterion', F.cross_entropy)
    ckpt_save_folder = config.get('ckpt_save_folder', './ckpts')
    save_ckpts = config.get('save_ckpts', False)
    
    ### Define local constants
    best_score = -np.inf
    
    ### Create folder for saving ckpts
    if save_ckpts:
        if not os.path.exists(ckpt_save_folder):
            os.makedirs(ckpt_save_folder)
    
    ### Start trainig
    for eph in range(1, epochs + 1):
            mean_loss = train_epoch(net, optimizer, train_loader, criterion, device)
            print(f"Epoch: {eph}/{epochs}, \t train loss: {mean_loss}")
            if test_loader:
                ### Start testing
                score = test_model(net, test_loader, device)

                if score > best_score:
                    best_score = score
                    if save_ckpts:
                        torch.save(net.state_dict(), p_join(ckpt_save_folder, f"model_best.ckpt"))
                print(f"Epoch: {eph}/{epochs}, \t total score test: {score}, [best score: {best_score}]")
            print()
    
    return best_score


#####################################
######    ML USEFUL METHODS     #####
#####################################

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV

def calc_ml_method(model, config, X, Y):
    res = {}

    scoring = config.get('scoring', 'accuracy')
    cv = config.get('cv', 5)
    n_jobs = config.get('n_jobs', 4)
    
    scores = cross_validate(model, X, Y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    res[str(scoring)] = scores
        
    return res

def greed_searc_cv(model_class, params, config, X, Y):
    res = {}

    scoring = config.get('scoring', 'accuracy')
    cv = config.get('cv', 5)
    n_jobs = config.get('n_jobs', 4)

    refit_ = 'accuracy' if isinstance(scoring, list) else True
    
    model = GridSearchCV(model_class,
                         params,
                         scoring=scoring,
                         refit=refit_,
                         cv=cv,
                         n_jobs=n_jobs)
    model.fit(X, Y)
    res[f'best_{str(scoring)}_score'] = model.best_score_
    res['best_params'] = model.best_params_
    res['cv_results'] = model.cv_results_
    res['best_index'] = model.best_index_
    
    return res

def count_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO]: Model "{model.__class__.__name__}" has {pytorch_total_params} trainable parameters')


######################################
######    GRAPH PLOT METHODS     #####
######################################

def plot_matshow(df, x_labels, y_labels, cmap_name='YlGn'):
    fig, ax = plt.subplots()
    cax = ax.matshow(df, cmap=plt.get_cmap(cmap_name))
    fig.colorbar(cax)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('Train')
    ax.set_xlabel('Test')
    return fig, ax