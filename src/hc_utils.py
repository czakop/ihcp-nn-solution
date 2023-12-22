import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import copy

def normalize_dataset(data):
    x = np.array([x for x,_ in data])
    x1 = x.reshape((x.shape[0], x.shape[1], -1))
    x = (x-x1.mean(axis=(0,2))[None, :, None, None])/x1.std(axis=(0,2))[None, :, None, None]
    
    for i in range(len(data)):
        data[i] = (x[i], data[i][1])
        
    return data

def load_data(path='./data/heat_cond_data_n10.pickle', normalize=False):
    with open(path, mode='rb') as input_file:
        dataset = pickle.load(input_file)
        
    if normalize:
        dataset = normalize_dataset(dataset)
        
    return dataset

class HeatConductionDataset(Dataset):
    
    def __init__(self, data, noise=None, normalize=False):
        self.data = data
        self.noise = noise
        self.normalize = normalize
        
    def __getitem__(self, index):
        x, y = self.data[index]
        
        if self.noise:
            x = x + np.random.normal(0, self.noise, x.shape)
            
        if self.normalize:
            x1 = x.reshape((x.shape[0],-1))
            x = (x-x1.mean(axis=1)[:, None, None])/x1.std(axis=1)[:, None, None]
    
        return torch.from_numpy(x).double(), torch.from_numpy(y).double()
        
    def __len__ (self):
        return len(self.data)
    
def get_data_loaders(dataset, test_size=0.2, batch_size=128, shuffle=True, random_state=1, training_noise=None, normalize=False):
    train_data, test_data = train_test_split(dataset, test_size=test_size, shuffle=shuffle, random_state=random_state)
    
    train_dataset = HeatConductionDataset(train_data, noise=training_noise, normalize=normalize)
    test_dataset = HeatConductionDataset(test_data, normalize=normalize)
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(test_dataset, batch_size=batch_size)

def epoch_step(model, dataloader, loss_fn, optimizer, device, train=True):
    losses = []
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()
            preds = model(x)
        else:
            with torch.no_grad():
                preds = model(x)

        loss = loss_fn(preds, y)

        losses.append(loss.item())

        if train:
            loss.backward()
            optimizer.step()
            
    return losses

def train_model(model, traindata, valdata, loss_fn, optimizer, device, n_epoch, prog_bar_desc='Epochs', early_stop=False, patience=10):
    train_losses, valid_losses = [], []
    model = model.to(device)
    patience_counter = 0
    min_loss = 100
    best_model = None

    for _ in tqdm(range(n_epoch), desc=prog_bar_desc):
        model.train()
        epoch_losses = epoch_step(model, traindata, loss_fn, optimizer, device, train=True)
        train_losses.append(np.mean(epoch_losses))

        model.eval()
        epoch_losses = epoch_step(model, valdata, loss_fn, optimizer, device, train=False)
        valid_losses.append(np.mean(epoch_losses))

        if early_stop:
            if valid_losses[-1] < min_loss:
                patience_counter = 0
                min_loss = valid_losses[-1]
                best_model = copy.deepcopy(model)
            elif patience_counter > patience:
                break
            else:
                patience_counter += 1

    return train_losses, valid_losses, best_model

def transfer_learning(model, traindata, valdata, loss_fn, optimizer, device, n_epoch, n_epoch_fn, lr_scaler=0.1):
    train_losses, valid_losses, _ = train_model(model, traindata, valdata, loss_fn, optimizer, device, n_epoch, prog_bar_desc='Training')
                                             
    model.requires_grad_(True)

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*lr_scaler

    train_losses_fn, valid_losses_fn, _ = train_model(model, traindata, valdata, loss_fn, optimizer, device, n_epoch_fn, prog_bar_desc='Finetuning')
                                             
    return train_losses+train_losses_fn, valid_losses+valid_losses_fn

def plot_loss(train_losses, valid_losses, start=0, end=None, filename=None):
    x = np.arange(len(train_losses[start:end]))+1

    fig, ax = plt.subplots(1, 1, figsize=(7,7))

    ax.set_title('MSE over epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    
    colors = plt.cm.Dark2(np.linspace(0, 1, 2))

    ax.plot(x, valid_losses[start:end], label='Validation', color=colors[0])
    ax.plot(x, train_losses[start:end], label='Training', color=colors[1])

    ax.legend()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_prediction(T, true, pred, figsize=(7,7), title='Upper plane', filename=None):
    n = T.shape[-1]
    
    _, ax = plt.subplots(1,1,figsize=figsize)
        
    ax.imshow(T[-1], cmap='coolwarm')
    ax.set_title(title, fontsize=20)

    x,y,z,d = true
    ax.plot(z*n-.5, y*n-.5, 'go', ms = 6, mfc = 'g')
    c1 = plt.Circle((z*n-.5, y*n-.5), d*n/2, color='g', fill=False, lw=2)
    ax.add_patch(c1)

    x,y,z,d = pred
    ax.plot(z*n-.5, y*n-.5, 'yo', ms = 6, mfc = 'y')
    c2 = plt.Circle((z*n-.5, y*n-.5), d*n/2, color='y', fill=False, lw=2)
    ax.add_patch(c2)

    if filename:
        plt.savefig(filename)
        
    plt.show()


class Evaluator():
    def __init__(self, model, dataloader=None, dataset=None, loss_fn=nn.MSELoss()):
        self.model = model
        self.loss_fn = loss_fn
        
        if dataloader:
            self.dataloader = dataloader
        else:
            assert dataset, 'Dataset must be specified'
            self.dataloader = DataLoader(HeatConductionDataset(dataset, noise=False), batch_size=1, shuffle=True)
            
        self.iterator = iter(self.dataloader)
        
    def calculate_loss(self, device='cpu'):
        self.model.eval()
        epoch_losses = epoch_step(self.model, self.dataloader, self.loss_fn, None, device, train=False)
        return np.mean(epoch_losses)
        
    def plot_next(self):
        x, y = next(self.iterator)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(x)

        print('true:', y[0].numpy())
        print('prediction:', preds[0].numpy())
        print('MSE:', self.loss_fn(y[0], preds[0]).item())
        
        plot_prediction(x[0], y[0], preds[0])

    def plot_average(self, avg=None, tol=5e-5, filename=None, device='cpu'):
        if avg is None:
            avg = self.calculate_loss()
        for x, y in self.dataloader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                preds = self.model(x)

            loss = self.loss_fn(preds, y)

            avg_losses = torch.abs(loss-avg)<tol

            if torch.sum(avg_losses)>0:
                index = torch.where(avg_losses)[0].item()
                plot_prediction(x[index], y[index], preds[index], title=f'MSE: {np.round(loss.item(), 5)}', filename=filename)
                break

        if index is None:
            raise Exception('No such a loss')