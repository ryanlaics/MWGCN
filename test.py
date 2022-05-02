import argparse
import math
import time
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from net import magnn
import numpy as np
import pywt
from util import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)####0

parser = argparse.ArgumentParser()

parser.add_argument('--save', type=str, default='model8.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=26,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.1,help='dropout rate')
parser.add_argument('--seq_length',type=float,default=32,help='seq length')
parser.add_argument('--subgraph_size',type=int,default=10,help='k')
parser.add_argument('--node_dim',type=int,default=4,help='dim of nodes')
parser.add_argument('--gnn_channels',type=int,default=3,help='gnn channels')
parser.add_argument('--scale_channels',type=int,default=3,help='scale channels')
parser.add_argument('--end_channels',type=int,default=3,help='end channels')
parser.add_argument('--layers',type=int,default=3,help='number of layers')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--lr',type=float,default=0.0001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--epochs',type=int,default=2000,help='')

args = parser.parse_args()
device = torch.device(args.device)
torch.set_num_threads(8)
def wavelet_FCNN_preprocessing_set(X, waveletLevel=args.layers-1, waveletFilter='haar'):
    X=X.squeeze()
    N = X.shape[0]
    feature_dim = X.shape[1]
    length = X.shape[2]
    signal_length = []
    signal_length.append(length)

    extened_X = []
    extened_X.append(np.transpose(X, (0, 2, 1)))

    for i in range(N):# for each sample
        for j in range(feature_dim): # for each dim
            wavelet_list = pywt.wavedec(X[i][j], waveletFilter, level=waveletLevel)
            if i == 0 and j == 0:
                for l in range(waveletLevel):
                    current_length = len(wavelet_list[waveletLevel - l])
                    signal_length.append(current_length)
                    extened_X.append(np.zeros((N,current_length,feature_dim)))
            for l in range(waveletLevel):
                extened_X[l+1][i,:,j] = wavelet_list[waveletLevel-l]
    wav_list=[]
    for mat in extened_X:
        mat_mean = mat.mean()
        mat_std = mat.std()
        if mat_std!=0:
            mat = np.transpose((mat-mat_mean)/(mat_std),(0,2,1))
        else:
            mat = np.transpose((mat-mat_mean),(0,2,1))
        mat=mat[:,np.newaxis,:,:]
        wav_list.append(mat)
    return wav_list

def evaluate(data, X, Y, model, batch_size):
    model.eval()
    total_loss = 0
    n_samples = 0
    total_y_pred_list=[]
    total_y_list=[]

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2, 3)
        X_wav = X.cpu().data.numpy()
        X_wav = wavelet_FCNN_preprocessing_set(X_wav)
        with torch.no_grad():
            output, adj_matrix = model(X_wav)
        loss = criterion(output, Y)
        total_y_pred_list.append(np.rint(output.T.cpu().data.numpy().reshape(-1)).astype(np.int32))
        total_y_list.append(Y.T.cpu().data.numpy().reshape(-1).astype(np.int32))
        total_loss += loss.item()
        n_samples += (output.size(0))
    total_y_pred=np.concatenate(total_y_pred_list)
    total_y=np.concatenate(total_y_list)
    tn, fp, fn, tp = confusion_matrix(total_y, total_y_pred).ravel()
    print(classification_report(total_y, total_y_pred))

    return total_loss / n_samples

def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    total_y_pred_list = []
    total_y_list = []
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2, 3)
        X_wav=X.cpu().data.numpy()
        X_wav=wavelet_FCNN_preprocessing_set(X_wav)  # X: (sample_num, feature_num, sequence_length)
        output, adj_matrix = model(X_wav)
        total_y_pred_list.append(np.rint(output.T.cpu().data.numpy().reshape(-1)).astype(np.int32))
        total_y_list.append(Y.T.cpu().data.numpy().reshape(-1).astype(np.int32))
        loss = criterion(output, Y)
        loss.backward()
        total_loss += loss.item()
        n_samples += output.size(0)
        optim.step()
        if iter%100==0:
            print('iter:{:3d} | loss: {:.5f}'.format(iter,loss.item()/(output.size(0))))
        iter += 1
    total_y_pred = np.concatenate(total_y_pred_list)
    total_y = np.concatenate(total_y_list)
    tn, fp, fn, tp = confusion_matrix(total_y, total_y_pred).ravel()
    print(classification_report(total_y, total_y_pred))

    return total_loss / n_samples





data_dir = "dataset/15wt.txt"

Data = DataLoaderS(data_dir, 0.6, 0.2, device, args.seq_length)

model = magnn(args.gcn_depth, args.num_nodes,
                  device, node_dim=args.node_dim, subgraph_size=args.subgraph_size, dropout=args.dropout,
                  scale_channels=args.scale_channels, end_channels= args.end_channels, gnn_channels = args.gnn_channels,
                  seq_length=args.seq_length,
                  layers=args.layers, propalpha=args.propalpha)
model = model.to(device)

# criterion = BCEFocalLoss().to(device)
criterion = torch.nn.BCELoss(reduction='sum')

best_val = 10000000
optim = Optim(model.parameters(), args.lr, args.clip, lr_decay=args.weight_decay)


# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
vtest_acc = evaluate(Data, Data.valid[0], Data.valid[1], model, args.batch_size)
test_acc = evaluate(Data, Data.test[0], Data.test[1], model, args.batch_size)
print("final test rse {:5.4f} ".format(test_acc))
