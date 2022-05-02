import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import math
from net import *
import random

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderS(object):
    def __init__(self, file_name, train, valid, device, window):    # test = 1 - train - valid
        self.window = window
        fin = open(file_name)
        self.dat = np.loadtxt(fin, delimiter=',')
        self.n, self.var_num = self.dat.shape
        self.var_num-=1
        self.scale = np.ones(self.var_num)
        self._split(0.6, 0.8)

        self.device = device


    def _split(self, train, valid):
        total_setX, total_setY = self._batchify(range(self.window, self.n))
        length = len(total_setY)
        index = torch.randperm(length)
        train=int(train*length)
        valid=int(valid*length)
        train_excerpt, valid_excerpt, test_excerpt = index[:train], index[train:valid], index[valid:],
        train_X, valid_X, test_X = total_setX[train_excerpt], total_setX[valid_excerpt], total_setX[test_excerpt]
        train_Y, valid_Y, test_Y  = total_setY[train_excerpt], total_setY[valid_excerpt], total_setY[test_excerpt]
        self.train = [train_X, train_Y]
        self.valid = [valid_X, valid_Y]
        self.test = [test_X, test_Y]

    def _batchify(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.window, self.var_num))
        Y = torch.zeros((n, 1))
        offset=0
        sum1=0
        sum0=0
        for i in range(n):
            if(i+offset>=n):
                break
            end = idx_set[i+offset]
            start = end - self.window
            X[i, :, :] = torch.tensor(self.dat[start:end, :-1])
            if sum(self.dat[start:end,-1])>=1:
                Y[i, :] = torch.tensor([1])
                sum1+=1
                offset+=self.window-1
                # offset+=int(self.window)//3 -1
            else:
                Y[i, :] = torch.tensor([0])
                offset+=self.window-1
                sum0+=1
            # Y[i,:] = torch.tensor(self.dat[end, -1])

        return X[:i], Y[:i]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        print('length',length)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

class Optim(object):

    def _makeOptimizer(self):
        self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)

    def __init__(self, params, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # Compute gradients norm.
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)

        self.optimizer.step()
        return grad_norm

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
        #only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()