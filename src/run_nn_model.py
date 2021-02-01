import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from src.f1_loss import F1_Loss


def run_model(model,running_mode='train', train_X=None, train_Y=None, valid_X=None, valid_Y=None, test_set=None,
    batch_size=1000, learning_rate=0.01, n_epochs=1, start_epoch=0, stop_thr=1e-4, shuffle=True, weights=[.5,.5], save_fn=""):

    if running_mode == 'train':
        loss = {'train':[], 'valid':[]}
        f1 = {'train':[], 'valid':[]}
        prev_loss = 0
        prev_train_loss = 100
        num_decays = 1
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_Y), batch_size=batch_size, shuffle=shuffle)
        if valid_X != None: valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_X, valid_Y), batch_size=batch_size, shuffle=shuffle)
        for epoch in range(start_epoch, n_epochs):
            print("Running epoch " + str(epoch) + ": lr: " + str(learning_rate) + ", weights: " + str(weights))

            # _, valid_loss = _test(model, valid_loader, weights)
            # valid_f1 = _get_f1(model, valid_X, valid_Y) * 100
            # loss['valid'].append(valid_f1)
            # print("    validation: loss: " + str(valid_loss) + ", f1: " + str(valid_f1))

            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            _, train_loss = _train(model, train_loader, optimizer, weights)
            train_f1, train_P, train_R = _get_f1(model, train_X, train_Y)
            print("    train: loss: " + str(train_loss) + ", P: " + str(train_P) + ", R: " + str(train_R) + ", f1: " + str(train_f1))
            loss['train'].append(train_loss)
            f1['train'].append(train_f1)
            if prev_train_loss < train_loss:
                learning_rate *= .6
                num_decays += 1
                new_w0 = 2*weights[0]/(1+2*weights[0])
                weights = [new_w0, 1-new_w0]
                prev_train_loss = 100
            else: prev_train_loss = train_loss

            if valid_X != None:
                _, valid_loss = _test(model, valid_loader, weights)
                valid_f1, valid_P, valid_R = _get_f1(model, valid_X, valid_Y)
                loss['valid'].append(valid_f1)
                print("    valid: loss: " + str(valid_loss) + ", P: " + str(valid_P) + ", R: " + str(valid_R) + ", f1: " + str(valid_f1))
                # if np.abs(valid_loss - prev_loss) < stop_thr: break;
                prev_loss = valid_loss

            # if epoch%10 == 0 or epoch == n_epochs - 1:
            torch.save(model.state_dict(), "../data/basic_nn/nn_save" + save_fn + "_epoch" + str(epoch) + ".pt")

        return model, loss, f1
    else:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
        return _test(model, test_loader)


def _train(model,data_loader,optimizer,weights,device=torch.device('cpu')):

    weights = torch.FloatTensor(weights)
    loss_func = nn.CrossEntropyLoss(weight=weights)
    # loss_func = F1_Loss()
    losses = []
    # accs = []
    for batch, target in data_loader:
        # target = target.reshape(-1)
        optimizer.zero_grad()
        output = model(batch.float())
        # print()
        # print(output.shape)
        # print(target.shape)
        loss = loss_func(output, target)
        # f1 = _get_f1(model, batch, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        # accs.append(f1)
        # print(acc)
    # print(losses)
    # print(f1)
    return model, np.mean(losses)


def _test(model, data_loader, weights, device=torch.device('cpu')):
    weights = torch.FloatTensor(weights)
    loss_func = nn.CrossEntropyLoss(weight=weights)
    losses = []
    for batch, target in data_loader:
        output = model(batch.float())
        loss = loss_func(output, target)
        losses.append(loss.item())
    return model, np.mean(losses)

def _get_f1(model, data, targets, epsilon=1e-7):
    pred = torch.argmax(model(data.float()),1)
    # print(pred)
    # print(targets)
    correct = sum(pred * targets)
    P = correct / (sum(pred) + epsilon)
    R = correct / (sum(targets) + epsilon)
    # print(P)
    # print(R)
    # assert(False)
    return 100*2*P*R / (P + R + epsilon), 100*P, 100*R

