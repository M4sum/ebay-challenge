import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class basic_nn_model:
    #baseline model, simply creating clusters based on shared attributes such as MPN, etc

    def __init__(self, mode = 'run', key_attrs = None, num_attrs = {1: 20, 2: 20, 3: 20, 4: 20, 5: 20}, layer_dims = [20,20,5]):
        self.mode = mode
        self.num_attrs = num_attrs
        self.key_attrs = {cat: key_attrs[cat][1:num_attrs[cat]+1] for cat in range(1,6)}
        if mode == 'run':
            print("Building nn model...")
            self.__build(layer_dims)
        elif mode == 'train' or mode == 'prepare':
            print("mode: ",mode)

    def __build(self, layer_dims):
        self.models = {}
        for cat in range(1,6):
            model = Ebay_Cat_Classifier(layer_dims)
            model.load_state_dict(torch.load("../data/basic_nn/nn_save_cat" + str(cat) + "_final.pt"))
            self.models[cat] = model
        self.clusters = {}


    def import_data(self, data, labels):
        self.data, self.data_labels = self.organize_data(data, labels)

    def organize_data(self, data, labels):
        print("grouping data...")
        gdata = self.group_data(data)
        Xc = {}
        Yc = {}
        print("organizing data...")
        for cat in range(1,6):
            print("organizing cat " + str(cat))
            examples = gdata[cat]
            num_listings = examples.shape[0]
            num_params = examples.shape[1]-1
            X = np.zeros((num_listings*(num_listings-1), num_params), dtype=int)
            Y = np.zeros((num_listings*(num_listings-1),1), dtype=int)
            print("X.shape",X.shape)
            print("Y.shape",Y.shape)
            xi = 0
            # for i in range(num_listings):
            #     for j in range(num_listings):
            for i in range(num_listings):
                for j in range(i+1,num_listings):
                    # if i==j: continue
                    if xi%100000 == 0: print("xi:"+str(xi))
                    X[xi,:],Y[xi,0] = self.getRow(examples, labels, i, j)
                    xi+=1
            Xc[cat] = X
            Yc[cat] = Y
            print(X[0:5,:])
            print(Y[0:5,:])
            # break
        print("done")
        return Xc, Yc

    def get_clusters(self, data, fn = 'basic_nn_clusters.tsv'):
        print("grouping data...")
        gdata = self.group_data(data)
        self.clusters = {}      # id -> clusterid
        output_file = open('../data/'+fn, 'w')  # write mode
        print("processing data clusters...")
        for cat in range(3,6):
            print("processing cat " + str(cat))
            examples = gdata[cat]
            examples_not_none = np.not_equal(examples, 0)
            model = self.models[cat]
            # skip = True
            for i in range(examples.shape[0]):
                if i%100 == 0:
                    output_file.flush()
                    print(i,"of",examples.shape[0])
                id = examples[i,0]
                # if cat == 1 and id == 102143: skip = False
                # if skip: continue
                if id in self.clusters: cluster = self.clusters[id]
                else:
                    cluster = str(cat) + str(id)
                    self.clusters[id] = cluster
                    output_file.write(str(id) + "\t" + cluster + "\n")
                # t2 = time.time()
                possibles = examples[i+1:,:]
                possibles_not_none = examples_not_none[i+1:,1:]
                rows = self.getRow_v(examples[i,1:],possibles[:,1:], possibles_not_none)
                # t3 = time.time()
                # print("time2: " + str(t3-t2))
                if len(rows) == 0: continue     #for last element
                prediction = torch.argmax(model(rows.float()), 1)
                # t4 = time.time()
                # print("time3: " + str(t4-t3))
                matches = torch.nonzero(prediction, as_tuple=False)
                if isinstance(matches, str): matches = [matches]
                for match in matches:
                    match_id = possibles[match,0]
                    self.clusters[match_id] = cluster
                    output_file.write(str(match_id) + "\t" + cluster + "\n")
        print("done")
        output_file.close()
        return self.clusters

    def save_clusters(self, clusters, fn = 'basic_nn_clusters_old.tsv'):
        print(clusters)
        with open('../data/'+fn, 'w') as f:
            s = ""
            for id in clusters:
                print(id)
                print(clusters[id])
                s += id + "\t" + clusters[id] + "\n"
            f.write(s)

    def group_data(self, data):
        groups = {i: [] for i in range(1, 6)}
        attr_dict = {}
        dict_id = 1
        for id in data:
            listing = data[id]
            cat = int(listing[0])
            # row = np.empty(self.num_attrs[cat]+1).astype(object)
            row = np.zeros(self.num_attrs[cat]+1, dtype=int)
            # row[:] = None
            row[0] = id
            for i,attr in enumerate(self.key_attrs[cat]):
                if attr in listing[3]:
                    attr_id, dict_id = self.fetch_from_attr_dict(listing[3][attr], attr_dict, dict_id)
                    row[i+1] = attr_id
            groups[cat].append(row)
            # groups[cat][id] = listing
        return {i:np.array(groups[i]) for i in groups}

    def fetch_from_attr_dict(self, attr_val, attr_dict, dict_id):
        if attr_val in attr_dict: return attr_dict[attr_val],dict_id
        else:
            attr_dict[attr_val] = dict_id
            return dict_id, dict_id+1

    def getRow(self, examples, labels, i, j):
        a = np.not_equal(examples[i, 1:], None)
        b = np.not_equal(examples[j, 1:], None)
        eq = np.equal(examples[i, 1:], examples[j, 1:])
        c = np.logical_and(np.logical_and(a,b),eq)
        Xi = c + a*b*(c-1)
        Yi = labels[examples[i,0]] == labels[examples[j,0]]
        # print(examples[i, 1:])
        # print(examples[j, 1:])
        # print(a)
        # print(b)
        # print(c)
        # print(Xi)
        # print(Yi)
        # assert(False)
        return Xi, Yi

    def getRow_v(self, example, possibles, possibles_not_none):
        # t1 = time.time()
        t2 = time.time()
        a = np.not_equal(example, 0)
        # print("   time1: " + str(t2-t1))
        # b = np.not_equal(possibles, None)
        b = possibles_not_none
        # t3 = time.time()
        # print("   time12: " + str(t3-t2))
        aandb = np.logical_and(a,b)
        # t4 = time.time()
        # print("   time13: " + str(t4-t3))
        eq = np.equal(example, possibles)
        # t5 = time.time()
        # print("   time14: " + str(t5-t4))
        c = np.logical_and(aandb,eq).astype(int)
        # t6 = time.time()
        # print("   time15: " + str(t6-t5))
        not_eq_or_none = np.logical_and(aandb,np.logical_not(eq)).astype(int)
        # t7 = time.time()
        # print("   time16: " + str(t7-t6))
        # Xi = c + aandb*(c-1)
        Xi = c - not_eq_or_none
        # t8 = time.time()
        # print("   time16: " + str(t8-t7))
        return torch.tensor(Xi)

    def attr_match(self, attr, listing1, listing2):
        if attr in listing1[3] and attr in listing2[3]:
            return listing1[3][attr] == listing2[3][attr]
        return False

    def write_out_data(self, fn = "", Xc = None, Yc = None):
        if Xc is None: Xc = self.data
        if Yc is None: Yc = self.data_labels
        for cat in Xc:
            print("saving cat " + str(cat))
            xfn = '../data/basic_nn/' + fn + 'X_'+str(cat) + ".csv"
            yfn = '../data/basic_nn/' + fn + 'Y_'+str(cat) + ".csv"
            pd.DataFrame(Xc[cat]).to_csv(xfn, index=False, header=False)
            pd.DataFrame(Yc[cat]).to_csv(yfn, index=False, header=False)

    def load_cat(self, cat):
        xfn = '../data/basic_nn/X_'+str(cat) + ".csv"
        yfn = '../data/basic_nn/Y_'+str(cat) + ".csv"
        Xc = pd.read_csv(xfn).to_numpy()
        Yc = pd.read_csv(yfn).to_numpy()
        return Xc, Yc

    def resample(self, target_perc = .05):
        pass



class Ebay_Cat_Classifier(nn.Module):

    def __init__(self, layer_dims = [60,20,5]):
        super(Ebay_Cat_Classifier, self).__init__()
        self.hidden1 = nn.Linear(layer_dims[0], layer_dims[1])
        self.hidden2 = nn.Linear(layer_dims[1], layer_dims[2])
        self.output_layer = nn.Linear(layer_dims[2], 2)

    def forward(self, inputs):
        x = F.relu(self.hidden1(inputs))
        x = F.relu(self.hidden2(x))
        x = self.output_layer(x)
        return x


def run_model(model, running_mode='train', train_X=None, train_Y=None, valid_X=None, valid_Y=None, test_set=None,
              batch_size=1000, learning_rate=0.01, n_epochs=1, start_epoch=0, stop_thr=1e-4, shuffle=True,
              weights=[.5, .5], save_fn="", prev_train_loss=100):

    if running_mode == 'train':
        loss = {'train': [], 'valid': []}
        f1 = {'train': [], 'valid': []}
        prev_loss = 0
        num_decays = 1
        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_Y),
                                                   batch_size=batch_size, shuffle=shuffle)
        if valid_X != None: valid_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(valid_X, valid_Y), batch_size=batch_size, shuffle=shuffle)
        for epoch in range(start_epoch, n_epochs):
            print("Running epoch " + str(epoch) + ": lr: " + str(learning_rate) + ", weights: " + str(weights))

            # _, valid_loss = _test(model, valid_loader, weights)
            # valid_f1 = _get_f1(model, valid_X, valid_Y) * 100
            # loss['valid'].append(valid_f1)
            # print("    validation: loss: " + str(valid_loss) + ", f1: " + str(valid_f1))

            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            _, train_loss = _train(model, train_loader, optimizer, weights)
            train_f1, train_P, train_R = _get_f1(model, train_X, train_Y)
            print("    train: loss: " + str(train_loss) + ", P: " + str(train_P) + ", R: " + str(
                train_R) + ", f1: " + str(train_f1))
            loss['train'].append(train_loss)
            f1['train'].append(train_f1)
            if prev_train_loss < train_loss:
                learning_rate *= .6
                num_decays += 1
                new_w0 = 2 * weights[0] / (1 + 2 * weights[0])
                weights = [new_w0, 1 - new_w0]
                prev_train_loss = 100
            else:
                prev_train_loss = train_loss

            if valid_X != None:
                _, valid_loss = _test(model, valid_loader, weights)
                valid_f1, valid_P, valid_R = _get_f1(model, valid_X, valid_Y)
                loss['valid'].append(valid_f1)
                print("    valid: loss: " + str(valid_loss) + ", P: " + str(valid_P) + ", R: " + str(
                    valid_R) + ", f1: " + str(valid_f1))
                # if np.abs(valid_loss - prev_loss) < stop_thr: break;
                prev_loss = valid_loss

            # if epoch%10 == 0 or epoch == n_epochs - 1:
            torch.save(model.state_dict(), "../data/basic_nn/nn_save" + save_fn + "_epoch" + str(epoch) + ".pt")

        return model, loss, f1
    else:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
        return _test(model, test_loader)

def _train(model, data_loader, optimizer, weights, device=torch.device('cpu')):

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
    pred = torch.argmax(model(data.float()), 1)
    # print(pred)
    # print(targets)
    correct = sum(pred * targets)
    P = correct / (sum(pred) + epsilon)
    R = correct / (sum(targets) + epsilon)
    # print(P)
    # print(R)
    # assert(False)
    return 100 * 2 * P * R / (P + R + epsilon), 100 * P, 100 * R
