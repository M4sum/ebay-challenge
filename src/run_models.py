import torch

# from src.baseline_model import Baseline_model
# from src.agg_cluster_model import agg_cluster_model
# from src.Masum_model import *
from src.basic_nn_model import basic_nn_model, Ebay_Cat_Classifier, run_model
# from src.my_dataset import MyDataset
# from src.slightly_better_model import Slightly_Better_model
from src.eval import *
# from src.naive_bayes_distance import naive_bayes_distance
from src.parser import Parser
# from src.image_nn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_baseline_model():
    print("Parsing data...")
    p = Parser()
    print("Processing Attributes...")
    attrs = p.get_category_attrs()
    print("getting validation set...")
    val_data, _ = p.get_validation_set()
    val_truth = p.get_validation_clusters()
    data = p.data_as_dict()

    print("building model...")
    num_attrs = {1: 10, 2: 5, 3: 6, 4: 7, 5: 4}
    model = Baseline_model(data, attrs, num_attrs)
    print("getting clusters...")
    proposed = model.get_clusters()
    # print("evaluating f1...")
    # f1 = f1_score(proposed, val_truth)
    # print(num_attrs,"attrs:",f1)
    print("getting pretty clusters...")
    pretty_clusters = model.get_pretty_clusters()
    print("saving model...")
    model.save_clusters(pretty_clusters)


def test_baseline_model():
    print("Parsing data...")
    p = Parser()
    print("Processing Attributes...")
    attrs = p.get_category_attrs()
    val_data, _ = p.get_validation_set()
    test_data, _ = p.get_test_set()
    p.clean_non_validation()
    val_truth = p.get_validation_clusters()
    test_truth = p.get_test_clusters()

    #gets max accuracy at 9 attributes, .478
    # for num_attrs in range(5,11):
    # num_attrs = {1: 10, 2: 5, 3: 6, 4: 5, 5: 5}
    num_attrs = {1: 10, 2: 5, 3: 6, 4: 7, 5: 4}
    # num_attrs = {1: 10, 2: 5, 3: 4, 4: 7, 5: 4}
    f1 = 0
    for cat in range(6,6):
        max_attr = 5
        max_f1 = 0
        for num_i in range(4,14):
            num_attrs[cat] = num_i
            model = Baseline_model(val_data, attrs, num_attrs)
            proposed = model.get_clusters()
            f1 = f1_score(proposed, val_truth)
            print(num_attrs,"attrs:",f1)
            if f1 > max_f1:
                max_f1 = f1
                max_attr = num_i
            # else: break
        num_attrs[cat] = max_attr

    if f1 == 0:
        print("Running Validation Data")
        model = Baseline_model(val_data, attrs, num_attrs)
        proposed = model.get_clusters()
        f1 = f1_score(proposed, val_truth)
        print(num_attrs,"attrs:",f1)
        pretty_clusters = model.get_pretty_clusters()
        model.save_clusters(pretty_clusters)

    # Test
    print("Running test data")
    test_model = Baseline_model(test_data, attrs, num_attrs)
    proposed = test_model.get_clusters()
    f1 = f1_score(proposed, test_truth)
    print(num_attrs,"attrs:",f1)
    print(test_model.get_pretty_clusters())

def run_slightly_better_model():
    print("Parsing data...")
    p = Parser()
    print("Processing Attributes...")
    attrs = p.get_category_attrs()
    val_data, _ = p.get_validation_set()
    test_data, _ = p.get_test_set()
    p.clean_non_validation()
    val_truth = p.get_validation_clusters()
    test_truth = p.get_test_clusters()

    #gets max accuracy at 9 attributes, .478
    # for num_attrs in range(5,11):
    # num_attrs = {1: 10, 2: 5, 3: 6, 4: 5, 5: 5}
    num_attrs = {1: 10, 2: 5, 3: 6, 4: 7, 5: 4}
    key_attrs = {cat: attrs[cat][1:num_attrs[cat]+2] for cat in range(1,6)}
    # key_attrs = {1: ['brand', 'size type', 'bottoms size womens', 'material', 'inseam', 'color', 'rise', 'style', 'silhouette', 'country/region of manufacture', 'wash', 'modified item'], 2: ['brand', 'style', 'us shoe size mens', 'color', 'product line', 'model', 'country/region of manufacture', 'material'], 3: ['brand', 'material', 'type', 'pattern', 'model', 'country/region of origin', 'mpn', 'modified item', 'size', 'finish'], 4: ['brand', 'color', 'material', 'type', 'mpn', 'pattern', 'size'], 5: ['brand', 'model', 'type', 'upc', 'item weight']}
    model = Slightly_Better_model(val_data, key_attrs)
    proposed = model.get_clusters()
    f1 = f1_score(proposed, val_truth)
    max_f1 = f1
    for cat in range(1,6):
        for attr_i in range(1,20):
            test_attr = attrs[cat][attr_i]
            model = Slightly_Better_model(val_data, key_attrs)
            new_attrs, proposed = model.test_attr(cat, test_attr)
            f1 = f1_score(proposed, val_truth)
            print("testing:",new_attrs)
            print("f1:",f1)
            if f1 > max_f1:
                max_f1 = f1
                key_attrs = new_attrs
                print("Better F1!")
            print("")

    if f1 == 0:
        print("Running Validation Data")
        model = Slightly_Better_model(val_data, key_attrs)
        proposed = model.get_clusters()
        f1 = f1_score(proposed, val_truth)
        print(num_attrs,"attrs:",f1)

    # Test
    print("Running test data")
    test_model = Slightly_Better_model(test_data, key_attrs)
    proposed = test_model.get_clusters()
    f1 = f1_score(proposed, test_truth)
    print(num_attrs,"attrs:",f1)

def run_agg_cluster_model():
    print("Parsing data, and extracting validation set...")
    p = Parser()
    val_data, _ = p.get_validation_set()
    val_labels = p.get_validation_labels_list()
    p.clean_non_validation()

    nn = Image_nn()
    print("training distance measure...")
    distance_measure = naive_bayes_distance()
    distance_measure.train(val_data, val_labels)
    model = agg_cluster_model(nn, distance_measure)

    print("calculating distance data in model...")
    model.loadData(val_data)

    print("training model...")
    model.train(val_labels)
    proposed = model.get_clusters()
    print("F1 Score:",list_f1_score(proposed, val_labels))

def run_masum_model():
    print("Parsing data, and extracting validation set...")
    p = Parser()
    val_data, _ = p.get_validation_set()
    val_labels = p.get_validation_labels_list()
    p.clean_non_validation()

    mm = Masum_model(val_data, val_labels)
    mm.find_distances()
    print(mm.distances[0,0])
    print(mm.distances)

def prepare_basic_nn_model(data = "valid"):
    print("Parsing data...")
    p = Parser()
    print("Processing Attributes...")
    attrs = p.get_category_attrs()
    fn = ''
    if data == 'valid':
        print("getting validation set...")
        data, labels = p.get_validation_set()
        fn = 'valid'
    elif data == 'test':
        print("getting test set...")
        data, labels = p.get_test_set()
        fn = 'test'

    print("building model...")
    num_attrs = {i:20 for i in range(1,6)}
    model = basic_nn_model(attrs, num_attrs)
    model.import_data(data, labels)
    model.write_out_data(fn)

def train_basic_nn_model(cat, load_fn=None, start_epoch=0, learning_rate=.01, weights=[.001,.999], prev_train_loss=100):
    xfn = '../data/basic_nn/validX_'+str(cat) + ".csv"
    yfn = '../data/basic_nn/validY_'+str(cat) + ".csv"
    print("Importing training...")
    X = torch.from_numpy(pd.read_csv(xfn).to_numpy())
    Y = torch.from_numpy(pd.read_csv(yfn).to_numpy()).reshape(-1)
    xfn = '../data/basic_nn/testX_'+str(cat) + ".csv"
    yfn = '../data/basic_nn/testY_'+str(cat) + ".csv"
    print("Importing test...")
    X_test = torch.from_numpy(pd.read_csv(xfn).to_numpy())
    Y_test = torch.from_numpy(pd.read_csv(yfn).to_numpy()).reshape(-1)
    # a = X[:,0:20]
    # b = X[:,20:40]
    # c = X[:,40:60]
    # X = c + a*b*(c-1)
    # a = X_test[:,0:20]
    # b = X_test[:,20:40]
    # c = X_test[:,40:60]
    # X_test = c + a*b*(c-1)
    print("done")
    # train_set = MyDataset(X,Y)
    layer_dims = [X.shape[1],20,5]
    # print(layer_dims)
    # assert(False)
    print(X.shape, Y.shape)
    print(X_test.shape, Y_test.shape)
    model = Ebay_Cat_Classifier(layer_dims)
    if load_fn:
        model.load_state_dict(torch.load("../data/basic_nn/" + load_fn))
        # model.eval()
    _, train_loss, train_acc = run_model(model, "train", train_X=X,train_Y=Y, valid_X=X_test,valid_Y=Y_test,
                                                 batch_size=1000, learning_rate=learning_rate, n_epochs=200, start_epoch=start_epoch, weights=weights,
                                                 save_fn="_cat" + str(cat), prev_train_loss=prev_train_loss)

    plt.title('Training and Validation Loss')
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.plot(range(len(train_loss['train'])), train_loss['train'], color='green', label='Training')
    # plt.plot(range(len(train_loss['valid'])), train_loss['valid'], color='blue', label='Validation')
    plt.legend()
    plt.show()

    plt.title('Training and Validation Accuracy')
    plt.ylabel("Accuracy")
    plt.xlabel('Epoch')
    plt.plot(range(len(train_acc['train'])), train_acc['train'], color='green', label='Training')
    # plt.plot(range(len(dog_train_acc['valid'])), dog_train_acc['valid'], color='blue', label='Validation')
    plt.legend()
    plt.show()
    print('done')

def run_basic_nn_model():
    print("Parsing data, and extracting validation set...")
    p = Parser()
    attrs = p.get_category_attrs()
    # val_data, _ = p.get_validation_set()
    # test_data, _ = p.get_test_set()
    data = p.data_as_dict()
    # val_truth = cluster_to_pair(p.get_validation_clusters())
    # test_truth = cluster_to_pair(p.get_test_clusters())
    p.clean_non_validation()

    num_attrs = {i:20 for i in range(1,6)}
    basic_nn = basic_nn_model('run',attrs, num_attrs)

    # val_clusters = basic_nn.get_clusters(data, fn='basic_nn_clusters_test.tsv')

    # val_clusters = basic_nn.get_clusters(val_data, fn='basic_nn_clusters_valid.tsv')
    # val_proposed = list(val_clusters.items())
    # f1 = list_f1_score(val_proposed, val_truth)
    # print("validation f1:",f1)
    #
    # test_clusters = basic_nn.get_clusters(test_data, fn='basic_nn_clusters_test.tsv')
    # test_proposed = list(test_clusters.items())
    # f1 = list_f1_score(test_proposed, test_truth)
    # # basic_nn.save_clusters(test_clusters)
    # print("test f1:",f1)

    test_clusters = basic_nn.get_clusters(data)
    # basic_nn.save_clusters(test_clusters)

# run_basic_nn_model()