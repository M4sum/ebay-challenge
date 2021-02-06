from src.baseline_model import Baseline_model
from src.eval import *
from src.parser import Parser

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
    model.get_clusters()
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

    num_attrs = {1: 10, 2: 5, 3: 6, 4: 7, 5: 4}
    f1 = 0
    for cat in range(1,6):
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
