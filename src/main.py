from src.baseline_model import Baseline_model
from src.eval import *
from src.parser import Parser


def run_baseline_model():
    p = Parser("../data/raw/mlchallenge_set_2021.tsv")
    attrs = p.get_category_attrs()
    for cat in attrs:
        print(cat, attrs[cat])
    val_data = p.get_validation_set("../data/validation_labels.tsv")
    p.clean_non_validation()

    #gets max accuracy at 9 attributes, .4594
    for num_attrs in range(5,11):
        model = Baseline_model(val_data, attrs, num_attrs)
        proposed = model.get_clusters()
        truth = p.validation_labels
        print(num_attrs,"attrs:",f1_score(proposed, truth))

run_baseline_model()