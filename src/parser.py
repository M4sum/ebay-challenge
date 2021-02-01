import pandas as pd
import re

class Parser:
    def __init__(self, filepath = "../data/raw/mlchallenge_set_2021.tsv"):
        self.filepath = filepath
        self.new_data = []
        self.data_dict = {}
        self.validation_data = {}
        self.validation_labels = {}
        self.validation_clusters = {}
        self.validation_labels_list = []
        self.test_data = {}
        self.test_labels = {}
        self.test_clusters = {}
        self.test_labels_list = []

        self.attr_parse()

    def attr_parse(self):
        self.new_data = []
        f = open(self.filepath)
        i=0
        for row in f:
            # print(row)
            row = re.sub(r'\n', "", row)
            row = row.split("\t")
            # print(row)
            self.new_data.append(row)
            self.new_data[-1][0] = int(self.new_data[-1][0])

            images = row[2].split(";")
            if images[-1] == '': images = images[:-1]
            self.new_data[-1][2] = images

            x = row[3]
            # print(x)
            x = re.sub(r'^\(', "", x)
            x = re.sub(r'^"\(', "", x)
            # print(x)
            x = re.sub(r'\)$', "", x)
            x = re.sub(r'""\)"', "", x)
            x = re.sub(r'\)"', "", x)
            # print(x)
            x = re.split(":+", x)
            # print(x)
            attr = {}
            for i in range(len(x)):
                x[i] = re.split(",", x[i])
                for j in range(len(x[i])): x[i][j] = self.clean_data(x[i][j])
                # if row[-1] == '3613': print(x[i])
            # if row[-1] == '3613': print(x)
            for i in range(len(x) - 1):
                if i != len(x) - 2:
                    attr[x[i][-1]] = ','.join(x[i + 1][:-1])
                else:
                    attr[x[i][-1]] = ','.join(x[i + 1])
            self.new_data[-1][3] = attr
            # if row[-1] == '3613':
            #     print(row)
            #     print(x)
            #     print(attr)
            #     assert(False)
            # print()
            # i+=1
            # if i>20: assert(False)

        return self.new_data

    def clean_data(self, s):
        s = re.sub(r'\/+', "/", s)
        s = re.sub(r'\\+', "/", s)
        s = re.sub(r'"+', "", s)
        s = re.sub("\)", " ", s)
        s = re.sub("\(", " ", s)
        s = re.sub("#", " number ", s)
        s = re.sub("\'", "", s)
        s = re.sub("'", "", s)
        s = re.sub(r'\s+', " ", s)
        return s.strip().lower()

    def data_as_dict(self):
        if len(self.data_dict): return self.data_dict
        self.data_dict = {}
        for row in self.new_data:
            self.data_dict[row[-1]] = row[:-1]
        return self.data_dict

    def get_validation_set(self, validation_source = "../data/validation_labels.tsv"):
        self.data_as_dict()
        f = open(validation_source)
        for row in f:
            row = re.sub(r'\n', "", row)
            row = row.split("\t")
            val_ids = row[1].split(",")
            cluster = int(row[0])
            self.validation_clusters[cluster] = val_ids
            for id in val_ids:
                self.validation_data[id] = self.data_dict[id]
                self.validation_labels[id] = cluster
                self.validation_labels_list.append((id,cluster))
        return self.validation_data, self.validation_labels

    def get_validation_clusters(self):
        if len(self.validation_clusters) == 0: return False
        return self.validation_clusters

    def get_validation_labels_list(self):
        if len(self.validation_labels_list) == 0: return False
        return self.validation_labels_list

    def get_test_set(self, test_source = "../data/test_labels.tsv"):
        self.data_as_dict()
        f = open(test_source)
        for row in f:
            row = re.sub(r'\n', "", row)
            row = row.split("\t")
            test_ids = row[1].split(",")
            cluster = int(row[0])
            self.test_clusters[cluster] = test_ids
            for id in test_ids:
                self.test_data[id] = self.data_dict[id]
                self.test_labels[id] = cluster
                self.test_labels_list.append((id,cluster))
        return self.test_data, self.test_labels

    def get_test_clusters(self):
        if len(self.test_clusters) == 0: return False
        return self.test_clusters

    def get_test_labels_list(self):
        if len(self.test_labels_list) == 0: return False
        return self.test_labels_list

    def clean_non_validation(self):
        self.data_dict = {}
        self.new_data = []

    def get_category_attr_counts(self, cat_id):
        attrs = {'count':0}
        for row in self.new_data:
            if row[0] == cat_id:
                attrs['count'] +=1
                for attr in row[3].keys():
                    if attr in attrs: attrs[attr] +=1
                    else: attrs[attr] = 1
        return {k: v for k, v in sorted(attrs.items(), key=lambda item: item[1], reverse=True)}

    def get_category_attrs(self):
        # print(self.get_category_attr_counts(1))
        # print(self.get_category_attr_counts(2))
        # print(self.get_category_attr_counts(3))
        # print(self.get_category_attr_counts(4))
        # print(self.get_category_attr_counts(5))
        # assert(False)
        return {cat: list(self.get_category_attr_counts(cat).keys()) for cat in range(1,6)}

    def count_distinct_attr_vals(self, cat_id):
        attrs = {}
        for row in self.new_data:
            if row[0] == cat_id:
                for attr in row[3].keys():
                    if attr not in attrs: attrs[attr] = {}
                    val = row[3][attr]
                    if val not in attrs[attr]: attrs[attr][val] = 1
                    else: attrs[attr][val] += 1
        return {k: v for k, v in sorted(attrs.items(), key=lambda item: sum(item[1].values()), reverse=True)}
