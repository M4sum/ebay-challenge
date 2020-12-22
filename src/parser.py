import pandas as pd
import re

class Parser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.new_data = []
        self.data_dict = {}
        self.validation_data = {}
        self.validation_labels = {}

        self.attr_parse()

    def attr_parse(self):
        self.new_data = []
        f = open(self.filepath)
        for row in f:
            row = re.sub(r'\n', "", row)
            row = row.split("\t")
            self.new_data.append(row)
            self.new_data[-1][0] = int(self.new_data[-1][0])

            images = row[2].split(";")
            if images[-1] == '': images = images[:-1]
            self.new_data[-1][2] = images

            x = row[3]
            x = re.sub(r'^\(', "", x)
            x = re.sub(r'\)$', "", x)
            x = re.split(":+", x)
            attr = {}
            for i in range(len(x)):
                x[i] = re.split(",", x[i])
            for i in range(len(x) - 1):
                if i != len(x) - 2:
                    attr[x[i][-1]] = ','.join(x[i + 1][:-1])
                else:
                    attr[x[i][-1]] = ','.join(x[i + 1])
            self.new_data[-1][3] = attr

        return self.new_data

    def data_as_dict(self):
        if len(self.data_dict): return self.data_dict
        self.data_dict = {}
        for row in self.new_data:
            self.data_dict[row[-1]] = row[:-1]

    def get_validation_set(self, validation_source):
        self.data_as_dict()
        f = open(validation_source)
        for row in f:
            row = re.sub(r'\n', "", row)
            row = row.split("\t")
            val_ids = row[1].split(",")
            self.validation_labels[int(row[0])] = val_ids
            for id in val_ids:
                self.validation_data[id] = self.data_dict[id]
        return self.validation_data

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
        return {cat: list(self.get_category_attr_counts(cat).keys()) for cat in range(1,6)}
