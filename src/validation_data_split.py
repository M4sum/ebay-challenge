import csv
import random

n = 0
clusters = {}
with open("../data/raw/mlchallenge_set_validation.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        n += 1
        if line[1] in clusters.keys():
            clusters[line[1]].append(line[0])
        else:
            clusters.update({line[1]:[line[0]]})
print("Read {} lines".format(n))

def get_val_test_split(c, p):
    val_num = int(p*len(list(c.keys())))
    test = []
    for k, v in c.items():
        test.append([k, v])
    val = []
    while len(val) < val_num:
        i = int(random.random() * len(test))
        val.append(test[i])
        test.remove(test[i])
    return val, test
    
def print_dist_vals_dict(c):
    vals = {}
    for k in list(c.keys()):
        l = len(c[k])
        if l in vals.keys():
            vals[l] += 1
        else:
            vals.update({l:1})
    print(vals)
    
    n = 0
    for k in list(vals.keys()):
        n += vals[k] * k
    print(n)

def print_dist_vals_list(c):
    vals = {}
    for ci in c:
        l = len(ci[1])
        if l in vals:
            vals[l] += 1
        else:
            vals.update({l:1})
    print(vals)
    
    n = 0
    for k in list(vals.keys()):
        n += vals[k]
    print(n)

def save_file(val, test):
    with open('../data/validation_labels.tsv', 'w') as f:
        s = ""
        for v in val:
            idstr = ",".join(v[1])
            s += str(v[0]) + "\t" + idstr + "\n"
        f.write(s)
    
    with open('../data/test_labels.tsv', 'w') as f:
        s = ""
        for t in test:
            idstr = ",".join(t[1])
            s += str(t[0]) + "\t" + idstr + "\n"
        f.write(s)
    
print_dist_vals_dict(clusters)
v, t = get_val_test_split(clusters, 0.7)
print("val and test sizes: ", len(v),len(t))
print("val")
print_dist_vals_list(v)
print("test")
print_dist_vals_list(t)

save = input("Save file? [y/n]")
if save == 'y':
    save_file(v, t)