import itertools
import re
from difflib import SequenceMatcher
import time


class Masum_model:

    def __init__(self, data, labels):
        self.data = data
        self.lables = labels

    def find_distances(self):
        self.distances = self.__find_dist(self.data)
        return self.distances

    def __find_dist(self, listings):
        a = b = 0
        distances1 = {}
        # for i, j in itertools.product(df1['attributes'], df1['attributes']):
        print("finding distances...")
        keys = list(listings.keys())
        for i in range(len(keys)):
            if i%10 == 0: print(i,"of",len(keys))
            for j in range(i+1,len(keys)):
                l1 = listings[keys[i]][3]
                l2 = listings[keys[j]][3]
                for attr in l1.keys():
                    is_common = 0
                    t1 = time.time()
                    if attr in l2.keys() and SequenceMatcher(None, l1[attr], l2[attr]).ratio() > 0.5:
                        is_common += 1
                a += 1
                if a == len(listings) - 1:
                    b += 1
                distances1[(a,b)] = ((is_common + 1) / (len(l1) + 1)) * ((is_common + 1) / (len(l2) + 1))