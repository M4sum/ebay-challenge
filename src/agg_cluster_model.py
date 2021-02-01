import heapq as hq
import math

from src.eval import *


class agg_cluster_model:

    def __init__(self, images_nn, att_distance_measure, image_weight=1, att_weight=1):
        # images_nn is a trained neural network, trained to identify if any 2 pictures are from the same listing
        # nlp is a model used to identify whether 2 sets of descriptions are the same, or different
        self.images_nn = images_nn
        self.att_distance_measure = att_distance_measure
        self.image_weight = image_weight
        self.att_weight = att_weight

        self.closestData = []
        self.clusters = {}
        self.keepN = 100000

    def dist_between(self, listing1, listing2):
        # listing1 and listing2 are rows, with the format:
        # category id, primary picture url, additional picture urls, attributes, index
        # att and image distances must be negative, for the heap sorting to work properly
        # print(listing1)
        # print(listing2)
        image_dist = self.get_image_dist(listing1[2].append(listing1[1]), listing2[2].append(listing2[1]))
        att_dist = self.att_distance_measure.get_dist(listing1[3], listing2[3])
        return self.image_weight * image_dist + self.att_weight * att_dist

    def get_image_dist(self, images1, images2):
        # matches = 0
        # for image1 in images1:
        #     for image2 in images2:
        #         is_match = self.images_nn.match(image1, image2)
        #         if is_match: matches += 1
        # return matches / (len(images1) * len(images2))
        return 0

    def loadData(self, data):
        self.closestData, self.clusters = [], {}
        ids = list(data.keys())
        for i, id1 in enumerate(ids):
            if i%100 == 0: print(i, "of", len(data))
            listing1 = data[id1]
            self.clusters[id1] = id1    #cluster ids dont matter, so initialize with indexs
            for id2 in ids[i+1:]:
                listing2 = data[id2]
                if id1 == id2 or listing1[0] != listing2[0]: continue   #don't compare a listing to itself, or to another category
                tup = (self.dist_between(listing1,listing2),(id1,id2))
                if len(self.closestData) <= self.keepN: hq.heappush(self.closestData,tup)
                else: hq.heapreplace(self.closestData, tup)

    def train(self, validation_labels, learning_rate = 1000):
        prev_f1, f1 = 0, 0
        i = 0
        while True:
            i += 1
            # print(self.closestData)
            # assert(False)
            self.merge_closest(learning_rate)
            f1 = list_f1_score(self.get_clusters(), validation_labels)
            print("trained ", i, "learning rate:", learning_rate, ", f1:",f1)
            if learning_rate == 0: break
            elif f1 < prev_f1:
                self.undo_last_merge()
                learning_rate = math.floor(learning_rate/1)
            else: prev_f1 = f1
        return prev_f1


    def merge_closest(self, n=1):
        largest = hq.nlargest(n,self.closestData)
        self.last_merge = []
        self.last_merge_values = {}
        for l in largest:
            (dist, (id1, id2)) = l
            self.last_merge.append(l)
            self.last_merge_values[id2] = self.clusters[id2]
            self.clusters[id2] = self.clusters[id1]
            self.closestData.remove(l)

    def undo_last_merge(self):
        for l in self.last_merge:
            (dist, (id1, id2)) = l
            self.clusters[id2] = self.last_merge_values[id2]
            hq.heappush(self.closestData, l)

    def get_clusters(self):
        return [(k, v) for k, v in self.clusters.items()]