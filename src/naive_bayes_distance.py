import numpy as np

class naive_bayes_distance:
    #problem: doesn't penalize attributes existing in one listing but not another

    def __init__(self):
        self.same_and_clustered_counts = {}
        self.same_counts = {}
        self.clustered_counts = {}
        self.total_counts = {}

        self.total_clustered_pairs = 0
        self.total_pairs = 0


    def train(self, validation_set, validation_labels_list):
        i = 0
        for i,(id1, cluster1) in enumerate(validation_labels_list):
            i += 1
            if i%100 == 0: print(i,"of",len(validation_labels_list))
            attrs1 = validation_set[id1][3]
            for (id2, cluster2) in validation_labels_list[i+1:]:

                self.total_pairs += 1
                clustered = cluster1 == cluster2
                if clustered: self.total_clustered_pairs += 1

                attrs2 = validation_set[id2][3]
                for attr in attrs1:
                    if attr in attrs2:
                        self.__inc(self.total_counts, attr)
                        same_val = attrs1[attr] == attrs2[attr]
                        if clustered: self.__inc(self.clustered_counts, attr)
                        if same_val: self.__inc(self.same_counts, attr)
                        if clustered and same_val: self.__inc(self.same_and_clustered_counts, attr)

    def __inc(self, dict, val):
        if val in dict: dict[val] += 1
        else: dict[val] = 1

    def get_dist(self, attrs1, attrs2):
        total_p = self.total_clustered_pairs / self.total_pairs     #prior
        for attr in attrs1:
            if attr in attrs2:
                same_val = attrs1[attr] == attrs2[attr]

                if attr not in self.same_and_clustered_counts: p_same_given_cluster = 1/len(self.total_counts)
                else: p_same_given_cluster = (self.same_and_clustered_counts[attr]+1) / (self.clustered_counts[attr] + len(self.total_counts))

                if attr not in self.same_counts: p_same = 1 / len(self.total_counts)
                else: p_same = (self.same_counts[attr]+1) / (self.total_counts[attr] + len(self.total_counts))

                if same_val: p = p_same_given_cluster / p_same
                else:  p = (1 - p_same_given_cluster) / (1 - p_same)
                total_p *= p
        dist = np.log(total_p)
        # if dist > -1 :
        #     print("comparing:")
        #     print(attrs1)
        #     print(attrs2)
        #     print("dist:",dist)
        return dist