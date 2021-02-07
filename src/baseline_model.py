
class Baseline_model:
    #baseline model, simply creating clusters based on shared attributes

    def __init__(self, data, key_attrs, num_attrs = {1: 10, 2: 5, 3: 6, 4: 7, 5: 4}):
        self.data = data
        self.key_attrs = {cat: key_attrs[cat][1:num_attrs[cat]+2] for cat in range(1,6)}
        print(self.key_attrs)
        self.clusters = {}

    def get_clusters(self):
        if self.clusters: return self.clusters
        size = len(self.data)
        print("total data", size)
        i = 0
        for id in self.data:
            if i%10000 == 0: print(str(i), "of", size)
            listing = self.data[id]
            cat = listing[0]
            attrs = listing[3]
            cluster = str(cat) + self.get_attr_string(cat, attrs)
            if cluster in self.clusters: self.clusters[cluster].append(id)
            else: self.clusters[cluster] = [id]
            i+=1
        return self.clusters

    def get_pretty_clusters(self, clusters = None):
        if not clusters:
            clusters = self.get_clusters()
        pretty_clusters = {}
        cluster_counters = {'1':0,'2':0,'3':0,'4':0,'5':0}
        for cluster in clusters:
            cat = cluster[0]
            new_cluster = cat + str(cluster_counters[cat]).zfill(6)
            for id in clusters[cluster]:
                pretty_clusters[id] = new_cluster
            cluster_counters[cat] += 1
        print(cluster_counters)
        return pretty_clusters

    def save_clusters(self, clusters, fn = 'baseline_clusters.tsv'):
        with open('../data/'+fn, 'w') as f:
            s = ""
            for id in clusters:
                s += id + "\t" + clusters[id] + "\n"
            f.write(s)

    def get_attr_string(self, cat, attrs):
        attr_str = ""
        for attr_name in self.key_attrs[cat]:
            if attr_name in attrs: attr_str += attrs[attr_name]
        return attr_str