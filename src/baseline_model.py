
class Baseline_model:
    #baseline model, simply creating clusters based on shared attributes such as MPN, etc

    def __init__(self, data, key_attrs, num_attrs = 5):
        self.data = data
        self.key_attrs = {cat: key_attrs[cat][1:num_attrs+2] for cat in range(1,6)}
        print(self.key_attrs)

    def get_clusters(self):
        clusters = {}
        for id in self.data:
            listing = self.data[id]
            cat = listing[0]
            attrs = listing[3]
            cluster = str(cat) + self.get_attr_string(cat, attrs)
            if cluster in clusters: clusters[cluster].append(id)
            else: clusters[cluster] = [cluster]
        return clusters

    def get_attr_string(self, cat, attrs):
        attr_str = ""
        for attr_name in self.key_attrs[cat]:
            if attr_name in attrs: attr_str += attrs[attr_name]
        return attr_str