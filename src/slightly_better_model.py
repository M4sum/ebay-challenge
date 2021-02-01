import copy


class Slightly_Better_model:
    #baseline model, simply creating clusters based on shared attributes such as MPN, etc

    def __init__(self, data, key_attrs):
        self.data = data
        self.key_attrs = key_attrs
        self.attr_maps = {1:{},2:{},3:{},4:{},5:{}}
        # print(self.key_attrs)



    def test_attr(self, cat, attr):
        print("trying attribute",attr,"for cat",cat)
        new_key_attrs = copy.deepcopy(self.key_attrs)
        if attr in self.key_attrs[cat]:
            new_key_attrs[cat].remove(attr)
        else: new_key_attrs[cat].append(attr)
        return new_key_attrs, self.get_clusters(new_key_attrs)

    def get_clusters(self, key_attrs = {}):
        if key_attrs == {}: key_attrs = self.key_attrs
        clusters = {}
        for id in self.data:
            listing = self.data[id]
            cat = listing[0]
            attrs = listing[3]
            cluster = str(cat) + self.get_attr_string(key_attrs, cat, attrs)
            if cluster in clusters: clusters[cluster].append(id)
            else: clusters[cluster] = [cluster]
        return clusters

    # not used
    def build_attr_maps(self):
        for id in self.data:
            listing = self.data[id]
            cat = listing[0]
            attrs = listing[3]
            for attr in attrs:
                if attr not in self.attr_maps[cat]:
                    self.attr_maps[cat][attr] = []
                self.attr_maps[cat][attr].append(id)

    # Not used
    def get_attr_sub_clusters(self, clusters, cat, attr):
        if len(self.attr_maps[1]) == 0: self.build_attr_maps()
        attr_ids = self.attr_maps[cat][attr]
        new_clusters = {}
        for cluster in clusters:
            for id in clusters[cluster]:
                if id in attr_ids:
                    if cluster not in new_clusters: new_clusters[cluster] = []
                    new_clusters[cluster].append(id)
        return new_clusters

    def get_attr_string(self, key_attrs, cat, attrs):
        attr_str = ""
        for attr_name in key_attrs[cat]:
            if attr_name in attrs: attr_str += attrs[attr_name]
        return attr_str