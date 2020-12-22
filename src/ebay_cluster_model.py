
class ebay_cluster_model:

    def __init__(self, images_nn, nlp):
        # images_nn is a trained neural network, trained to identify if any 2 pictures are from the same listing
        # nlp is a model used to identify whether 2 sets of descriptions are the same, or different
        self.images_nn = images_nn
        self.nlp = nlp

    def dist_between(listing1, listing2):
        # listing1 and listing2 are rows, with the format:
        # category id, primary picture url, additional picture urls, attributes, index
        pass
