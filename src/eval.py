def cluster_to_pair(clusters):
    # Convert dictionary of clusters {cluster id : list of product indices} to list of tuple pairs (product index, cluster id)
    pairs = []
    for k in clusters.keys():
        for v in clusters[k]:
            pairs.append((v, k))
    return pairs

def f1_score(proposed, truth):
    # proposed and truth are both lists of tuples: (index, clusterid)
    # Returns F1 score, or False if either precision or recall are incalculable
    # Indexes that do not exist in either truth or proposed will be ignored
    #
    # Actual cluster ids do not matter, what matters is whether the same ids are clustered in proposed and truth
    # Confusion matrix defined by:
    # D00: number of pairs with both clusterings having the listings not clustered together
    # D10: number of pairs with the true label clustering having the listings clustered together but the proposed clustering not having the listings clustered together
    # D01: number of pairs with the true label clustering not having the listings clustered together but the proposed clustering having the listings clustered together
    # D11: number of pairs with both clusterings having the listings clustered together

    D00 = D10 = D01 = D11 = 0
    proposed_pairs = {}
    for i,(id1, cluser1) in enumerate(proposed):
        for id2, cluster2 in proposed[i+1:]:
            proposed_pairs[(id1,id2)] = cluser1 == cluster2
    for i,(id1, cluser1) in enumerate(truth):
        for id2, cluster2 in truth[i+1:]:
            true_val = cluser1 == cluster2
            if (id1,id2) in proposed_pairs: proposed_val = proposed_pairs[(id1,id2)]
            elif (id2,id1) in proposed_pairs: proposed_val = proposed_pairs[(id2,id1)]
            else: continue      #edge case for pair not found

            # print(id1, id2, ":", true_val, proposed_val)
            if not true_val and not proposed_val: D00 += 1
            elif true_val and not proposed_val: D10 += 1
            elif not true_val and proposed_val: D01 += 1
            elif true_val and proposed_val: D11 += 1

    # print(D00, D10, D01, D11)
    if D11 + D01 == 0 or D11 + D10 == 0: return False   #edge case, incalculable P or R
    P = D11 / (D11 + D01)
    R = D11 / (D11 + D10)
    return 2*P*R / (P + R)
