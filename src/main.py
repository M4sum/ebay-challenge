from src.downloader import *
from src.run_models import *


def print_attr_vals(total, attrs, attr_name):
    attr = attrs[attr_name]
    sortd = {k: v for k, v in sorted(attr.items(), key=lambda item: item[1], reverse=True)}
    total_listings = sum(sortd.values())
    print("Attr:",attr_name)
    print("total listings:",total_listings,"/",total)
    # print(sorted)
    num_cats = 0
    culm = 0
    for key in sortd:
        if sortd[key] > 1000:
            num_cats+=1
            culm+=sortd[key]
            print(key,",",sortd[key],",",sortd[key]/total_listings*100)
    print("")
    print("num cats:",num_cats)
    print("perc covered:",100*culm/total_listings)
    print("perc unknown:",100*(total-total_listings)/total)
    print("")


run_basic_nn_model()

# run_baseline_model()
# run_slightly_better_model()
# run_agg_cluster_model()
# run_basic_nn_model('valid')
# train_basic_nn_model(3, weights=[.001,.999])
# train_basic_nn_model(3, start_epoch=113, load_fn="nn_save_cat3_epoch112.pt", learning_rate=.001, weights=[0.04645254944656142, 0.9535474505534386])
# train_basic_nn_model(3, load_fn="nn_save_cat3_epoch86.pt", start_epoch=87, learning_rate=8.39808e-05, weights=[0.16953642384105955, 0.8304635761589405])
# train_basic_nn_model(2)
# train_basic_nn_model(2, load_fn="nn_save_cat2_final.pt", start_epoch=1, learning_rate=0.001, weights=[0.05683836589698044, 0.9431616341030196])
# train_basic_nn_model(1, load_fn="nn_save_cat1_epoch165.pt", start_epoch=166, learning_rate=.003, weights=[0.25321463897131546, 0.7467853610286845], prev_train_loss=4.438651289500494e-05)
# train_basic_nn_model(3, learning_rate=.005)
# train_basic_nn_model(4, load_fn="nn_save_cat4_epoch59.pt", learning_rate=.0005, weights=[.1,.9])
# train_basic_nn_model(5, load_fn="nn_save_cat5_epoch178.pt", learning_rate=0.05, weights=[0.25321463897131546, 0.7467853610286845])

# test_baseline_model()


# print("Parsing data...")
# p = Parser()
# print("Processing Attributes...")
# # for i in range(1,6):
# #     print("Checking attribute",i)
# #     attrs = p.count_distinct_attr_vals(i)
# #     j = 0
# #     for attr in attrs:
# #         j+=1
# #         if j > 20: break
# #         print(attr,":",len(attrs[attr]),"values")
# #     print("")
#
#
# for listing in p.new_data:
#     if listing[0] != 1: continue
#     attrs = listing[3]
#     if 'length' in attrs:
#         download_if_not_exists(listing)
#     # break

# attrs = p.count_distinct_attr_vals(1)
# total = len(p.new_data)
# print_attr_vals(total, attrs, 'brand')
# print_attr_vals(total, attrs, 'size type')
# print_attr_vals(total, attrs, 'bottoms size womens')
# print_attr_vals(total, attrs, 'material')
# print_attr_vals(total, attrs, 'inseam')
# print_attr_vals(total, attrs, 'color')
# print_attr_vals(total, attrs, 'rise')
# print_attr_vals(total, attrs, 'style')
# print_attr_vals(total, attrs, 'silhouette')
# print_attr_vals(total, attrs, 'country/region of manufacture')
# print_attr_vals(total, attrs, 'wash')
# print_attr_vals(total, attrs, 'treatment')
# print_attr_vals(total, attrs, 'length')
# print_attr_vals(total, attrs, 'features')
# print_attr_vals(total, attrs, 'pattern')
# print_attr_vals(total, attrs, 'garment care')
# print_attr_vals(total, attrs, 'model')
# print_attr_vals(total, attrs, 'accents')
