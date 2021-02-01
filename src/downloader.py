import time
import urllib.request
import os
import ssl
import imageio
import requests

def download_if_not_exists(listing):
    cat = listing[0]
    index = listing[-1]
    folder = "../data/images/"+str(cat)+"/"+index + "/"
    if not os.path.exists(folder): download_all(listing)


def download_all(listing):
    # print(listing)
    cat = listing[0]
    index = listing[-1]
    prime_img = listing[1]
    imgs = listing[2]
    fetch_img(cat, index, 0, prime_img)
    for i,url in enumerate(imgs):
        fetch_img(cat, index, i+1, url)


def fetch_img(cat, index, i, url):
    folder = "../data/images/"+str(cat)+"/"+index + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        with open(folder + str(i + 1) + ".jpg", 'wb') as f:
            f.write(readimg(url))
    except:
        print("An fetching exception occurred!!!")
        time.sleep(1)
        fetch_img(cat, index, i, url)

def readimg(url):
    context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(url, context=context) as response:
            return response.read()
    except:
        print("An exception occurred!!!")
        time.sleep(1)
        readimg(url)

# listing = [2, 'http://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/iYYAAOxydgZTJwYc/$_1.JPG?set_id=880000500F', ['http://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/iYYAAOxydgZTJwYc/$_1.JPG?set_id=880000500F', 'http://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/M-4AAMXQPatTJwYR/$_1.JPG?set_id=880000500F', 'http://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/gRkAAOxy3HJTJwYV/$_1.JPG?set_id=880000500F', 'http://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/nOkAAOxykMpTJwYe/$_1.JPG?set_id=880000500F', 'http://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/yM0AAOxyf1dTJwYl/$_1.JPG?set_id=880000500F', 'http://i.ebayimg.com/00/s/MTYwMFgxMjAw/z/Ua0AAOxyUI1TJwYs/$_1.JPG?set_id=880000500F', 'http://i.ebayimg.com/00/s/MTIwMFgxNjAw/z/NiwAAMXQPatTJwYt/$_1.JPG?set_id=880000500F'], {'brand': 'shimano', 'us shoe size mens': '4.5', 'modified item': 'no', 'style': 'cycling'}, '0']
# listing = [1, 'https://i.ebayimg.com/00/s/OTk5WDc0OQ==/z/xdEAAOSwCLVbxhE1/$_12.JPG?set_id=880000500F', ['https://i.ebayimg.com/00/s/OTk5WDc0OQ==/z/xdEAAOSwCLVbxhE1/$_12.JPG?set_id=880000500F', 'https://i.ebayimg.com/00/s/OTk5WDc0OQ==/z/YcYAAOSwu7xbxhE4/$_12.JPG?set_id=880000500F', 'https://i.ebayimg.com/00/s/OTEzWDY3MQ==/z/SFkAAOSwWhVbxhE4/$_12.JPG?set_id=880000500F', 'https://i.ebayimg.com/00/s/OTk5WDc0OQ==/z/N8wAAOSw7SFbxhE5/$_12.JPG?set_id=880000500F', 'https://i.ebayimg.com/00/s/OTk5WDc0OQ==/z/cNAAAOSworhbxhE6/$_12.JPG?set_id=880000500F', 'https://i.ebayimg.com/00/s/OTk5WDc0OQ==/z/rz4AAOSwhZ5bxhE7/$_12.JPG?set_id=880000500F', 'https://i.ebayimg.com/00/s/OTk5WDc0OQ==/z/eysAAOSwd2ZbxhE7/$_12.JPG?set_id=880000500F', 'https://i.ebayimg.com/00/s/OTk5WDc0OQ==/z/4VEAAOSwJ6NbxhE8/$_12.JPG?set_id=880000500F', 'https://i.ebayimg.com/00/s/OTk5WDc0OQ==/z/EFwAAOSwb8VbxhE9/$_12.JPG?set_id=880000500F', 'https://i.ebayimg.com/00/s/OTk5WDc0OQ==/z/RQIAAOSwq9RbxhE-/$_12.JPG?set_id=880000500F'], {'brand': 'joes jeans', 'material': 'cotton blend', 'model': 'curvy bootcut', 'size type': 'regular', 'silhouette': 'bootcut', 'inseam': '30 in.', 'length': 'full length', 'color': 'blue', 'bottoms size womens': '28', 'treatment': 'medium wash', 'features': 'curvy', 'country/region of manufacture': 'united states'}, '53819']
# download_all(listing)