from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from argparse import ArgumentParser
from PIL import Image
from path import Path
from scipy import misc
import numpy as np
import random
import os
import sys

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_path", default="../yashu/datasets/nabirds/")
    argparser.add_argument("--image_path", default="images/")
    argparser.add_argument("--save_path", default="datasets/nabirdsresized/")
    return argparser.parse_args()

def load_image_paths(dataset_path='', path_prefix=''):
    paths = {}
    print dataset_path
    with open(os.path.join(dataset_path, 'images.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(path_prefix, pieces[1])
            paths[image_id] = path

    return paths

def mean_sub_and_normalize(img):
    img = img.astype(float)
    img -= np.mean(img, axis=0)
    img /= np.std(img, axis=0)
    return img

if __name__ == "__main__":
    args = parse_args()
    datapath = Path(args.dataset_path)
    imgpath = datapath + Path(args.image_path)
    savepath = Path(args.save_path)
    print len(imgpath.dirs())
    for species in imgpath.dirs():
        if not Path(savepath + species.name).exists():
                print "Making new directory..."
                newdir = savepath + species.name
                newdir.mkdir()
        else:
            print "Skipping " + species.name
            continue
        for img in species.files():
            print img.name
            image = Image.open(img)
            image = image.resize((256, 256), Image.ANTIALIAS)
            image.save(savepath + species.name + "/" + img.name)
