from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from argparse import ArgumentParser
from PIL import Image
from path import Path
from scipy import misc
import numpy as np

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_path", default="datasets/nabirdsresized/")
    argparser.add_argument("--image_path", default="")
    argparser.add_argument("--save_path", default="datasets/nabirdspreprocessed/")
    return argparser.parse_args()


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
    i = 0
    image_sum = 0
    for species in imgpath.dirs():
        print species.name
        if not Path(savepath + species.name).exists():
                print "Making new directory..."
                newdir = savepath + species.name
                newdir.mkdir()
        for img in species.files():
            i += 1
            image = misc.imread(img)
            image_sum += image.mean()
    mean = image_sum * 1.0 / i
    print "Number of images: %d", i
    print "Sum of images: %s", image_sum
    print "Mean: %f", mean
    for species in imgpath.dirs():
        for img in species.files():
            print img.name
            image = misc.imread(img)
            image = image.astype(float)
            image -= mean
            misc.imsave(savepath + species.name + "/" + img.name, image)
            # image = Image.open(img)
            # image = image.resize((240, 320), Image.ANTIALIAS)
            # image.save(savepath + species.name + "/" + img.name)
