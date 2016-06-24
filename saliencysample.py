from matplotlib import pyplot as plt
from argparse import ArgumentParser
from path import Path
from scipy import misc
import numpy as np


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_path", default="../../tianyiwu/saliency-2016-cvpr/saliency_map/0342/1629aa0f4fa04f88aade70529692e841.png")
    argparser.add_argument("--image_path", default="images/")
    argparser.add_argument("--save_path", default="saliency_maps/0342/20samples/1629aa0f4fa04f88aade70529692e841")
    return argparser.parse_args()


def normalize(image):
    image = image.astype(float)
    image /= np.sum(image)
    return image


def sample(image, num):
    nums = np.random.choice(image.size, num, p=image.flatten())
    row_coords = [num / image.shape[1] for num in nums]
    col_coords = [num % image.shape[1] - 1 for num in nums]
    return row_coords, col_coords


if __name__ == "__main__":
    args = parse_args()
    datapath = Path(args.dataset_path)
    savepath = Path(args.save_path)
    if not savepath.exists():
        savepath.mkdir()
    saliency_map = misc.imread(datapath)
    saliency_map = normalize(saliency_map)
    for i in range(20):
        rows, cols = sample(saliency_map, 20)
        plot = plt.imread(datapath)
        implot = plt.imshow(plot)
        scat = plt.scatter(x=cols, y=rows, c='r', s=30)
        plt.savefig(Path(savepath + "/test%d.png") % i)
        scat.remove()
