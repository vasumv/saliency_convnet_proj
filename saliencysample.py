from matplotlib import pyplot as plt
from argparse import ArgumentParser
from path import Path
import numpy as np


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_path",
                           default="../../tianyiwu/saliency-2016-cvpr/saliency_map/")
    argparser.add_argument("--image_path", default="images/")
    argparser.add_argument("--save_path",
                           default="saliency_map_samples/")
    return argparser.parse_args()


def normalize(image):
    image = image.astype(float)
    image /= np.sum(image)
    return image


def sample(image, num):
    nums = np.random.choice(image.size, num, p=image.flatten())
    row_coords = [num1 / image.shape[1] for num1 in nums]
    col_coords = [num1 % image.shape[1] - 1 for num1 in nums]
    return row_coords, col_coords


def get_image_id(imagename):
    return imagename.split(".")[0]


def save_sample(rows, cols, savepath):
    coords = np.vstack((rows, cols))
    np.save(savepath, coords)


def plot_and_save_sample(imagepath, rows, cols, savepath):
    plot = plt.imread(imagepath)
    implot = plt.imshow(plot)
    scat = plt.scatter(x=cols, y=rows, c='r', s=30)
    plt.savefig(savepath)
    scat.remove()


if __name__ == "__main__":
    args = parse_args()
    datapath = Path(args.dataset_path)
    savepath = Path(args.save_path)
    for species in datapath.dirs():
        print species
        speciespath = Path(savepath + species.name)
        if not speciespath.exists():
            speciespath.mkdir()
        for img in species.files():
            with open(savepath + "finishedsamples.txt", "r") as f:
                text = f.read()
                finished = set(text.split("\n"))
            id = get_image_id(img.name)
            print id
            if id in finished:
                print "Skipping %s" % id
                continue
            saliency_map = plt.imread(img)
            saliency_map = normalize(saliency_map)
            rows, cols = sample(saliency_map, 20)
            coordssavepath = Path(speciespath + "/" + id + ".npy")
            # imgsavepath = Path(speciespath + "/" + id + ".png")
            save_sample(rows, cols, coordssavepath)
            # plot_and_save_sample(img, rows, cols, imgsavepath)
            with open(savepath + "finishedsamples.txt", "a") as f:
                f.write(id + "\n")
