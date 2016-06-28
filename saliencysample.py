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
                           default="saliency_map_samples/samples10/")
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
    return coords


def plot_and_save_sample(imagepath, rows, cols, savepath):
    plot = plt.imread(imagepath)
    implot = plt.imshow(plot)
    scat = plt.scatter(x=cols, y=rows, c='r', s=30)
    plt.savefig(savepath)
    scat.remove()


def sample_in_bounds(center, patch_size, size=(240, 320)):
    row, col = center
    if row - patch_size / 2 < 0 or row + patch_size / 2 > size[0] or col - patch_size / 2 < 0 or col + patch_size / 2 > size[1]:
        return False
    else:
        return True

if __name__ == "__main__":
    args = parse_args()
    datapath = Path(args.dataset_path)
    savepath = Path(args.save_path)
    num_samples = 0
    num_out = 0
    for species in datapath.dirs():
        print species
        speciespath = Path(savepath + species.name)
        if not speciespath.exists():
            speciespath.mkdir()
        for img in species.files():
            with open(savepath + "finishedsamples10.txt", "r") as f:
                text = f.read()
                finished = set(text.split("\n"))
            id = get_image_id(img.name)
            print id
            if id in finished:
                print "Skipping %s" % id
                continue
            saliency_map = plt.imread(img)
            saliency_map = normalize(saliency_map)
            rows, cols = sample(saliency_map, 10)
            coordssavepath = Path(speciespath + "/" + id + ".npy")
            centers = save_sample(rows, cols, coordssavepath).T
            # for center in centers:
                # num_samples += 1
                # if not sample_in_bounds((center[0], center[1]), 80):
                    # num_out += 1
            with open(savepath + "finishedsamples10.txt", "a") as f:
                f.write(id + "\n")
    # print num_out, num_samples
