from argparse import ArgumentParser
from matplotlib import pyplot as plt
import tensorflow as tf
from path import Path
import numpy as np
import vgg16
import utils

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_path", default="../datasets/nabirdsresized")
    argparser.add_argument("--sample_path", default="../saliency_map_samples/samples10/")
    return argparser.parse_args()


def create_patch(image, center, patch_size, tfsession):

    x1 = center[1] - patch_size / 2
    y1 = center[0] - patch_size / 2
    x_diff = 0
    y_diff = 0
    offset_width = x1
    offset_height = y1
    if x1 < 0 or x1 + patch_size > image.shape[1] or y1 < 0 or y1 + patch_size > image.shape[0]:
        pad_width = 0
        pad_height = 0
        if x1 < 0:
            offset_width = 0
            x_diff = x1
            pad_width = -x1
        elif x1 + patch_size > image.shape[1]:
            offset_width = x1
            x_diff = image.shape[1] - (x1 + patch_size)
            pad_width = 0
        if y1 < 0:
            offset_height = 0
            y_diff = y1
            pad_height = -y1
        elif y1 + patch_size > image.shape[0]:
            offset_height = y1
            y_diff = image.shape[0] - (y1 + patch_size)
            pad_height = 0
        crop_image = tf.image.crop_to_bounding_box(image, offset_height,
                                                   offset_width,
                                                   patch_size + y_diff,
                                                   patch_size + x_diff)
        pad_image = tf.image.pad_to_bounding_box(crop_image, pad_height,
                                                 pad_width, patch_size,
                                                 patch_size)
        return pad_image.eval(session=tfsession)
    else:
        return tf.image.crop_to_bounding_box(image,
                                             offset_height, offset_width,
                                             patch_size, patch_size).eval(session=tfsession)

def get_image_id(imagename):
    return imagename.split(".")[0]

if __name__ == "__main__":
    args = parse_args()
    datapath = Path(args.dataset_path)
    samplepath = Path(args.sample_path)

    with tf.Session(
            config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
        batches = []
        for folder in datapath.dirs()[:1]:
            print folder.name
            for img in folder.files():
                id = get_image_id(img.name)
                print id
                image = plt.imread(img)
                centers = np.load(samplepath + folder.name + "/" + id + ".npy").T
                patches = [create_patch(image, center, 80, sess) for center in centers]
                for patch in patches:
                    batches.append(utils.load_image(patch))
        batch = batches[0].reshape((1, 224, 224, 3))
        for b in batches[1:]:
            batch = np.concatenate((batch, b.reshape((1, 224, 224, 3))), 0)
            images = tf.placeholder("float", [batch.shape[0], 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)

