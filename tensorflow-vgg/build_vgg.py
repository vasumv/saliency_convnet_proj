from argparse import ArgumentParser
from matplotlib import pyplot as plt
import tensorflow as tf
from path import Path
from nabirds import load_image_labels, load_train_test_split, load_image_paths
import numpy as np
import vgg16
import utils

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_path", default="../datasets/nabirdsresized/")
    argparser.add_argument("--sample_path", default="../saliency_map_samples/samples10/")
    argparser.add_argument("--save_path", default="patches/")
    return argparser.parse_args()


def create_patch(image, center, patch_size):

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
        return pad_image
    else:
        return tf.image.crop_to_bounding_box(image,
                                             offset_height, offset_width,
                                             patch_size, patch_size)

def get_image_id(label_id):
    return label_id.replace("-", "")
    
def get_label_id(image_id):
    label_id = image_id[:8] + "-" + image_id[8:12] + "-" + image_id[12:16] + "-" + image_id[16:20] + "-" + image_id[20:]
    return label_id 

def create_labels(patch_ids, image_labels):
    labels = [] 
    for id in patch_ids: 
	labels.append(int(image_labels[id]))
    return labels

if __name__ == "__main__":
    args = parse_args()
    datapath = Path(args.dataset_path)
    samplepath = Path(args.sample_path)
    savepath = Path(args.save_path)
    train_images, test_images = load_train_test_split()
    image_paths = load_image_paths()
    image_labels = load_image_labels()
    batch_size = 30
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	vgg = vgg16.Vgg16(555, 80)
	images = tf.placeholder("float", [batch_size, 224, 224, 3])
	with tf.name_scope("content_vgg"):
	     vgg.build(images)
	conv_6 = vgg.conv3_2
	conv_9 = vgg.conv4_2
	conv_12 = vgg.conv5_2
	print conv_6
	print conv_9
	print conv_12
	fc = vgg.fc_layer(conv_6, "fc1", True)
	logits = vgg.softmax_layer(fc, "softmax1", True)
	labels_placeholder = tf.placeholder(tf.int32, [batch_size])
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits, labels_placeholder, name='xentropy')
	loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
	opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	grads_and_vars = opt.compute_gradients(loss, var_list=tf.trainable_variables())
	init_op = tf.initialize_all_variables()
	train_op = opt.apply_gradients(grads_and_vars)
        training_paths = [x for x in image_paths.keys() if x in train_images]
	sess.run(init_op)
        for i in range(0, len(training_paths), 3):
	    patches = []
	    patch_ids = []
	    for img_id in training_paths[i:i+3]:
                print img_id
	        image = plt.imread(datapath + Path(image_paths[img_id]))
	        centers = np.load(samplepath + image_paths[img_id].split(".")[0] + ".npy").T
	        for i in range(len(centers)):
                    patch = create_patch(image, centers[i], 80).eval()
		    patch = utils.load_image(patch).reshape((1, 224, 224, 3))
		    patches.append(patch)
		    patch_ids.append(img_id)
            batch = np.concatenate(patches) 
            print "Batch shape: ", batch.shape
	    labels = create_labels(patch_ids, image_labels)
	    feed_dict = {images: batch, labels_placeholder: labels}
	    sess.run(train_op, feed_dict=feed_dict)
