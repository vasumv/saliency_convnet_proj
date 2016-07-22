from argparse import ArgumentParser
from matplotlib import pyplot as plt
import tensorflow as tf
from path import Path
from nabirds import load_image_labels, load_train_test_split, load_image_paths
from random import shuffle
import numpy as np
import vgg16
#import vgg16
import utils

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_path", default="../datasets/nabirdsresized/")
    argparser.add_argument("--sample_path", default="../saliency_map_samples/samples10/")
    argparser.add_argument("--save_path", default="patches/")
    return argparser.parse_args()


def create_patches(image, centers, patch_size):
    patches = []
    padding = int(patch_size / 2)
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'constant')
    for (y, x) in centers:
        y += padding
        x += padding
        patch = padded_image[y-padding:y+padding, x-padding:x+padding, :]
        patch = utils.load_image(patch).reshape((224, 224, 3))
        patches.append(patch)
    return patches

def get_image_id(label_id):
    return label_id.replace("-", "")
    
def get_label_id(image_id):
    label_id = image_id[:8] + "-" + image_id[8:12] + "-" + image_id[12:16] + "-" + image_id[16:20] + "-" + image_id[20:]
    return label_id 

def create_labels(patch_ids, image_labels, num_classes):
    labels = []
    for i in range(len(patch_ids)): 
	id = patch_ids[i]
        label_row = [0] * num_classes
	label_row[int(image_labels[id])] = 1
	labels.append(np.array(label_row))
    return np.vstack(labels) 

def sample_validation(train_images):
    shuffle(train_images)
    sub_training = train_images[:int(0.9 * len(train_images))] 
    validation = train_images[int(0.9 * len(train_images)):]
    return sub_training, validation

def create_sparse_labels(patch_ids, image_labels, num_classes):
    labels = []
    for i in range(len(patch_ids)): 
	id = patch_ids[i]
	labels.append(int(image_labels[id]))
    print len(labels)
    return np.array(labels) 

def check(path):
    return path[-3:] != "JPG"

def accuracy(predictions, labels):
    correct = 0.0
    for i in range(predictions.shape[0]):
        if np.argmax(predictions[i]) == np.argmax(labels[i]):
            correct += 1
    return 100.0 * (correct / predictions.shape[0]) 

if __name__ == "__main__":
    args = parse_args()
    datapath = Path(args.dataset_path)
    samplepath = Path(args.sample_path)
    summaries_dir = "summaries/"
    num_patches = 10
    num_epochs = 10
    num_classes = 555
    savepath = Path(args.save_path)
    train_images, test_images = load_train_test_split()
    image_paths = load_image_paths()
    image_labels = load_image_labels()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    batch_size = 30

    with tf.Session(config=tf.ConfigProto()) as sess:
        vgg = vgg16.Vgg16(555, 80)
        images = tf.placeholder("float", [batch_size, 224, 224, 3])
        with tf.name_scope("content_vgg"):
             vgg.build(images)
        conv_1 = vgg.conv1_1
        conv_6 = vgg.conv3_2
        conv_9 = vgg.conv4_2
        conv_12 = vgg.conv5_2
        batch_cnt = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.0001
        epoch_num = len(train_images) * num_patches
        learning_rate = tf.train.exponential_decay(starter_learning_rate, batch_cnt,
                                               epoch_num / batch_size, 0.9, staircase=True)
        drop_conv_12 = tf.nn.dropout(conv_12, 0.5)
        fc = vgg.fc_layer(drop_conv_12, "fc1", True)
        labels_placeholder = tf.placeholder(tf.float32, [batch_size, num_classes])
        with tf.name_scope('loss'):
            loss, logits = vgg.softmax_layer(fc, labels_placeholder,"softmax1", True)
            loss_summary = tf.scalar_summary('loss', loss)
        train_writer = tf.train.SummaryWriter(summaries_dir + '/train', graph=sess.graph)
        train_op = tf.train.RMSPropOptimizer(learning_rate, l1_regularization_strength=1.0,
                    l2_regularization_strength=1.0).minimize(loss, global_step=batch_cnt)
        init_op = tf.initialize_all_variables()
        training_paths = [x for x in image_paths.keys() if x in train_images and "JPG" not in image_paths[x]]
        sess.run(init_op)
        for epoch in range(num_epochs): 
            sum_cost = 0.0
            sum_accuracy = 0.0
            sub_training_paths, validation = sample_validation(training_paths[:-2])
            sub_length = len(sub_training_paths) - (len(sub_training_paths) % batch_size)
            for i in range(0, sub_length, batch_size):
                patches = []
                patch_ids = []
                for img_id in sub_training_paths[i:i+batch_size]:
                    image = plt.imread(datapath + Path(image_paths[img_id]))
                    image = utils.load_image(image).reshape((224, 224, 3))
                    patches.append(image)
                    patch_ids.extend([img_id])
                if len(patches) == 0:
                    continue
                batch = np.stack(patches)
                labels = create_labels(patch_ids, image_labels, num_classes)
                feed_dict = {images: batch, labels_placeholder: labels}
                cost, softmax, summary, _ = sess.run([loss, logits, loss_summary, train_op], feed_dict=feed_dict)
                train_accuracy = accuracy(softmax, labels)
                sum_cost += cost
                sum_accuracy += train_accuracy 
                train_writer.add_summary(summary, i)
                train_writer.flush()
                print "Epoch: ", epoch, "Iteration: ", i / 30, "Cost: ", cost, "Accuracy: ", train_accuracy 
                with open("build_vgg_log_images.txt", "a") as f:
                    f.write("Epoch: %d Iteration: %d Cost: %f Accuracy: %f \n" % (epoch, i / 30, cost, train_accuracy))
            print "Epoch: ", epoch, "Average epoch cost: ", sum_cost / (sub_length / batch_size)
            print "Epoch: ", epoch, "Average epoch accuracy: ", sum_accuracy / (sub_length / batch_size)
            with open("build_vgg_log_images.txt", "a") as f:
                f.write("Epoch: %d Average epoch cost: %f \n" % (epoch, sum_cost / (sub_length / batch_size)))
                f.write("Epoch: %d Average epoch accuracy: %f \n" % (epoch, sum_accuracy / (sub_length / batch_size)))
            sum_cost = 0.0
            sum_accuracy = 0.0
            valid_length = len(validation) - (len(validation) % batch_size)
            for i in range(0, valid_length, batch_size):
                patches = []
                patch_ids = []
                for img_id in validation[i:i+batch_size]:
                    image = plt.imread(datapath + Path(image_paths[img_id]))
                    image = utils.load_image(image).reshape((224, 224, 3))
                    patches.append(image)
                    patch_ids.extend([img_id])
                if len(patches) == 0:
                    continue
                batch = np.stack(patches)
                labels = create_labels(patch_ids, image_labels, num_classes)
                feed_dict = {images: batch, labels_placeholder: labels}
                cost, softmax, valid_summary, _ = sess.run([loss, logits, loss_summary], feed_dict=feed_dict)
                valid_accuracy = accuracy(softmax, labels)
                sum_cost += cost
                sum_accuracy += valid_accuracy 
                train_writer.add_summary(valid_summary, i)
                train_writer.flush()
                print "Epoch: ", epoch, "Iteration: ", i / 30, "Validation Cost: ", cost, "Validation Accuracy: ", valid_accuracy 
                with open("build_vgg_log_images.txt", "a") as f:
                    f.write("Epoch: %d Iteration: %d Valid Cost: %f Valid Accuracy: %f \n" % (epoch, i / 30, cost, valid_accuracy))
            print "Epoch: ", epoch, "Average epoch valid cost: ", sum_cost / (valid_length / batch_size)
            print "Epoch: ", epoch, "Average epoch valid accuracy: ", sum_accuracy / (valid_length / batch_size)
            with open("build_vgg_log_images.txt", "a") as f:
                f.write("Epoch: %d Average epoch valid cost: %f \n" % (epoch, sum_cost / (valid_length / batch_size)))
                f.write("Epoch: %d Average epoch valid accuracy: %f \n" % (epoch, sum_accuracy / (valid_length / batch_size)))
            sum_cost = 0.0
            sum_accuracy = 0.0
            test_length = len(test_images) - (len(test_images) % batch_size)
            for i in range(0, test_length, batch_size):
                patches = []
                patch_ids = []
                for img_id in test_images[i:i+batch_size]:
                    image = plt.imread(datapath + Path(image_paths[img_id]))
                    image = utils.load_image(image).reshape((224, 224, 3))
                    patches.append(image)
                    patch_ids.extend([img_id])
                if len(patches) == 0:
                    continue
                batch = np.stack(patches)
                labels = create_labels(patch_ids, image_labels, num_classes)
                feed_dict = {images: batch, labels_placeholder: labels}
                cost, softmax, test_summary, _ = sess.run([loss, logits, loss_summary], feed_dict=feed_dict)
                test_accuracy = accuracy(softmax, labels)
                sum_cost += cost
                sum_accuracy += test_accuracy 
                train_writer.add_summary(test_summary, i)
                train_writer.flush()
                print "Epoch: ", epoch, "Iteration: ", i / 30, "Test Cost: ", cost, "Test Accuracy: ", test_accuracy 
                with open("build_vgg_log_images.txt", "a") as f:
                    f.write("Epoch: %d Iteration: %d Test Cost: %f Test Accuracy: %f \n" % (epoch, i / 30, cost, test_accuracy))
            print "Epoch: ", epoch, "Average epoch test cost: ", sum_cost / (test_length / batch_size)
            print "Epoch: ", epoch, "Average epoch test accuracy: ", sum_accuracy / (test_length / batch_size)
            with open("build_vgg_log_images.txt", "a") as f:
                f.write("Epoch: %d Average epoch test cost: %f \n" % (epoch, sum_cost / (test_length / batch_size)))
                f.write("Epoch: %d Average epoch test accuracy: %f \n" % (epoch, sum_accuracy / (test_length / batch_size)))
            shuffle(test_images)
