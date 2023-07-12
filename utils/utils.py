import numpy as np
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
from sklearn.metrics import confusion_matrix
import math
import os
import cv2
from pathlib import Path

# Prints the number of parameters of a model
def get_params(model):
    # Init models (variables and input shape)
    total_parameters = 0
    for variable in model.variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    print("Total parameters of the net: " + str(total_parameters) + " == " + str(total_parameters / 1000000.0) + "M")

# preprocess a batch of images
def preprocess(x, mode='imagenet'):
    if mode is None: return x

    if 'imagenet' in mode:
        return tf.keras.applications.xception.preprocess_input(x)
    elif 'normalize' in mode:
        return  x.astype(np.float32) / (255/2) - 1

# applies to a lerarning rate tensor (lr) a decay schedule, the polynomial decay
def lr_decay(lr, init_learning_rate, end_learning_rate, epoch, total_epochs, power=0.9):
    lr.assign(
        (init_learning_rate - end_learning_rate) * math.pow(1 - epoch / 1. / total_epochs, power) + end_learning_rate)

# converts a list of arrays into a list of tensors
def convert_to_tensors(list_to_convert):
    if list_to_convert != []:
        return [tf.convert_to_tensor(list_to_convert[0])] + convert_to_tensors(list_to_convert[1:])
    else:
        return []

# restores a checkpoint model
def restore_state(saver, checkpoint):
    try:
        saver.restore(None, checkpoint)
        print('Model loaded')
    except Exception as e:
        print('Model not loaded: ' + str(e))

# inits a models (set input)
def init_model(model, input_shape):
    model._set_inputs(np.zeros(input_shape))


# Erase the elements if they are from ignore class. returns the labesl and predictions with no ignore labels
def erase_ignore_pixels(labels, predictions, mask):
    indices = tf.squeeze(tf.where(tf.greater(mask, 0)))  # not ignore labels
    labels = tf.cast(tf.gather(labels, indices), tf.int64)
    predictions = tf.gather(predictions, indices)

    return labels, predictions

# generate and write an image into the disk
def generate_image(image_scores, output_dir, dataset, loader, train=False, suffix=""):
    # Get image name
    if train:
        list = loader.image_train_list
        index = loader.index_train
    else:
        list = loader.image_test_list
        index = loader.index_test

    dataset_name = dataset.split('/')
    if dataset_name[-1] != '':
        dataset_name = dataset_name[-1]
    else:
        dataset_name = dataset_name[-2]

    # Get output dir name
    out_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # write it
    image = np.argmax(image_scores, axis=-1).astype(float)
    image *= 255/image_scores.shape[-1]

    # this is the shittiest path handling ive ever seen ffs
    # name_split = list[index - 1].split('/')[-1].split('.')
    # name = name_split[0] + suffix + "." + name_split[1]
    # name = name.replace('.jpg', '.png').replace('.jpeg', '.png')
    
    file_path = Path(list[index - 1]) # create a Path object from the file path
    name = file_path.stem + suffix + ".png" # create new filename with suffix and new extension

    cv2.imwrite(os.path.join(out_dir, name), image)

def inference(model, batch_images, n_classes, flip_inference=True, scales=[1], preprocess_mode=None):
    x = preprocess(batch_images, mode=preprocess_mode)
    [x] = convert_to_tensors([x])

    # creates the variable to store the scores
    y_ = convert_to_tensors([np.zeros((x.shape[0], x.shape[1], x.shape[2], n_classes), dtype=np.float32)])[0]

    for scale in scales:
        # scale the image
        x_scaled = tf.compat.v1.image.resize_images(x, (int(x.shape[1] * scale), int(x.shape[2] * scale)),
                                          method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        y_scaled = model(x_scaled, training=False)

        # DEBUGGING
        print('# of y scaled unique b4 resizing', np.unique(y_scaled).size)
        #  rescale the output
        y_scaled = tf.compat.v1.image.resize_images(y_scaled, (x.shape[1], x.shape[2]),
                                          method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        # get scores
        y_scaled = tf.nn.softmax(y_scaled)

        if flip_inference:
            # calculates flipped scores
            y_flipped_ = tf.image.flip_left_right(model(tf.image.flip_left_right(x_scaled), training=False))
            # resize to rela scale
            y_flipped_ = tf.compat.v1.image.resize_images(y_flipped_, (x.shape[1], x.shape[2]),
                                                method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
            # get scores
            y_flipped_score = tf.nn.softmax(y_flipped_)

            y_scaled += y_flipped_score

        y_ += y_scaled

    return y_

# get accuracy and miou from a model
def get_metrics(loader, model, n_classes, train=True, flip_inference=False, scales=[1], write_images=False,
                preprocess_mode=None, n_samples_max=None):
    if train:
        loader.index_train = 0
    else:
        loader.index_test = 0

    accuracy = tf.metrics.Accuracy()
    conf_matrix = np.zeros((n_classes, n_classes))

    n_samples = len(loader.image_train_list) if train else len(loader.image_test_list)
    if n_samples_max is not None:
        n_samples = min(n_samples, n_samples_max)

    batch_size = 1
    for step in range(n_samples):  # for every batch
        print(f"Parsing img {step + 1}/{n_samples}", end="\r")
        x, y_raw, mask = loader.get_batch(size=batch_size, train=train, augmenter=False)

        [y] = convert_to_tensors([y_raw])
        y_ = inference(model, x, n_classes, flip_inference, scales, preprocess_mode=preprocess_mode)

        # generate images
        if write_images:
            for img_idx in range(batch_size):
                generate_image(y_[img_idx, ...], 'images_out', loader.dataFolderPath, loader, train)

                # Also write test properly
                generate_image(y_raw[img_idx, ...], 'images_out', loader.dataFolderPath, loader, train, suffix="_label")

        # DEBUGGING, y scores from inference
        y_max = np.argmax(y_[0,:,:], axis=-1).astype(float)
        print('y scores max: ', y_max)
        print('y scores unique:', np.unique(y_max))

        # Reshape
        y = tf.reshape(y, [y.shape[1] * y.shape[2] * y.shape[0], y.shape[3]])
        y_ = tf.reshape(y_, [y_.shape[1] * y_.shape[2] * y_.shape[0], y_.shape[3]])
        mask = tf.reshape(mask, [mask.shape[1] * mask.shape[2] * mask.shape[0]])

        print('shape after reshaping', str(y_.shape))

        labels, predictions = erase_ignore_pixels(labels=tf.argmax(y, 1), predictions=tf.argmax(y_, 1), mask=mask)
        accuracy(labels, predictions)
        conf_matrix += confusion_matrix(labels.numpy(), predictions.numpy(), labels=range(0, n_classes))

    # get the train and test accuracy from the model
    return accuracy.result(), compute_iou(conf_matrix)

# computes the miou given a confusion amtrix
def compute_iou(conf_matrix):
    epsilon = 1e-7
    intersection = np.diag(conf_matrix)
    ground_truth_set = conf_matrix.sum(axis=1)
    predicted_set = conf_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / (union.astype(np.float32) + epsilon)
    IoU[np.isnan(IoU)] = 0
    print(IoU)
    miou = np.mean(IoU)
    '''
    print(ground_truth_set)
    miou_no_zeros=miou*len(ground_truth_set)/np.count_nonzero(ground_truth_set)
    print ('Miou without counting classes with 0 elements in the test samples: '+ str(miou_no_zeros))
    '''
    return miou
