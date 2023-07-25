from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import glob
import cv2
from .augmenters import get_augmenter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

np.random.seed(7)

class Loader:
    def __init__(self, 
                 dataFolderPath, 
                 width=224, 
                 height=224, 
                 channels=3, 
                 n_classes=14, 
                 problemType='segmentation',
                 median_frequency=0, 
                 other=False, 
                 channels_events=0, 
                 r_train_samples=1):
        self.dataFolderPath = dataFolderPath
        self.height = height
        self.channels_events = channels_events
        self.width = width
        self.dim = channels
        self.freq = np.zeros(n_classes)  # vector for calculating the class frequency
        self.index_train = 0  # indexes for iterating while training
        self.index_test = 0  # indexes for iterating while testing
        self.median_frequency_soft = median_frequency  # softener value for the median frequency balancing (if median_frequency==0, nothing is applied, if median_frequency==1, the common formula is applied)
        self.problemType = problemType

        # Share of training samples to use
        self.r_train_samples = r_train_samples

        print('Reading files...')
        '''
        possible structures:
        dataset
                train
                    images
                        image n..
                    labels
                        label n ..
                    weights       [optional]
                        weight n..
                test
                    images
                        image n..
                    labels
                        label n ..
                    weights       [optional]
                        weight n..
        '''

        # Load filepaths
        files = glob.glob(os.path.join(dataFolderPath, '*', '*', '*'))

        print('Structuring test and train files...')
        self.test_list = [file for file in files if '/test/' in file]
        self.train_list = [file for file in files if '/train/' in file]

        if other:
            self.test_list = [file for file in files if '/other/' in file]

        


        # problemType == 'segmentation':
        # The structure has to be dataset/train/images/image.png
        # The structure has to be dataset/train/labels/label.png
        # Separate image and label lists
        # Sort them to align labels and images

        self.image_train_list = [file for file in self.train_list if '/images/' in file]
        self.image_test_list = [file for file in self.test_list if '/images/' in file]
        self.label_train_list = [file for file in self.train_list if '/labels/' in file]
        self.label_test_list = [file for file in self.test_list if '/labels/' in file]
        self.events_train_list = [file for file in self.train_list if '/events/' in file]
        self.events_test_list = [file for file in self.test_list if '/events/' in file]

        self.label_test_list.sort()
        self.image_test_list.sort()
        self.label_train_list.sort()
        self.image_train_list.sort()
        self.events_train_list.sort()
        self.events_test_list.sort()

        # Shuffle train
        self.n_train_samples = int(len(self.image_train_list)*self.r_train_samples)
        self.suffle_segmentation()


        """
        image_train_list ['../datasets/processed/train/images/scene13_dyn_test_01_000000_classical_2.npy'
                        '../datasets/processed/train/images/scene12_dyn_test_00_000000_classical_1.npy'
                        '../datasets/processed/train/images/scene15_dyn_test_04_000000_classical_3.npy']
        image_test_list ['../datasets/processed/test/images/scene13_dyn_test_00_000000_classical_0.npy', 
                        '../datasets/processed/test/images/scene13_dyn_test_00_000000_classical_1.npy', 
                        '../datasets/processed/test/images/scene13_dyn_test_00_000000_classical_2.npy']
        
        label_train_list ['../datasets/processed/train/labels/scene13_dyn_test_01_000000_mask_2.npy'
                        '../datasets/processed/train/labels/scene12_dyn_test_00_000000_mask_1.npy'
                        '../datasets/processed/train/labels/scene15_dyn_test_04_000000_mask_3.npy']
        label_test_list ['../datasets/processed/test/labels/scene13_dyn_test_00_000000_mask_0.npy', 
                        '../datasets/processed/test/labels/scene13_dyn_test_00_000000_mask_1.npy', 
                        '../datasets/processed/test/labels/scene13_dyn_test_00_000000_mask_2.npy']
        
        events_train_list ['../datasets/processed/train/events/scene13_dyn_test_01_000000_2.npy'
                        '../datasets/processed/train/events/scene12_dyn_test_00_000000_1.npy'
                        '../datasets/processed/train/events/scene15_dyn_test_04_000000_3.npy']
        events_test_list ['../datasets/processed/test/events/scene13_dyn_test_00_000000_0.npy', 
                        '../datasets/processed/test/events/scene13_dyn_test_00_000000_1.npy', 
                        '../datasets/processed/test/events/scene13_dyn_test_00_000000_2.npy']
        """

        print('Loaded ' + str(len(self.image_train_list)) + ' training samples')
        print('Loaded ' + str(len(self.image_test_list)) + ' testing samples')
        self.n_classes = n_classes

        if self.median_frequency_soft != 0:
            self.median_freq = self.median_frequency_exp(soft=self.median_frequency_soft)

        print('Dataset contains ' + str(self.n_classes) + ' classes')

    def suffle_segmentation(self):
        s = np.arange(len(self.image_train_list))
        np.random.shuffle(s)

        if self.r_train_samples < 1: s = s[:self.n_train_samples]

        self.image_train_list = np.array(self.image_train_list)[s]
        self.label_train_list = np.array(self.label_train_list)[s]
        self.events_train_list = np.array(self.events_train_list)[s]

    # Returns a weighted mask from a binary mask
    def _from_binarymask_to_weighted_mask(self, labels, masks):
        '''
        used to balance the impact of each class on the loss function by assigning higher weights to less frequent classes

        the input [mask] is an array of N binary masks 0/1 of size [N, H, W ] where the 0 are pixeles to ignore from the labels [N, H, W ]
        and 1's means pixels to take into account.
        This function transofrm those 1's into a weight using the median frequency
        '''
        weights = self.median_freq
        for i in range(masks.shape[0]):
            # for every mask of the batch
            label_image = labels[i, :, :]
            mask_image = masks[i, :, :]
            dim_1 = mask_image.shape[0]
            dim_2 = mask_image.shape[1]
            label_image = np.reshape(label_image, (dim_2 * dim_1))
            mask_image = np.reshape(mask_image, (dim_2 * dim_1))

            for label_i in range(self.n_classes):
                # multiply the mask so far, with the median frequency wieght of that label
                mask_image[label_image == label_i] = mask_image[label_image == label_i] * weights[label_i]
            # unique, counts = np.unique(mask_image, return_counts=True)

            mask_image = np.reshape(mask_image, (dim_1, dim_2))
            masks[i, :, :] = mask_image

        return masks

    def _perform_augmentation_segmentation(self, img, label, mask_image, augmenter, event=None, events=False):
        seq_image, seq_label, seq_mask, seq_event = get_augmenter(name=augmenter, c_val=255)

        # apply some contrast  to de rgb image
        img = img.reshape(sum(((1,), img.shape), ()))
        img = seq_image.augment_images(img)
        img = img.reshape(img.shape[1:])

        label = label.reshape(sum(((1,), label.shape), ()))
        label = label.astype(np.int32)
        label = seq_label.augment_images(label)
        label = label.reshape(label.shape[1:])

        mask_image = mask_image.reshape(sum(((1,), mask_image.shape), ()))
        mask_image = mask_image.astype(np.int32)
        mask_image = seq_mask.augment_images(mask_image)
        mask_image = mask_image.reshape(mask_image.shape[1:])

        if events:
            event = event.reshape(sum(((1,), event.shape), ()))
            # event = self.augment_event(event)
            event = seq_event.augment_images(event)
            event = event.reshape(event.shape[1:])
            return img, label, mask_image, event

        return img, label, mask_image

    # Returns a random batch of segmentation images: X, Y, mask
    def _get_batch_segmentation(self, size=32, train=True, augmenter=None):
        events = self.channels_events > 0

        # init numpy arrays
        if events:
            x = np.zeros([size, self.height, self.width, int(self.dim + self.channels_events)], dtype=np.float32)
        else:
            x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
        y = np.zeros([size, self.height, self.width], dtype=np.uint8)
        mask = np.ones([size, self.height, self.width], dtype=np.float32)

        # print("Shape of x: ", x.shape)
        # print("Shape of y (labels/mask): ", y.shape)
        # print("Shape of mask: ", mask.shape)

        if train:
            image_list = self.image_train_list
            label_list = self.label_train_list
            event_list = self.events_train_list

            # Get [size] random numbers
            indexes = [i % len(image_list) for i in range(self.index_train, self.index_train + size)]
            self.index_train = indexes[-1] + 1

        else:
            image_list = self.image_test_list
            label_list = self.label_test_list
            event_list = self.events_test_list

            # Get [size] random numbers
            indexes = [i % len(image_list) for i in range(self.index_test, self.index_test + size)]
            self.index_test = indexes[-1] + 1

        random_images = [image_list[number] for number in indexes]
        random_labels = [label_list[number] for number in indexes]
        random_event_list = [event_list[number] for number in indexes]

        # for every random image, get the image, label and mask.
        # the augmentation has to be done separately due to augmentation
        for index in range(size):
            img = np.load(random_images[index])
            # flip the image bc in EVIMO the classical images are upside-down for some reason
            img = np.flipud(img)
            img = np.fliplr(img)
            
            label = np.load(random_labels[index])
            label = self.evimo_label_processor(label)

            if self.problemType=="edges":
                unique_labels = np.unique(label)
                semantic_edge_map = np.zeros_like(label)
                for i in unique_labels:
                    # Create a binary image for the current class label
                    label_map = np.where(label == i, i, 0).astype(np.uint8)
                    # Erode the mask
                    eroded = cv2.erode(label_map, np.ones((3,3), np.uint8), iterations=1)
                    # border = original mask - eroded mask
                    label_border = label_map - eroded

                    # # Gaussian blur to smoothen the borders
                    # smooth_label_border = cv2.GaussianBlur(label_border, (11,11), 0)
                    # # Threshold the blurred image to get binary borders again; adjust threshold value
                    # _, smooth_label_border = cv2.threshold(smooth_label_border, 50, i, cv2.THRESH_BINARY)
                    # condition = np.logical_or(semantic_edge_map == 0, smooth_label_border == 0)
                    # semantic_edge_map[condition] += smooth_label_border[condition]
                                           
                    # Add current class border to the overall border image
                    condition = np.logical_or(semantic_edge_map == 0, label_border == 0)
                    semantic_edge_map[condition] += label_border[condition]
 
                    # convert back to index of class - only if label_map uses values 255 instead of i
                    # edge_map = np.where(edge_map==255, i, edge_map)
                    
                    # semantic_edge_map = np.concatenate((semantic_edge_map, edge_map))
                label = semantic_edge_map
            # no need because using np arrays
            # if self.dim == 1:
                # img = cv2.imread(random_images[index], 0)
            # else:
                # img = cv2.imread(random_images[index])
                # img = tf.keras.preprocessing.image.load_img(random_images[index])
                # img = tf.keras.preprocessing.image.img_to_array(img)

            # label = cv2.imread(random_labels[index], 0)
            if events:
                event = np.load(random_event_list[index])
                #event=np.swapaxes(np.swapaxes(event, 0, 2), 0, 1)

            mask_image = mask[index, :, :]

            # Reshape images if its needed
            if img.shape[1] != self.width or img.shape[0] != self.height:
                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
            if label.shape[1] != self.width or label.shape[0] != self.height:
                label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                if events:
                    event = cv2.resize(event, (self.width, self.height), interpolation=cv2.INTER_NEAREST)


            if train and augmenter:
                if events:
                    img, label, mask_image, event = self._perform_augmentation_segmentation(img, label, mask_image, augmenter, event, events)
                else:
                    img, label, mask_image = self._perform_augmentation_segmentation(img, label, mask_image, augmenter)

            # plt.figure('LABEL BEFORE')
            # plt.imshow(label, cmap='viridis')
            # plt.axis('off')
            # plt.show()
            # modify the mask and the labels. 
            #  this only works if the labels have index determined by number of classes?
            mask_ignore = label >= self.n_classes # creates binary 2D numpy array
            mask_zeros = label==0  # added label==0 for the pixels without a class
            mask_image[mask_zeros] = 0
            mask_image[mask_ignore] = 0  # The ignore pixels will have a value of 0 in the mask
            label[mask_ignore] = 0  # The ignore label will be n_classes

            # DEBUGGING 
            # print('label values', np.unique(label))

            if self.dim == 1:
                img = np.reshape(img, (img.shape[0], img.shape[1], self.dim))

            if self.dim > 0:
                x[index, :, :, :self.dim] = img.astype(np.float32)
            if events:
                x[index, :, :, self.dim:] = event[:,:,:self.channels_events].astype(np.float32)

            y[index, :, :] = label
            mask[index, :, :] = mask_image

        # Apply weights to the mask
        if self.median_frequency_soft > 0:
            mask = self._from_binarymask_to_weighted_mask(y, mask)

        # the labeling to categorical (if 5 classes and value is 2:  2 -> [0,0,1,0,0])
        a, b, c = y.shape
        y = y.reshape((a * b * c))

        # Convert to categorical. Add one class for ignored pixels
        y = to_categorical(y, num_classes=self.n_classes)
        y = y.reshape((a, b, c, self.n_classes)).astype(np.uint8)

        return x, y, mask


    def _get_key_by_value(self, dictionary, value_searching):
        for key, value in dictionary.iteritems():
            if value == value_searching:
                return key

        return None

    # Returns a random batch
    def get_batch(self, size=32, train=True, augmenter=None):
        '''
        Gets a batch of size [size]. If [train] the data will be training data, if not, test data.
        if augmenter is no None, image augmentation will be perform (see file augmenters.py)
        if images are bigger than max_size of smaller than min_size, images will be resized (forced)
        '''
        return self._get_batch_segmentation(size=size, train=train, augmenter=augmenter)

    # Returns the median frequency for class imbalance. It can be soften with the soft value (<=1)
    def median_frequency_exp(self, soft=1):
        for image_label_train in self.label_train_list:
            # image = cv2.imread(image_label_train, 0)
            image = np.load(image_label_train)
            for label in range(self.n_classes):
                self.freq[label] = self.freq[label] + sum(sum(image == label))

        zeros = self.freq == 0
        if sum(zeros) > 0:
            print('There are some classes which are not contained in the training samples')

        results = np.median(self.freq) / self.freq
        results[zeros] = 0  # for not inf values.
        results = np.power(results, soft)
        print(results)
        return results


    def augment_event(self, event_image, swap_max=0.35, delete_pixel_max=0.80, make_up_max=0.02, change_value_max=0.45):
        _, w, h, c = event_image.shape
        pixels = w*h

        swap_pixels_max=int(pixels*swap_max)
        delete_pixel_pixels_max=int(pixels*delete_pixel_max)
        make_up_pixels_max=int(pixels*make_up_max)
        change_value_pixels_max=int(pixels*change_value_max)

        swap_pixels = np.random.randint(0, high=swap_pixels_max)
        delete_pixel_pixels = np.random.randint(0, high=delete_pixel_pixels_max)
        make_up_pixels = np.random.randint(0, high=make_up_pixels_max)
        change_value_pixels = np.random.randint(0, high=change_value_pixels_max)

        for index in range(swap_pixels):
            i = np.random.randint(0, w)
            j = np.random.randint(0, h)
            i_n, j_n = get_neighbour(i, j, w-1, h-1)
            value_aux = event_image[:, i, j, :]
            event_image[:, i, j, :] = event_image[:, i_n, j_n, :]
            event_image[:, i_n, j_n, :] = value_aux

        for index in range(change_value_pixels):
            i = np.random.randint(0, w)
            j = np.random.randint(0, h)
            i_n, j_n = get_neighbour(i, j, w-1, h-1)
            if event_image[0, i_n, j_n, 0] > - 1 or event_image[0, i_n, j_n, 1] > - 1:
                event_image[:, i, j, :] = event_image[:, i_n, j_n, :]

        for index in range(make_up_pixels):
            i = np.random.randint(0, w)
            j = np.random.randint(0, h)
            event_image[:, i, j, 0] = np.random.random() * 2 - 1
            event_image[:, i, j, 1] = np.random.random() * 2 - 1
            event_image[:, i, j, 2] = np.random.random() * 2 - 1
            event_image[:, i, j, 3] = np.random.random()
            event_image[:, i, j, 4] = np.random.random() * 2 - 1
            event_image[:, i, j, 5] = np.random.random()

        for index in range(delete_pixel_pixels):
            i = np.random.randint(0, w)
            j = np.random.randint(0, h)
            event_image[:, i, j, 0] = -1
            event_image[:, i, j, 1] = -1
            event_image[:, i, j, 2] = 0
            event_image[:, i, j, 3] = 0
            event_image[:, i, j, 4] = 0
            event_image[:, i, j, 5] = 0



        '''
        #intercambiar pixels  en todos los canales (can vecinos)

        #en pixel aleatorio:
        -eliminar totalmente todos sus valores
        -inventarte (copiar uno de un vecino)
        - subir o bajar algo el valor de cualquier canal
        '''
        return event_image

    def evimo_label_processor(self, label_img):
        if self.n_classes != 14:
            return label_img
        # print("Using 14 class labels")
        # print("Before mapping:", np.unique(label_img))
        mapping = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 1,
            6: 2,
            7: 0, # not sure what it is
            8: 3,
            9: 4,
            10: 5,
            11: 6,
            12: 7,
            13: 8,
            14: 9,
            15: 9,
            16: 0, # not sure what it is
            17: 10,
            18: 10,
            19: 10,
            20: 10,
            21: 0,  # not sure what it is
            22: 11,
            23: 12,
            24: 13,
            25: 14,
            26: 14,
            27: 14,
            28: 14
        }
        # Vectorize the mapping function
        vectorized_mapping = np.vectorize(mapping.get)

        # Apply the mapping to the label img
        new_labels = vectorized_mapping(label_img)
        # print("after mapping:", np.unique(new_labels))

        return new_labels


    def batchex_printer(self, x, y, mask, rgb= False, predicted=False):
        if rgb:
            for i in range(1):
                plt.figure(figsize=(15,10))

                plt.subplot(2,4,1)
                plt.imshow((x[i, :, :, :3]).astype(np.uint8)) # RGB image
                plt.title('x')

                plt.subplot(2,4,2)
                plt.imshow((x[i, :, :, 3] * 127).astype(np.int8), cmap='gray')
                plt.title('dvs+')

                plt.subplot(2,4,3)
                plt.imshow((x[i, :, :, 4] * 127).astype(np.int8), cmap='gray')
                plt.title('dvs-')

                plt.subplot(2,4,4)
                plt.imshow((x[i, :, :, 5] * 127).astype(np.int8), cmap='gray')
                plt.title('dvs+mean')

                plt.subplot(2,4,5)
                plt.imshow((x[i, :, :, 6] * 255).astype(np.int8), cmap='gray')
                plt.title('dvs-std')

                plt.subplot(2,4,6)
                plt.imshow((x[i, :, :, 7] * 127).astype(np.int8), cmap='gray')
                plt.title('dvs+std')

                plt.subplot(2,4,7)
                plt.imshow((x[i, :, :, 8] * 255).astype(np.int8), cmap='gray')
                plt.title('dvs-std')

                plt.subplot(2,4,8)
                # Converting back to label encoding for visualization
                y_label_encoded = np.argmax(y, axis=-1)  # original labels ranging from 0 to 24
                # visualize one of these label-encoded images with a color gradient
                plt.imshow(y_label_encoded[0], cmap='viridis')  # Change the index as needed
                plt.colorbar()  #colorbar on the side representing class labels
                plt.title('y')
                # plt.imshow((np.argmax(y, 3)[i, :, :] * 35).astype(np.uint8), cmap='gray')

                plt.figure()
                plt.imshow((mask[i, :, :] * 255).astype(np.uint8), cmap='gray')
                plt.title('mask')
                
                plt.show()
                plt.close()
        elif predicted:
            y = y[0,:,:]
            y_ = np.argmax(y, axis=-1).astype(float)
            print("y_ classes unique (predicted, during training):", np.unique(y_))
            y_ *= 255/y.shape[-1]
            plt.figure()
            plt.imshow(y_, cmap='viridis')
            plt.colorbar()
            plt.savefig('temp_predicted_output.png')
            plt.close()
        else:
            for i in range(1):

                plt.figure(figsize=(15,10))

                plt.subplot(2,4,1)
                plt.imshow((mask[i, :, :] * 255).astype(np.uint8), cmap='gray')
                plt.title('mask')

                plt.subplot(2,4,2)
                plt.imshow((x[i, :, :, 0] * 127).astype(np.int8), cmap='gray')
                plt.title('dvs+')

                plt.subplot(2,4,3)
                plt.imshow((x[i, :, :, 1] * 127).astype(np.int8), cmap='gray')
                plt.title('dvs-')

                plt.subplot(2,4,4)
                plt.imshow((x[i, :, :, 2] * 127).astype(np.int8), cmap='gray')
                plt.title('dvs+mean')

                plt.subplot(2,4,5)
                plt.imshow((x[i, :, :, 3] * 255).astype(np.int8), cmap='gray')
                plt.title('dvs-std')

                plt.subplot(2,4,6)
                plt.imshow((x[i, :, :, 4] * 127).astype(np.int8), cmap='gray')
                plt.title('dvs+std')

                plt.subplot(2,4,7)
                plt.imshow((x[i, :, :, 5] * 255).astype(np.int8), cmap='gray')
                plt.title('dvs-std')

                plt.subplot(2,4,8)
                # Converting back to label encoding for visualization
                y_label_encoded = np.argmax(y, axis=-1)  # original labels ranging from 0 to 24
                # visualize one of these label-encoded images with a color gradient
                plt.imshow(y_label_encoded[0], cmap='viridis')  # Change the index as needed
                plt.colorbar()  #colorbar on the side representing class labels
                plt.title('y')
                # plt.imshow((np.argmax(y, 3)[i, :, :] * 35).astype(np.uint8), cmap='gray')
                plt.close()
                



def  get_neighbour(i, j, max_i, max_j):
    random_number= np.random.random()
    if random_number < 0.25:
        j += 1

    elif random_number < 0.50:
        j -= 1

    elif random_number < 0.75:
        i -= 1

    else:
        i += 1

    if j < 0: j = 0
    if i < 0: i = 0
    if j > max_j: j = max_j
    if i > max_i: i = max_i

    return i, j



# if __name__ == "__main__":

#     loader = Loader('../datasets/processed', problemType='edges', n_classes=25, width=640, height=480,
#                     median_frequency=0.00, channels=3, channels_events=6)
#     # print(loader.median_frequency_exp())
#     # x, y, mask = loader.get_batch(size=1, augmenter='segmentation')
#     x, y, mask = loader.get_batch(size=1, augmenter=None)

#     for i in range(1):
#         plt.figure(figsize=(15,10))

#         plt.subplot(2,4,1)
#         plt.imshow((x[i, :, :, :3]).astype(np.uint8)) # RGB image
#         plt.title('x')

#         plt.subplot(2,4,2)
#         plt.imshow((x[i, :, :, 3] * 127).astype(np.int8), cmap='gray')
#         plt.title('dvs+')

#         plt.subplot(2,4,3)
#         plt.imshow((x[i, :, :, 4] * 127).astype(np.int8), cmap='gray')
#         plt.title('dvs-')

#         plt.subplot(2,4,4)
#         plt.imshow((x[i, :, :, 5] * 127).astype(np.int8), cmap='gray')
#         plt.title('dvs+mean')

#         plt.subplot(2,4,5)
#         plt.imshow((x[i, :, :, 6] * 255).astype(np.int8), cmap='gray')
#         plt.title('dvs-std')

#         plt.subplot(2,4,6)
#         plt.imshow((x[i, :, :, 7] * 127).astype(np.int8), cmap='gray')
#         plt.title('dvs+std')

#         plt.subplot(2,4,7)
#         plt.imshow((x[i, :, :, 8] * 255).astype(np.int8), cmap='gray')
#         plt.title('dvs-std')

#         plt.subplot(2,4,8)
#         # Converting back to label encoding for visualization
#         y_label_encoded = np.argmax(y, axis=-1)  # original labels ranging from 0 to 24
#         # visualize one of these label-encoded images with a color gradient
#         plt.imshow(y_label_encoded[0], cmap='viridis')  # Change the index as needed
#         plt.colorbar()  #colorbar on the side representing class labels
#         plt.title('y')
#         # plt.imshow((np.argmax(y, 3)[i, :, :] * 35).astype(np.uint8), cmap='gray')

#         plt.figure()
#         plt.imshow((mask[i, :, :] * 255).astype(np.uint8), cmap='gray')
#         plt.title('mask')
        
#         plt.show()


    # x, y, mask = loader.get_batch(size=3, train=False)
