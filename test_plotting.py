import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils.Loader_v2 import Loader


loader = Loader('../datasets/processed-fixed', problemType='segmentation', n_classes=30, width=640, height=480,
                median_frequency=0.00, channels=3, channels_events=6)
# print(loader.median_frequency_exp())
# x, y, mask = loader.get_batch(size=1, augmenter='segmentation')
x, y, mask = loader.get_batch(size=1, augmenter=None)

for i in range(100):
    x, y, mask = loader.get_batch(size=1, augmenter=None)
    # plt.figure(figsize=(15,10))

    # plt.subplot(2,4,1)
    # plt.imshow((x[0, :, :, :3]).astype(np.uint8)) # RGB image
    # plt.title('x')

    # plt.subplot(2,4,2)
    # plt.imshow((x[0, :, :, 3] * 127).astype(np.int8), cmap='gray')
    # plt.title('dvs+')

    # plt.subplot(2,4,3)
    # plt.imshow((x[0, :, :, 4] * 127).astype(np.int8), cmap='gray')
    # plt.title('dvs-')

    # plt.subplot(2,4,4)
    # plt.imshow((x[0, :, :, 5] * 127).astype(np.int8), cmap='gray')
    # plt.title('dvs+mean')

    # plt.subplot(2,4,5)
    # plt.imshow((x[0, :, :, 6] * 255).astype(np.int8), cmap='gray')
    # plt.title('dvs-std')

    # plt.subplot(2,4,6)
    # plt.imshow((x[0, :, :, 7] * 127).astype(np.int8), cmap='gray')
    # plt.title('dvs+std')

    # plt.subplot(2,4,7)
    # plt.imshow((x[0, :, :, 8] * 255).astype(np.int8), cmap='gray')
    # plt.title('dvs-std')

    # # plt.figure()
    # plt.subplot(2,4,8)
    # # Converting back to label encoding for visualization
    # y_label_encoded = np.argmax(y, axis=-1)  # original labels ranging from 0 to ?
    # # visualize one of these label-encoded images with a color gradient
    # plt.imshow(y_label_encoded[0], cmap='viridis')  # for first item in batch
    # plt.colorbar()  #colorbar on the side representing class labels
    # plt.title('y')
    # # plt.imshow((np.argmax(y, 3)[i, :, :] * 35).astype(np.uint8), cmap='gray')

    # plt.figure()
    # plt.imshow((mask[0, :, :] * 255).astype(np.uint8), cmap='gray')
    # plt.title('mask')
    
    # plt.show()



    # use the following if you just want to see labels and rgb
    y_label_encoded = np.argmax(y, axis=-1)  # original labels ranging from 0 to ?
    isin = np.isin([10], np.unique(y_label_encoded[0]))
    if np.any(isin):
        plt.figure(figsize=(10,6))
        plt.subplot(1,2,1)
        plt.imshow((x[0, :, :, :3]).astype(np.uint8)) # RGB image
        plt.title('x')

        plt.subplot(1,2,2)
        # Converting back to label encoding for visualization
        # y_label_encoded = np.argmax(y, axis=-1)  # original labels ranging from 0 to ?
        # visualize one of these label-encoded images with a color gradient
        plt.imshow(y_label_encoded[0], cmap='viridis')  # for first item in batch        

        plt.colorbar()  #colorbar on the side representing class labels
        plt.title('y')
        print(f"Unique label values: {np.unique(y_label_encoded[0])}")

        plt.show()



