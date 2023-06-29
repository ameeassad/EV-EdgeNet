import numpy as np
import matplotlib.pyplot as plt
import utils.Loader as Loader


def load_from_file():
# Load .npy file
    image = np.load('../datasets/processed2/train/images/scene12_dyn_test_01_000000_classical_1.npy')
    mask = np.load('../datasets/processed2/train/labels/scene12_dyn_test_01_000000_mask_1.npy')
    return image, mask

def load_from_loader():
    loader = Loader.Loader(dataFolderPath='../datasets/processed', 
                           n_classes=17, 
                           problemType='segmentation',
                           width=640, 
                           height=480, 
                           channels=3, 
                           channels_events=6,
                           r_train_samples=1)
    x, y, mask = loader.get_batch(size=1, augmenter='segmentation')


def display_img_and_mask(image, mask):
    # Display RGB image
    plt.figure('image')
    plt.imshow(image)
    plt.axis('off')
    plt.show()


    # Plot the mask
    plt.figure('mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image, mask = load_from_file()
    display_img_and_mask(image, mask)