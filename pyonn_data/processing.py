""" Created by Daniel-Iosif Trubacs 25 February 2024. The purpose fo this
module is to process images datasets so they can be feed into an optical
neural network."""

import idx2numpy
import numpy as np
import cv2 as cv
import pickle
from typing import Optional


def create_optical_images(
    data_path: str, image_size: tuple, saved_data_path: Optional[str] = None
) -> np.ndarray:
    """Create optical images from raw black and white images.

    Args:
        data_path: Path to where the raw data is located. The images
            must be saved as ubyte and they must represent black and white
            images (such as the MNIST or Fashion-MNIST datasets) and with pixel
            values between 0 and 255.
        image_size: The size to which the images to be saved.
        saved_data_path: Where to save the process the data (and the
            name of it). E.g. data/train_images.
    Returns:
        Numpy array of shape (n_samples, image_size[0], image_size[1])
        representing the processed data. All the data is normalized between
        0 and 1 (which represent a complex amplitude).
    """
    # load the data from the file and convert it to numpy array
    data = idx2numpy.convert_from_file(data_path)

    # normalize the data to have values between 0 and 1
    data = data / 255

    # the processed data numpy array
    processed_data = np.zeros(
        shape=(data.shape[0], image_size[0], image_size[1])
    )

    # resize every image in the dataset
    for i in range(data.shape[0]):
        # flip the image upside down (necessary for plotting)
        processed_data[i] = np.flipud(cv.resize(data[i], image_size))

    # save the data id necessary
    if saved_data_path is not None:
        with open(saved_data_path, "wb") as handle:
            pickle.dump(processed_data, handle)

    # return the processed images
    return processed_data


def generate_optical_label(label: int) -> np.ndarray:
    """Generates optical label from a g iven value

    Args:
        label: Integer between 0 and 9.

    Returns:
        NUmpy array representing the optical label. The label represent
        a detector of size (120, 120) where only a region is 'lit up'
        (1 values).
    """
    # processed label
    processed_label = np.zeros(shape=(120, 120))

    # generate the label for 0
    if label == 0:
        processed_label[10:30, 10:30] = 1
    # generate the label for 1
    if label == 1:
        processed_label[10:30, 50:70] = 1
    # generate the label for 2
    if label == 2:
        processed_label[10:30, 90:110] = 1
    # generate the label for 3
    if label == 3:
        processed_label[50:70, 5:25] = 1
    # generate the label for 4
    if label == 4:
        processed_label[50:70, 35:55] = 1
    # generate the label for 5
    if label == 5:
        processed_label[50:70, 65:85] = 1
    # generate the label for 6
    if label == 6:
        processed_label[50:70, 95:115] = 1
    # generate the label for 7
    if label == 7:
        processed_label[90:110, 10:30] = 1
    # generate the label for 8
    if label == 8:
        processed_label[90:110, 50:70] = 1
    # generate the label for 9
    if label == 9:
        processed_label[90:110, 90:110] = 1

    # flip the image upside down return the processed label
    return np.flipud(processed_label)


def convert_optical_label(optical_label: np.ndarray) -> tuple:
    """Converts an optical label into a normal label.

    Args:
        optical_label: Numpy array representing the optical label. The label
            represent  a detector of size (120, 120) where only a region is
            'lit up' (1 values). Must be made with generate_optical_label

    Returns:
        The value with the maximum intensity measured in the activated regions
        and all the other values normalized to all the intensity in the
        region.
    """
    # all the detector region intensity values
    detector_regions = np.zeros(shape=(10,))

    # get the region intensity average for the 0 label
    detector_regions[0] = np.mean(optical_label[10:30, 10:30])

    # get the region intensity average for the 1 label
    detector_regions[1] = np.mean(optical_label[10:30, 50:70])

    # get the region intensity average for the 2 label
    detector_regions[2] = np.mean(optical_label[10:30, 90:110])

    # get the region intensity average for the 3 label
    detector_regions[3] = np.mean(optical_label[50:70, 5:25])

    # get the region intensity average for the 4 label
    detector_regions[4] = np.mean(optical_label[50:70, 35:55])

    # get the region intensity average for the 5 label
    detector_regions[5] = np.mean(optical_label[50:70, 65:85])

    # get the region intensity average for the 6 label
    detector_regions[6] = np.mean(optical_label[50:70, 95:115])

    # get the region intensity average for the 7 label
    detector_regions[7] = np.mean(optical_label[90:110, 10:30])

    # get the region intensity average for the 8 label
    detector_regions[8] = np.mean(optical_label[90:110, 50:70])

    # get the region intensity average for the 9 label
    detector_regions[9] = np.mean(optical_label[90:110, 90:110])

    return (
        np.argmax(detector_regions),
        detector_regions / np.sum(detector_regions),
    )


def create_optical_labels(
    labels_path: str, saved_label_path: Optional[str] = None
) -> np.ndarray:
    """Create optical labels and saves them.

    Args:
        labels_path: Path to where the raw labels are saved. Must be saved
            as ubyte and they should represent digits from 0 to 9.
        saved_label_path:  Where to save the process the labels (and the
            name of it). E.g. data/train_labels.

    Returns:
        Numpy array of shape (n_samples, 120, 120). See more details
        in the generate_optical_label function.
    """

    # load the data from the file and convert it to numpy array
    labels = idx2numpy.convert_from_file(labels_path)

    # the processed labels numpy array
    processed_labels = np.zeros(shape=(labels.shape[0], 120, 120))

    # create an optical label for each raw label
    for i in range(labels.shape[0]):
        processed_labels[i] = generate_optical_label(labels[i])

    # save the data if necessary
    if saved_label_path is not None:
        with open(saved_label_path, "wb") as handle:
            pickle.dump(processed_labels, handle)

    # return the processed images
    return processed_labels


def convert_fashion_mnist_label(label: int) -> str:
    """Convert the Fashion Mnist label from integer to clothing type.

    Args:
        label: Integer between 0 and 9 representing a class of clothing.

    Returns:
        The clothing type:
    """
    # the dictionary with all the clothing type
    clothing_dict = {
        0: "T - shirt / top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    # return the clothing type
    return clothing_dict[label]
