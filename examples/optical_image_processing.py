""" Created by Daniel-Iosif Trubacs on 27 February 2024. The purpose of this
module is to serve as an example from processing black and white images and c
converting them to optical images."""

from pyonn_data.processing import create_optical_images, create_optical_labels
import os

# convert the fashion mnist dataset into optical images and labels
os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN/data")

# convert the training data
print("Creating the train dataset. this might take a while.")
create_optical_images(
    data_path="fashion_mnist_raw_data/train-images-idx3-ubyte",
    image_size=(120, 120),
    saved_data_path="fashion_mnist_processed_data/train_images",
)
create_optical_labels(
    labels_path="fashion_mnist_raw_data/train-labels-idx1-ubyte",
    saved_label_path="fashion_mnist_processed_data/train_labels",
)

# convert the test data
print("Creating the test dataset. this might take a while.")
create_optical_images(
    data_path="fashion_mnist_raw_data/t10k-images-idx3-ubyte",
    image_size=(120, 120),
    saved_data_path="fashion_mnist_processed_data/test_images",
)
create_optical_labels(
    labels_path="fashion_mnist_raw_data/t10k-labels-idx1-ubyte",
    saved_label_path="fashion_mnist_processed_data/test_labels",
)

print("The optical test and train datasets have been created successfully!")
