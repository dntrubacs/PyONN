"""
Created by Daniel-Iosif Trubacs on 26 February 2024. The purpose of this
module is to test already trained models.
"""
from pyonn.diffractive_layers import (
    InputDiffractiveLayer,
    DetectorLayer,
    DiffractiveLayer,
)
import numpy as np
from matplotlib import pyplot as plt
import os
from pyonn.utils import create_square_grid_pattern
from pyonn_data.datasets import OpticalImageDataset
from pyonn_data.processing import (
    convert_optical_label,
    convert_fashion_mnist_label,
)
import torch


# Device configuration (used always fore very torch tensor declared)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the data (must be optical images and labels)
os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN")
train_images = np.load(
    file="data/fashion_mnist_processed_data/train_images", allow_pickle=True
)
train_labels = np.load(
    file="data/fashion_mnist_processed_data/train_labels", allow_pickle=True
)
test_images = np.load(
    file="data/fashion_mnist_processed_data/test_images", allow_pickle=True
)
test_labels = np.load(
    file="data/fashion_mnist_processed_data/test_labels", allow_pickle=True
)

# create an optical image dataset
train_dataset = OpticalImageDataset(
    optical_images=train_images, optical_labels=train_labels
)

test_dataset = OpticalImageDataset(
    optical_images=test_images, optical_labels=test_labels
)

# number of samples in the dataset
n_samples_train = train_dataset.__len__()
n_samples_test = test_dataset.__len__()

# wavelength of light
wavelength = 1.55e-6

# create a square grid pattern centred on [0, 0] with pixel size 0.8 um
# and pixel number 120 (120^2 pixels in total)
square_grid_pattern = create_square_grid_pattern(
    center_coordinates=np.array([0, 0]),
    pixel_length=0.8e-6,
    pixel_number=120,
    pixel_separation=0.0,
    grid_z_coordinate=0,
)
# get the coordinates
x_coordinates = square_grid_pattern[1]


class DiffractiveNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wavelength = 1.55e-6
        self.neuron_size = 120
        self.x_coordinates = x_coordinates
        self.input_layer = InputDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=0,
            z_next=10e-6,
        )
        self.diffractive_layer_0 = DiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=10e-6,
            z_next=20e-6,
        )
        self.diffractive_layer_1 = DiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=20e-6,
            z_next=30e-6,
        )
        self.diffractive_layer_2 = DiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=30e-6,
            z_next=40e-6,
        )
        self.diffractive_layer_3 = DiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=40e-6,
            z_next=50e-6,
        )
        self.diffractive_layer_4 = DiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=50e-6,
            z_next=60e-6,
        )
        self.detector_layer = DetectorLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            z_coordinate=60e-6,
        )

    # the forward pass
    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.diffractive_layer_0(x)
        x = self.diffractive_layer_1(x)
        x = self.diffractive_layer_2(x)
        x = self.diffractive_layer_3(x)
        x = self.diffractive_layer_4(x)
        x = self.detector_layer(x)
        return x


# load the trained weights
model = DiffractiveNN().to(device)
model.load_state_dict(torch.load("dnn_models/fashion_mnist_model_5_layers_v2"))

# n correct for test and train dataset
n_correct_train = 0
n_correct_test = 0


# find the accuracy for the training data
for i in range(n_samples_train):
    train_image, train_label = train_dataset[i]
    prediction = model(train_image)
    np_prediction = prediction.detach().cpu().numpy()
    np_label = train_label.detach().cpu().numpy()
    real_prediction = convert_optical_label(optical_label=np_prediction)[0]
    real_label = convert_optical_label(optical_label=np_label)[0]
    if real_prediction == real_label:
        n_correct_train += 1
    if i % 1000 == 0 and i > 1:
        print(i, n_correct_train, n_correct_train / i * 100)

# find the accuracy for the testing data
for i in range(n_samples_test):
    test_image, test_label = test_dataset[i]
    prediction = model(test_image)
    np_prediction = prediction.detach().cpu().numpy()
    np_label = test_label.detach().cpu().numpy()
    real_prediction = convert_optical_label(optical_label=np_prediction)[0]
    real_label = convert_optical_label(optical_label=np_label)[0]
    if real_prediction == real_label:
        n_correct_test += 1

    if i % 1000 == 0 and i > 1:
        print(i, n_correct_test, n_correct_test / i * 100)

print(f"train accuracy: {n_correct_train/n_samples_train*100} %")
print(f"test accuracy: {n_correct_test/n_samples_test*100} %")


with torch.no_grad():
    for j in range(50):
        # get a random index
        random_index = np.random.randint(low=0, high=1000)

        # get a random image and label
        test_image, test_label = test_dataset[random_index]

        # get the prediction for images
        test_prediction = model(test_image)

        # convert them to numpy
        test_image = test_image.detach().cpu().numpy()
        test_label = test_label.detach().cpu().numpy()
        test_prediction = test_prediction.detach().cpu().numpy()

        # get the 'real' labels (integer not optical images
        real_prediction = convert_optical_label(optical_label=test_prediction)[
            0
        ]
        real_label = convert_optical_label(optical_label=test_label)[0]

        # plot the image, prediction and label
        # create the figure
        figure, axis = plt.subplots(1, 3, figsize=(30, 8))

        # plot the initial image
        axis[0].set_title(
            f"Input Image: " f"{convert_fashion_mnist_label(real_label)}"
        )
        c = axis[0].imshow(test_image, cmap="jet")
        figure.colorbar(mappable=c)

        # plot the prediction
        axis[1].set_title(
            f"Prediction: " f"{convert_fashion_mnist_label(real_prediction)}"
        )
        a = axis[1].imshow(test_prediction, cmap="inferno")
        figure.colorbar(mappable=a)

        # plot the label
        axis[2].set_title(
            f"Label: " f"{convert_fashion_mnist_label(real_label)}"
        )
        b = axis[2].imshow(test_label, cmap="inferno")
        figure.colorbar(mappable=b)
        plt.savefig(
            f"results/model_predictions/fashion_mnist"
            f"/model_prediction_{j}.png"
        )
        plt.show()
