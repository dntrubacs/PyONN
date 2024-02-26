""" Created by Daniel-Iosif Trubacs 0n 25 February 2024. The purpose of this
module is tp show an example of a deep diffractive neural network trained
on the MNIST dataset."""

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
from pyonn_data.processing import convert_optical_label
import torch
from torch.utils.data import DataLoader

# load the data (must be optical images and labels)
os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN/data")
train_images = np.load("mnist_processed_data/train_images", allow_pickle=True)
train_labels = np.load("mnist_processed_data/train_labels", allow_pickle=True)

train_images = train_images[0:1000]
train_labels = train_labels[0:1000]

# create an optical image dataset
train_dataset = OpticalImageDataset(
    optical_images=train_images, optical_labels=train_labels
)

# number of samples in the dataset
n_samples = train_dataset.__len__()

# create a data loader for the training data
train_loader = DataLoader(
    dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0
)

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
        self.detector_layer = DetectorLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            z_coordinate=20e-6,
        )

    # the forward pass
    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.diffractive_layer_0(x)
        x = self.detector_layer(x)
        return x


# build the model
model = DiffractiveNN()


# loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs
n_epochs = 5

# a list of all losses after an epoch
losses = []

# a list of accuracies predicted after an epoch
accuracies = []

for epoch in range(n_epochs):

    # number of correct predictions
    n_correct = 0

    # a list of all losses calculated after a batch size training
    batch_losses = []

    for i, (images, labels) in enumerate(train_loader):
        # forward pass
        output = model(images)
        loss = criterion(output, labels)

        # find the gradients and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # move from torch to numpy to find out the correct predictions
        np_predicted = output.detach().cpu().numpy()
        np_labels = labels.detach().cpu().numpy()

        # find the number of correct predictions
        # for index in range(np_predicted.shape[0]):
        # prediction = convert_optical_label(
        #      optical_label=np_predicted[index]
        #  )[0]
        #  label = convert_optical_label(
        #     optical_label=np_labels[index]
        #  )[0]
        # if prediction == label:
        #    n_correct += 1

        # add the batch loss
        batch_losses.append(loss.item())

    # find out and append the total loss
    loss = sum(batch_losses) / len(batch_losses)
    losses.append(loss)

    # print the prediction and batch loss
    print(f"epoch: {epoch}, loss: {loss}")

    # plot a histogram of the loss vs epoch
    plt.figure(figsize=(12, 8))
    plt.ylabel("Mean squared Error")
    plt.xlabel("Epochs")
    plt.plot(losses)
    plt.show()

# show 10 random predictions
with torch.no_grad():
    for j in range(10):
        # get a random index
        random_index = np.random.randint(low=0, high=1000)

        # get a random image and label
        test_image, test_label = train_dataset[random_index]

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
        axis[0].set_title(f"Input Image: {real_label}")
        c = axis[0].imshow(test_image, cmap="jet")
        figure.colorbar(mappable=c)

        # plot the prediction
        axis[1].set_title(f"Prediction: {real_prediction}")
        a = axis[1].imshow(test_prediction, cmap="jet")
        figure.colorbar(mappable=a)

        # plot the label
        axis[2].set_title(f"Label: {real_label}")
        b = axis[2].imshow(test_label, cmap="jet")
        figure.colorbar(mappable=b)

        plt.show()
