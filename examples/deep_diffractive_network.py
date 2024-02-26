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
import torch
from torch.utils.data import DataLoader

# load the data (must be optical images and labels)
os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN/data")
train_images = np.load("mnist_processed_data/train_images", allow_pickle=True)
train_labels = np.load("mnist_processed_data/train_labels", allow_pickle=True)

# create an optical image dataset
train_dataset = OpticalImageDataset(
    optical_images=train_images, optical_labels=train_labels
)

# create a data loader for the training data
train_loader = DataLoader(
    dataset=train_dataset, batch_size=160, shuffle=True, num_workers=0
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

n_epochs = 100
for epoch in range(n_epochs):

    for i, (images, labels) in enumerate(train_loader):
        # forward pass
        output = model(images)
        loss = criterion(output, labels)

        # find the gradients and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # copy the output and show it as plt
        if i % 100 == 0:
            print("epoch:", epoch, "i: ", i, " loss: ", loss)
            np_predicted = output.detach().cpu().numpy()
            np_labels = labels.detach().cpu().numpy()

            plot_label = np_labels[0]
            plot_predicted = np_predicted[0]

            plt.figure(figsize=(8, 4))
            plt.subplot(
                121, title=f"Prediction after epoch: {epoch}, " f"i: {i}"
            )
            plt.imshow(plot_predicted, origin="lower")
            plt.colorbar()
            plt.subplot(122, title="Label")
            plt.imshow(plot_label, origin="lower")
            plt.colorbar()
            plt.show()
