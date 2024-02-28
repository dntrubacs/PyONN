""" Created by Daniel-Iosif Trubacs 0n 25 February 2024. The purpose of this
module is tp show an example of a deep diffractive neural network trained
on the MNIST dataset."""

from pyonn.diffractive_layers import (
    InputDiffractiveLayer,
    DetectorLayer,
    DiffractiveLayer,
)
import numpy as np
import os
from pyonn.utils import (
    create_square_grid_pattern,
    plot_model_testing,
    plot_training_histogram,
)
from pyonn_data.datasets import OpticalImageDataset
from pyonn_data.processing import convert_optical_label
import torch
from torch.utils.data import DataLoader, random_split

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

# create an optical image dataset f
dataset = OpticalImageDataset(
    optical_images=train_images, optical_labels=train_labels
)

# in this case the size of the data is 60000 images, so the dataset will
# be split 54000:6000 into validation and training
train_dataset, validation_dataset = random_split(
    dataset=dataset, lengths=[54000, 6000]
)

# create a train and validation data loader
train_loader = DataLoader(
    dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0
)
validation_loader = DataLoader(
    dataset=validation_dataset, batch_size=32, shuffle=True, num_workers=0
)

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


# build the model and move to cuda if available
model = DiffractiveNN().to(device)

# loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs
n_epochs = 50

# a list of all train and validation losses after an epoch
train_losses = []
validation_losses = []

for epoch in range(n_epochs):
    # the current train and validation loss
    train_loss = 0
    validation_loss = 0

    # train the model
    model.train()

    # train the model
    for images, labels in train_loader:
        # forward pass
        output = model(images)
        loss = criterion(output, labels)

        # find the gradients and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add the training loss
        train_loss += loss.item()

    # validate the model
    # evaluate model:
    model.eval()

    with torch.no_grad():
        for images, labels in validation_loader:
            # forward pass
            output = model(images)
            loss = criterion(output, labels)

            # add the validation loss
            validation_loss += loss.item()

    # save the current train and validation loss
    train_losses.append(train_loss / len(train_loader))
    validation_losses.append(validation_loss / len(validation_loader))

    # print the train and validation after each epoch
    print(
        f"Epoch: {epoch}, train loss: {train_losses[-1]}, "
        f"validation loss: {validation_losses[-1]}"
    )

    # plot a histogram of the loss vs epoch
    plot_training_histogram(
        training_losses=train_losses, validation_losses=validation_losses
    )


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
        plot_model_testing(
            input_image=test_image,
            predicted_image=test_prediction,
            label_image=test_label,
            input_image_title=f"Input Image: {real_label}",
            predicted_image_title=f"Prediction: {real_prediction}",
            label_image_title=f"Label: {real_label}",
        )

# save the trained model
torch.save(model.state_dict(), "dnn_models/fashion_mnist_model_5_layers_v2")
