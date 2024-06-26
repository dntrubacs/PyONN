""" Created by Daniel-Iosif Trubacs 0n 25 February 2024. The purpose of this
module is tp show an example of a deep diffractive neural network trained
on the MNIST dataset."""

import numpy as np
import os
from pyonn.testing import (
    plot_training_histogram,
    plot_model_testing,
    test_model_on_image,
    test_model_on_optical_dataset,
)
from pyonn.utils import save_model_metric
from pyonn_data.datasets import OpticalImageDataset
from pyonn.prebuilts import FiveLayerDiffractiveNN
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

# build the model and move to cuda if available
model = FiveLayerDiffractiveNN().to(device)

# loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# number of epochs
n_epochs = 50

# a list of all train and validation losses after each epoch
train_losses = []
validation_losses = []

# a list of all train and validation losses after each epoch
train_accuracies = []
validation_accuracies = []

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

    # get the accuracy on the training and validation dataset
    train_accuracy = test_model_on_optical_dataset(
        model=model, dataset=train_dataset
    )
    validation_accuracy = test_model_on_optical_dataset(
        model=model, dataset=validation_dataset
    )

    # save the current train and validation accuracy
    train_accuracies.append(train_accuracy)
    validation_accuracies.append(validation_accuracy)

    # plot a histogram of the loss vs epoch
    plot_training_histogram(
        training_losses=train_losses,
        validation_losses=validation_losses,
        training_accuracies=train_accuracies,
        validation_accuracies=validation_accuracies,
        loss_label="Mean squared error and Accuracy",
    )


# show 10 random predictions
with torch.no_grad():
    for j in range(10):
        # get a random index
        random_index = np.random.randint(low=0, high=54000)

        # get a random image and label
        test_image, test_label = train_dataset[random_index]

        # test the model on the random data
        output_test = test_model_on_image(
            model=model, optical_image=test_image, optical_label=test_label
        )

        # plot the image, prediction and label
        plot_model_testing(
            input_image=output_test[0],
            predicted_image=output_test[1],
            label_image=output_test[2],
            x_coordinates=model.input_layer.x_coordinates,
            y_coordinates=model.input_layer.y_coordinates,
            input_image_title=f"Input Image: {output_test[4]}",
            predicted_image_title=f"Prediction: {output_test[3]}",
            label_image_title=f"Label: {output_test[4]}",
        )

# save the model inside a folder of the same name
# also save the training and validation accuracies and losses (necessary
# for later plotting)
os.chdir(
    "C:/Users/dit1u20/PycharmProjects/PyONN/"
    "saved_models/fully_optical/normal_diffractive"
)

save_model_metric(
    train_accuracies=np.array(train_accuracies),
    validation_accuracies=np.array(validation_accuracies),
    train_losses=np.array(train_losses),
    validation_losses=np.array(validation_losses),
    save_folder="fashion_mnist_model_5_layers_50_epochs",
)

# save the trained model
torch.save(
    model.state_dict(),
    f="fashion_mnist_model_5_layers_50_epochs/model",
)
