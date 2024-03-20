"""
Created by Daniel-Iosif Trubacs on 26 February 2024. The purpose of this
module is to test already trained models.
"""

import numpy as np
import os
from pyonn.prebuilts import OpticalEncoder
from pyonn_data.datasets import HybridImageDataset
from pyonn.testing import (
    get_optical_encoder_prediction,
    plot_optical_encoder,
    test_model_on_hybrid_dataset,
)
import torch


# Device configuration (used always fore very torch tensor declared)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the data (must be optical images and labels)
os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN")

train_images = np.load(
    file="data/mnist_processed_data/train_images", allow_pickle=True
)
train_labels = np.load(
    file="data/mnist_raw_data/train_labels", allow_pickle=True
)

test_images = np.load(
    file="data/mnist_processed_data/test_images", allow_pickle=True
)
test_labels = np.load(
    file="data/mnist_raw_data/test_labels", allow_pickle=True
)

# load the trained weights
model = OpticalEncoder().to(device)
model.load_state_dict(
    torch.load(
        "saved_models/optical_encoders/"
        "mnist_model_optical_encoder_100_epochs"
    )
)

# create an optical image dataset
train_dataset = HybridImageDataset(
    optical_images=train_images, scalar_labels=train_labels
)

test_dataset = HybridImageDataset(
    optical_images=test_images, scalar_labels=test_labels
)


# find the accuracy for the training data
print("Finding the accuracy on training data")
train_accuracy = test_model_on_hybrid_dataset(
    model=model, dataset=train_dataset
)

# find the accuracy for the test data
print("Finding the accuracy on test data")
test_accuracy = test_model_on_hybrid_dataset(model=model, dataset=test_dataset)

print(f"Train accuracy: {train_accuracy*100} %")
print(f"Test accuracy: {test_accuracy*100} %")


with torch.no_grad():
    for j in range(10):
        # get a random index
        random_index = np.random.randint(low=0, high=10000)

        # get a random image and label
        test_image, test_label = test_dataset[random_index]
        test_label = test_label.detach().cpu().numpy()

        predicted_label = get_optical_encoder_prediction(
            model=model, optical_image=test_image
        )

        plot_optical_encoder(
            model=model,
            optical_image=test_image,
            label=test_label,
            x_coordinates=model.input_layer.x_coordinates,
            y_coordinates=model.input_layer.y_coordinates,
            image_title=None,
            save_path=f"results/model_predictions/optical_encoder/"
            f"mnist/model_predictions_{j}.png",
        )

    """
        # test the model on the random data
        output_test = test_model_on_image(
            model=model, optical_image=test_image, optical_label=test_label
        )

        # convert the fashion mnist labels to strings
        predicted_label = output_test[3]
        real_label = output_test[4]

        # titles used for plotting
        input_image_label = real_label
        predicted_image_label = predicted_label
        labeled_image_label = real_label

        # plot the image, prediction and label
        plot_model_testing(
            input_image=output_test[0],
            predicted_image=output_test[1],
            label_image=output_test[2],
            x_coordinates=model.input_layer.x_coordinates,
            y_coordinates=model.input_layer.y_coordinates,
            input_image_title=f"Input Image: " f"{input_image_label}",
            predicted_image_title=f"Prediction: " f"{predicted_image_label}",
            label_image_title=f"Label: " f"{labeled_image_label}"
            # save_path=f"results/model_predictions/fashion_mnist_predictions/"
            # f"model_predictions_{j}.png",
        )
    """
