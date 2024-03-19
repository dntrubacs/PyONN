""" Created by Daniel-Iosif Trubacs on 12 March 2023. The purpose of this
module is to visualize the model diffractive layers and it's outputs."""

import numpy as np
import os
from pyonn.prebuilts import FiveLayerDiffractiveNN, InverseReLUDiffractiveNN
from pyonn_data.datasets import OpticalImageDataset
from pyonn.diffractive_layers import DiffractiveInverseReLU
from pyonn.utils import (
    test_model_on_image,
    plot_model_testing,
)
from pyonn_data.processing import convert_fashion_mnist_label
import torch
from pyonn.angular_spectrum_propagation_method import plot_real_maps

# Device configuration (used always fore very torch tensor declared)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the data (must be optical images and labels)
os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN")

test_images = np.load(
    file="data/fashion_mnist_processed_data/test_images", allow_pickle=True
)
test_labels = np.load(
    file="data/fashion_mnist_processed_data/test_labels", allow_pickle=True
)

test_dataset = OpticalImageDataset(
    optical_images=test_images, optical_labels=test_labels
)

# load the trained weights
normal_model = FiveLayerDiffractiveNN().to(device)
relu_model = InverseReLUDiffractiveNN().to(device)
normal_model.load_state_dict(
    torch.load(
        "saved_models/fully_optical/"
        "fashion_mnist_model_5_layers_50_"
        "epochs"
    )
)
relu_model.load_state_dict(
    torch.load(
        "saved_models/fully_optical/"
        "fashion_mnist_model_inverse_relu_5_layers_50_epochs"
    )
)


with torch.no_grad():
    # get a random index
    random_index = 0

    # get a random image and label
    test_image, test_label = test_dataset[random_index]

    # test the normal and relu model on the random data
    normal_output_test = test_model_on_image(
        model=normal_model, optical_image=test_image, optical_label=test_label
    )
    relu_output_test = test_model_on_image(
        model=relu_model, optical_image=test_image, optical_label=test_label
    )

    # convert the fashion mnist labels to strings
    normal_predicted_label = normal_output_test[3]
    normal_real_label = normal_output_test[4]

    # titles used for plotting
    input_image_label = convert_fashion_mnist_label(normal_real_label)
    predicted_image_label = convert_fashion_mnist_label(normal_predicted_label)
    labeled_image_label = convert_fashion_mnist_label(normal_real_label)

    # show all the diffractive layers amplitude and phase maps
    # model.diffractive_layer_0.plot_weights_map()
    # model.diffractive_layer_1.plot_weights_map()
    #  model.diffractive_layer_2.plot_weights_map()
    #  model.diffractive_layer_3.plot_weights_map()
    #  model.diffractive_layer_4.plot_weights_map()

    # show all layer outputs
    #   model.input_layer.plot_output_map()
    #   model.diffractive_layer_0.plot_output_map()
    #   model.diffractive_layer_1.plot_output_map()
    #  model.diffractive_layer_2.plot_output_map()
    normal_model.input_layer.plot_output_map()
    normal_model.diffractive_layer_4.plot_output_map()

    relu = DiffractiveInverseReLU(beta=0.8)
    complex_map_input = relu(relu_model.input_layer.output_map)
    dif_layer_output = relu(relu_model.diffractive_layer_4.output_map)
    plot_real_maps(
        complex_amplitude_map=complex_map_input,
        x_coordinates=normal_model.input_layer.x_coordinates,
    )
    plot_real_maps(
        complex_amplitude_map=dif_layer_output,
        x_coordinates=normal_model.input_layer.x_coordinates,
    )

    # plot the image, prediction and label
    plot_model_testing(
        input_image=normal_output_test[0],
        predicted_image=normal_output_test[1],
        label_image=normal_output_test[2],
        x_coordinates=normal_model.input_layer.x_coordinates,
        y_coordinates=normal_model.input_layer.y_coordinates,
        input_image_title=f"Input Image: " f"{input_image_label}",
        predicted_image_title=f"Prediction: " f"{predicted_image_label}",
        label_image_title=f"Label: " f"{labeled_image_label}",
    )

    plot_model_testing(
        input_image=relu_output_test[0],
        predicted_image=relu_output_test[1],
        label_image=relu_output_test[2],
        x_coordinates=relu_model.input_layer.x_coordinates,
        y_coordinates=relu_model.input_layer.y_coordinates,
        input_image_title=f"Input Image: " f"{input_image_label}",
        predicted_image_title=f"Prediction: " f"{predicted_image_label}",
        label_image_title=f"Label: " f"{labeled_image_label}",
    )
