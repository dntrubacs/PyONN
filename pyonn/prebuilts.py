""" Created by Daniel-Iosif Trubacs on 28 February 2024 for the UoS
QLM group. The purpose of this module ios to built certain off-the-shelves
Deep Diffractive Networks (DNNs) that can be imported and used for
training. For now, all the DNNs use DiffractiveLayers of 120x120 neurons
and each neuron has a length of 0.8 um."""

from pyonn.utils import create_square_grid_pattern
from pyonn.diffractive_layers import (
    InputDiffractiveLayer,
    DetectorLayer,
    DiffractiveLayer,
    DiffractiveReLU,
    DiffractiveInverseReLU,
    PhaseDiffractiveLayer,
    BinaryAmplitudeDiffractiveLayer,
)
import numpy as np
import torch

# create a square grid pattern centred on [0, 0] with pixel size 0.8 um
# and pixel number 120 (120^2 pixels in total)
square_grid_pattern = create_square_grid_pattern(
    center_coordinates=np.array([0, 0]),
    pixel_length=0.8e-6,
    pixel_number=120,
    pixel_separation=0.0,
    grid_z_coordinate=0,
)

# get the x coordinate. this coordinates will be used in all DNNs
x_coordinates = square_grid_pattern[1]


class FiveLayerDiffractiveNN(torch.nn.Module):
    """Diffractive Neural Network consisting of 5 layers.

    The wavelength of light is set at 1.55 um, the neuron size is always
    120 and the distance between 2 layers is always 10 um.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wavelength = 1.55e-6
        self.neuron_size = 120
        self.x_coordinates = x_coordinates

        # input layer used to propagate optical images
        self.input_layer = InputDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=0,
            z_next=10e-6,
        )

        # diffractive layers
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

        # detector layer used to measure the output intensity
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


class ReLUDiffractiveNN(torch.nn.Module):
    """Diffractive Neural Network consisting of 5 layers.

    The wavelength of light is set at 1.55 um, the neuron size is always
    120 and the distance between 2 layers is always 10 um.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wavelength = 1.55e-6
        self.neuron_size = 120
        self.x_coordinates = x_coordinates

        # input layer used to propagate optical images
        self.input_layer = InputDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=0,
            z_next=10e-6,
        )

        # diffractive layers
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

        # detector layer used to measure the output intensity
        self.detector_layer = DetectorLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            z_coordinate=60e-6,
        )
        self.relu = DiffractiveReLU(alpha=0.01)

    # the forward pass
    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.diffractive_layer_0(x)
        x = self.relu(x)
        x = self.diffractive_layer_1(x)
        x = self.relu(x)
        x = self.diffractive_layer_2(x)
        x = self.relu(x)
        x = self.diffractive_layer_3(x)
        x = self.relu(x)
        x = self.diffractive_layer_4(x)
        x = self.relu(x)
        x = self.detector_layer(x)
        return x


class InverseReLUDiffractiveNN(torch.nn.Module):
    """Diffractive Neural Network consisting of 5 layers.

    The wavelength of light is set at 1.55 um, the neuron size is always
    120 and the distance between 2 layers is always 10 um.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wavelength = 1.55e-6
        self.neuron_size = 120
        self.x_coordinates = x_coordinates

        # input layer used to propagate optical images
        self.input_layer = InputDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=0,
            z_next=10e-6,
        )

        # diffractive layers
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

        # detector layer used to measure the output intensity
        self.detector_layer = DetectorLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            z_coordinate=60e-6,
        )
        self.inverse_relu = DiffractiveInverseReLU(beta=0.8)

    # the forward pass
    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.diffractive_layer_0(x)
        x = self.inverse_relu(x)
        x = self.diffractive_layer_1(x)
        x = self.inverse_relu(x)
        x = self.diffractive_layer_2(x)
        x = self.inverse_relu(x)
        x = self.diffractive_layer_3(x)
        x = self.inverse_relu(x)
        x = self.diffractive_layer_4(x)
        x = self.inverse_relu(x)
        x = self.detector_layer(x)
        return x


class OpticalEncoder(torch.nn.Module):
    """Optical Encoder consisting of a Deep Diffractive Layer and normal
    digital layers.

    The architecture of the Network is: Input Layer (Diffractive) ->
    Deep Diffractive Layer -> Detector (Diffractive) -> MaxPool -> FCC.

    The wavelength of light is set at 1.55 um, the neuron size of the
    Diffractive layer is 120 and the distance between 2 layers is always 10 um.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wavelength = 1.55e-6
        self.neuron_size = 120
        self.x_coordinates = x_coordinates

        # input layer used to propagate optical images
        self.input_layer = InputDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=0,
            z_next=10e-6,
        )

        # diffractive layers
        self.diffractive_layer = DiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=10e-6,
            z_next=20e-6,
        )

        # detector layer
        self.detector_layer = DetectorLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            z_coordinate=20e-6,
        )

        # max pool layer
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3)

        # linear layers
        self.linear_0 = torch.nn.Linear(
            in_features=40 * 40, out_features=256, dtype=torch.double
        )
        self.linear_1 = torch.nn.Linear(
            in_features=256, out_features=256, dtype=torch.double
        )
        self.linear_2 = torch.nn.Linear(
            in_features=256, out_features=10, dtype=torch.double
        )

        # relu activation
        self.relu = torch.nn.ReLU()

        # softmax activation
        self.softmax = torch.nn.Softmax(dim=1)

    # the forward pass
    def forward(self, x) -> torch.Tensor:
        # deep diffractive network
        x = self.input_layer(x)
        x = self.diffractive_layer(x)
        x = self.detector_layer(x)

        # reshape the vector to feed into the max pool layer
        if len(x.shape) == 2:
            x = torch.reshape(x, shape=(1, x.shape[0], x.shape[1]))
        if len(x.shape) == 2:
            x = torch.reshape(x, shape=(x.shape[0], 1, x.shape[1], x.shape[2]))

        # max pool
        x = self.max_pool(x)

        # flatten the data to feed into FCC
        x = torch.flatten(x, start_dim=1)

        # FCC network
        x = self.linear_0(x)
        x = self.relu(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class FiveLayerPhaseModulatedDiffractiveNN(torch.nn.Module):
    """Diffractive Neural Network consisting of 5 phase modulated only layers.

    The wavelength of light is set at 1.55 um, the neuron size is always
    120 and the distance between 2 layers is always 10 um.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wavelength = 1.55e-6
        self.neuron_size = 120
        self.x_coordinates = x_coordinates

        # input layer used to propagate optical images
        self.input_layer = InputDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=0,
            z_next=10e-6,
        )

        # diffractive layers
        self.phase_diffractive_layer_0 = PhaseDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=10e-6,
            z_next=20e-6,
        )
        self.phase_diffractive_layer_1 = PhaseDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=20e-6,
            z_next=30e-6,
        )
        self.phase_diffractive_layer_2 = PhaseDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=30e-6,
            z_next=40e-6,
        )
        self.phase_diffractive_layer_3 = PhaseDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=40e-6,
            z_next=50e-6,
        )
        self.phase_diffractive_layer_4 = PhaseDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=50e-6,
            z_next=60e-6,
        )

        # detector layer used to measure the output intensity
        self.detector_layer = DetectorLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            z_coordinate=60e-6,
        )

    # the forward pass
    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.phase_diffractive_layer_0(x)
        x = self.phase_diffractive_layer_1(x)
        x = self.phase_diffractive_layer_2(x)
        x = self.phase_diffractive_layer_3(x)
        x = self.phase_diffractive_layer_4(x)
        x = self.detector_layer(x)
        return x


class FiveLayerBinaryAmplitudeDiffractiveNN(torch.nn.Module):
    """Diffractive Neural Network consisting of 5 Binary Ampltiude
    diffractive layers.

    The wavelength of light is set at 1.55 um, the neuron size is always
    120 and the distance between 2 layers is always 10 um.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wavelength = 1.55e-6
        self.neuron_size = 120
        self.x_coordinates = x_coordinates

        # input layer used to propagate optical images
        self.input_layer = InputDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=0,
            z_next=10e-6,
        )

        # diffractive layers
        self.ba_diffractive_layer_0 = BinaryAmplitudeDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=10e-6,
            z_next=20e-6,
        )
        self.ba_diffractive_layer_1 = BinaryAmplitudeDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=20e-6,
            z_next=30e-6,
        )
        self.ba_diffractive_layer_2 = BinaryAmplitudeDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=30e-6,
            z_next=40e-6,
        )
        self.ba_diffractive_layer_3 = BinaryAmplitudeDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=40e-6,
            z_next=50e-6,
        )
        self.ba_diffractive_layer_4 = BinaryAmplitudeDiffractiveLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            z_coordinate=50e-6,
            z_next=60e-6,
        )

        # detector layer used to measure the output intensity
        self.detector_layer = DetectorLayer(
            n_size=self.neuron_size,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.x_coordinates,
            z_coordinate=60e-6,
        )

    # the forward pass
    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.ba_diffractive_layer_0(x)
        x = self.ba_diffractive_layer_1(x)
        x = self.ba_diffractive_layer_2(x)
        x = self.ba_diffractive_layer_3(x)
        x = self.ba_diffractive_layer_4(x)
        x = self.detector_layer(x)
        return x
