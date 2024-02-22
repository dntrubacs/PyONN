# flake8: noqa
"""
Created by Daniel-Iosif Trubacs on 21 February 2024 for the UoS Integrated
Nanophotonics Group. The purpose of this module is to create a diffractive
layer class (based on the PyTorch architecture) DiffractiveLayer to be used on
the simulation of optical neural networks.

The current diffractive layers are only phase modulated and represent
only scalar simulation based on the Fourier optics method. All the
equations used as a backend for the propagation of light are available in the
angular_spectrum_propagation_method module.

For more information about the mathematical algorithms behind the optical
neural network used please check the following References:

Qian, C., Lin, X., Lin, X. et al. Performing optical logic operations by a
diffractive neural network. Light Sci Appl 9, 59 (2020).
https://doi.org/10.1038/s41377-020-0303-2

1. C. Wu, J. Zhao, Q. Hu, R. Zeng, and M. Zhang, Non-volatile
reconfigurable digital optical diffractive neural network based on
phase change material, 2023. DOI:10 . 48550 / ARXIV. 2305 . 11196.
"""
import numpy as np
import torch
from utils import find_coordinate_matrix
from matplotlib import pyplot as plt
from angular_spectrum_propagation_method import *


class DiffractiveLayer(torch.nn.Module):
    """Diffractive Layer with the architecture based on the Lin2018 paper built
    using the PyTorch backend and propagation based on the angular spectrum
    method.

    The diffractive layer's weights represent a matrix with complex valued
    elements that have an amplitude (always between 0 and 1) and phase (always
    between 0 and 2*pi). The forward pass is based on the Angular Spectrum
    method. In this case, the amplitude of each neuron is always set to 1.

    Keep in mind that in a neural network composed of multiple Diffractive
    Layers, all neuron and x and y coordinates must be the same. Also,
    all diffractive layers must have the same number of neurons.

    Attributes:
        n_size: The numbers of neurons in a given column or row (the total
            number of neurons will be n_size x n_size). The values of the
            neurons is complex, and it will always have the form:
            e^(j*phase).
        x_coordinates: The x coordinates of all neurons. Must be a torch
            tensor of shape (n_size, ).
        y_coordinates: The x coordinates of all neurons. Must be a torch
            tensor of shape (n_size, ).
        z_coordinate: The z coordinated of the layer implemented (corresponding
            to the physical implementation). Keep in mind that all neurons will
            have this z coordinates as their position.
        weights: torch.nn.Parameter object containing a size x size
            matrix with the phase valued elements.
        z_next: The z coordinate of the next layer (used to
            find the complex amplitude map at the next layer).
        wavelength: The wavelength of light.
    """

    def __init__(
        self,
        n_size: int,
        x_coordinates: torch.Tensor,
        y_coordinates: torch.Tensor,
        z_coordinate: float,
        z_next: float,
        wavelength: float,
    ) -> None:
        super().__init__()
        self.n_size = n_size
        self.x_coordinates = self.y_coordinates
        self.y_coordinates = self.y_coordinates
        self.z_coordinate = z_coordinate
        self.z_next = z_next

        # initialize a size x size matrix and instantiate all elements as
        # Parameters
        self.weights = torch.nn.Parameter(
            torch.randn(size=(n_size, n_size), dtype=torch.float64)
        )

        # the wavelength of light
        self.wavelength = wavelength

    def _get_phase_map(self) -> np.ndarray:
        """Gets the phase map of the neurons weights.

        Returns:
            Numpy arrays of shape (size, size) containing the phase
            value of each weight.
        """
        # copy the weights to a numpy array
        numpy_weights = self.weights.detach().cpu().numpy()

        # get all the phases between 0 and 2pi
        phase_map = numpy_weights % (2 * np.pi)

        # transfer to [-pi, pi] interval
        phase_map = phase_map - np.pi

        return phase_map

    def plot_phase_map(self) -> None:
        """Plots the phase map."""
        # if the number of neurons is greater than 15, the labels get
        # too crowded, so show only 15 values.
        if self.size < 15:
            x_ticks = (
                np.arange(start=0, stop=self.size) + 0.5
            ) * self.neuron_length
            y_ticks = (
                np.arange(start=0, stop=self.size) + 0.5
            ) * self.neuron_length

        else:
            x_ticks = (
                np.linspace(start=0, stop=self.size, num=15) + 0.5
            ) * self.neuron_length
            y_ticks = (
                np.linspace(start=0, stop=self.size, num=15) + 0.5
            ) * self.neuron_length

        # show the figure
        plt.figure(figsize=(12, 8))
        plt.title("Phase Map (measured in radians)")
        plt.xlabel("x-position (m)")
        plt.ylabel("y-position (m)")
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.imshow(
            X=self._get_phase_map(),
            origin="lower",
            extent=[0, self.length, 0, self.length],
        )
        plt.colorbar()
        plt.show()

    def forward(self, x) -> torch.Tensor:
        """Forward model based on the Rayleigh-Sommerfeld model. Please
        check the reference supplementary for more details.

        Args:
            x: Tensor representing the values of the output of the layer before
                at the given points where the weights are (has to be the same
                size as the weights).

        Returns:
            Tensor representing the output of this layer at the given points
            of the next layer.
        """
        # the transmission matrix containing the weights
        transmission_matrix = self._clip_weights()

        # source optical modes are the optical modes propagated from the layer
        # before (x) times the neuron valued weight (amplitude and phase)
        neuron_optical_modes = torch.mul(transmission_matrix, x)

        # find the optical modes at the next layer
        # in this case the detector position are simply the positions of the
        # neurons in the next layer
        optical_modes = find_optical_modes(
            source_positions=self.neuron_coordinates,
            detector_positions=self.neuron_coordinates_next,
            source_optical_modes=neuron_optical_modes,
            wavelength=self.wavelength,
        )

        # return the intensity map
        return optical_modes


class InputLayer(torch.nn.Module):
    """Input layer used for Diffractive neural networks.

    This diffractive layer transforms black and white image to an
    'equivalent source of light' where each pixel is represented by a neuron
    with amplitude value 1 and phase 0. The forward method will find the
    optical modes of light at the first diffractive layer.

    Keep in mind that this layer is not trainable.

    Attributes:
        size: The numbers of neurons in a given column or row (the total
            number of neurons will be n_size x n_size). This must be equivalent
            with the size of the image.
        length: The length of the matrix (corresponding to a physical
            implementation of the layer). This is used to find the coordinates
            of each neuron.
        neuron_length: The length of one physical neuron (length/size).
        weights: Torch tensor object containing a size x size
            matrix with the complex valued amplitude (phase is always 0).
        z_coordinate: The z coordinated of the layer implemented (corresponding
            to the physical implementation). Keep in mind that all neurons will
            have this z coordinates as their position.
        neuron_coordinates: Tensor of shape (size, size, 3) representing the
            position of all neurons (x, y, z). See utils.find_coordinate_matrix
            for more information.
        neuron_coordinates_next: Tensor of shape (size, size, 3) representing
            the position of all neurons (x, y, z) in the first diffractive
            layer. See utils.find_coordinate_matrix for more information.
        z_next: The z coordinate of the first diffractive layer.
        wavelength: The wavelength of light.
        size_next: The number of neurons in the first diffractive layer.
    """

    def __init__(
        self,
        size: int,
        length: float,
        z_coordinate: float,
        z_next: float,
        size_next: int,
        wavelength: float,
    ) -> None:
        super().__init__()
        self.size = size
        self.length = length
        self.neuron_length = self.length / self.size
        self.z_coordinate = z_coordinate
        self.z_next = z_next
        self.size_next = size_next
        # initialize as None as this are not trainable
        self.weights = None

        # the position of each neuron
        self.neuron_coordinates = torch.from_numpy(
            find_coordinate_matrix(
                n_size=self.size,
                n_length=self.length,
                z_coordinate=self.z_coordinate,
            )
        )

        # the position of each neuron in the next layer
        self.neuron_coordinates_next = torch.from_numpy(
            find_coordinate_matrix(
                n_size=self.size_next,
                n_length=self.length,
                z_coordinate=self.z_next,
            )
        )

        # the wavelength of light
        self.wavelength = wavelength

    def forward(self, x) -> torch.tensor:
        """Finds the optical mode generated by the input source at the
        position of the first diffractive layer's neurons.
        """

        # resize all pixel values between 0 and 1
        self.weights = torch.divide(x, torch.max(x))

        # find the optical modes at the next layer
        # in this case the detector position are simply the positions of the
        # neurons in the next layer
        optical_modes = find_optical_modes(
            source_positions=self.neuron_coordinates,
            detector_positions=self.neuron_coordinates_next,
            source_optical_modes=self.weights,
            wavelength=self.wavelength,
        )

        # return the optical mode
        return optical_modes


class DetectorLayer(torch.nn.Module):
    """Detector layer used as the last layer in a diffractive neural network.

    This layer simply return the intensity map obtained by a physical
    detector. In this case, each pixel simply represents an object that
    can measure the intensity of light. In this case the forward method is
    static as the output is only the intensity of a given optical mode.

    Attributes:
        size: The numbers of pixels in a given column or row (the total
            number of neurons will be n_size x n_size) that the detector has.
        length: The length of the matrix (corresponding to a physical
            implementation of the detector). This is used to find the
            coordinates of each pixel.
        pixel_length: The length of one physical neuron (length/size).
        z_coordinate: The z coordinated of the detector implemented
            (corresponding to the physical implementation).
        pixel_coordinates: Tensor of shape (size, size, 3) representing the
            position of all neurons (x, y, z). See utils.find_coordinate_matrix
            for more information.
    """

    def __init__(self, size: int, length: float, z_coordinate: float) -> None:
        super().__init__()
        self.size = size
        self.length = length
        self.pixel_length = self.length / self.size
        self.z_coordinate = z_coordinate

        # the position of each pixel
        self.pixel_coordinates = torch.from_numpy(
            find_coordinate_matrix(
                n_size=self.size,
                n_length=self.length,
                z_coordinate=self.z_coordinate,
            )
        )

    @staticmethod
    def forward(x: torch.tensor) -> torch.tensor:
        """Simply return the intensity map from a given optical mode
        matrix."""
        intensity_map = find_intensity_map(x)

        return intensity_map


if __name__ == "__main__":
    # used only for testing and debugging
    # try the forward pass
    debug_input = torch.ones(size=(40, 40))
    debug_input_layer = InputLayer(
        size=40,
        length=1,
        z_coordinate=0.0,
        z_next=1.0,
        size_next=40,
        wavelength=0.652,
    )
    debug_diffractive_layer = DiffractiveLayer(
        size=40, length=1, z_coordinate=1.0, z_next=2.0, wavelength=0.652
    )

    debug_detector_layer = DetectorLayer(size=40, length=1.0, z_coordinate=2.0)

    output = debug_input_layer(debug_input)
    output = debug_diffractive_layer(output)
    debug_intensity_map = debug_detector_layer(output)
    show_intensity_map = debug_intensity_map.detach().cpu().numpy()
    plt.imshow(show_intensity_map)
    plt.show()
