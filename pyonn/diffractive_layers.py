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
from pyonn.angular_spectrum_propagation_method import propagate_complex_amplitude_map
from pyonn.utils import plot_complex_amplitude_map


class DiffractiveLayer(torch.nn.Module):
    """Diffractive Layer with the architecture based on the Lin2018 paper built
    using the PyTorch backend and propagation based on the angular spectrum
    method.

    The diffractive layer's weights represent a matrix with complex valued
    elements that have an amplitude (always between 0 and 1) and phase (always
    between 0 and 2*pi). The forward pass is based on the Angular Spectrum
    method.

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
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        z_coordinate: float,
        z_next: float,
        wavelength: float,
    ) -> None:
        super().__init__()
        self.n_size = n_size
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_coordinate = z_coordinate
        self.z_next = z_next

        # initialize a size x size matrix and instantiate all elements as
        # Parameters. The amplitude must be always smaller than 1.
        weights = torch.randn(size=(n_size, n_size), dtype=torch.complex64)
        self.weights = torch.nn.Parameter(
            torch.divide(weights, torch.max(torch.abs(weights)))
        )

        # the wavelength of light
        self.wavelength = wavelength

    def _clip_weights(self) -> torch.Tensor:
        # always clip the weights to have an absolute values smaller than 1
        # keep only the absolute values who are greater than 1
        new_amplitude_weights = torch.clamp(torch.abs(self.weights), min=1)

        # divide the tensors element wise to get rid of all the weights with
        # amplitude larger than
        new_amplitude_weights = torch.div(self.weights, new_amplitude_weights)

        # return the new amplitude weights
        return new_amplitude_weights

    def plot_weights_map(self) -> None:
        """Plots the intensity and phase map of the weights."""
        # always clip the weights
        clipped_weights = self._clip_weights()

        # copy the weights to a numpy array
        numpy_weights = clipped_weights.detach().cpu().numpy()

        # plot the amplitude3 and phase map
        plot_complex_amplitude_map(
            complex_amplitude_map=numpy_weights,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.y_coordinates,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model based on the angular spectrum propagation method.

        Args:
            x: Tensor representing the values of the output of the layer before
                at the given points where the weights are (has to be the same
                size as the weights). All values must represent a complex
                amplitude.

        Returns:
            Tensor representing the output of this layer at the given points
            of the next layer. The output is simply the complex map measured at
            the next layer.
        """
        # the transmission matrix containing the weights
        transmission_matrix = self._clip_weights()

        # multiply each complex amplitude by the value of a neuron
        complex_amplitude_map = torch.mul(transmission_matrix, x)

        # find the propagated complex amplitude map at z
        propagated_complex_amplitude_map = propagate_complex_amplitude_map(
            complex_amplitude_map=complex_amplitude_map,
            x_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            distance=self.z_next - self.z_coordinate,
        )

        # return the propagated complex amplitude map
        return propagated_complex_amplitude_map


class StaticDiffractiveLayer:
    """Diffractive Layer but the weights are kept as static.

    For a full documentation about attributes see DiffractiveLayer class.

     Attributes:
        weights: Given torch.Tensor. Keep in ind that this static and it
            cannot be trained.
    """

    def __init__(
        self,
        n_size: int,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        z_coordinate: float,
        z_next: float,
        weights: torch.Tensor,
        wavelength: float,
    ) -> None:
        self.n_size = n_size
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_coordinate = z_coordinate
        self.z_next = z_next

        # initialize a size x size matrix and instantiate all elements as
        # Parameters. The amplitude must be always smaller than 1.
        self.weights = weights

        # the wavelength of light
        self.wavelength = wavelength

    def _clip_weights(self) -> torch.Tensor:
        # always clip the weights to have an absolute values smaller than 1
        # keep only the absolute values who are greater than 1
        new_amplitude_weights = torch.clamp(torch.abs(self.weights), min=1)

        # divide the tensors element wise to get rid of all the weights with
        # amplitude larger than
        new_amplitude_weights = torch.div(self.weights, new_amplitude_weights)

        # return the new amplitude weights
        return new_amplitude_weights

    def plot_weights_map(self) -> None:
        """Plots the intensity and phase map of the weights."""
        # always clip the weights
        clipped_weights = self._clip_weights()

        # copy the weights to a numpy array
        numpy_weights = clipped_weights.detach().cpu().numpy()

        # plot the amplitude3 and phase map
        plot_complex_amplitude_map(
            complex_amplitude_map=numpy_weights,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.y_coordinates,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model based on the angular spectrum propagation method.

        Args:
            x: Tensor representing the values of the output of the layer before
                at the given points where the weights are (has to be the same
                size as the weights). All values must represent a complex
                amplitude.

        Returns:
            Tensor representing the output of this layer at the given points
            of the next layer. The output is simply the complex map measured at
            the next layer.
        """
        # the transmission matrix containing the weights
        transmission_matrix = self._clip_weights()

        # multiply each complex amplitude by the value of a neuron
        complex_amplitude_map = torch.mul(transmission_matrix, x)

        # find the propagated complex amplitude map at z
        propagated_complex_amplitude_map = propagate_complex_amplitude_map(
            complex_amplitude_map=complex_amplitude_map,
            x_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            distance=self.z_next - self.z_coordinate,
        )

        # return the propagated complex amplitude map
        return propagated_complex_amplitude_map


class InputDiffractiveLayer:
    """Diffractive Layer but with a given complex amplitude map.

    For a full documentation about attributes see DiffractiveLayer class.

     Attributes:
        complex_amplitude_map: Given torch.Tensor.
    """

    def __init__(
        self,
        n_size: int,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        z_coordinate: float,
        z_next: float,
        complex_amplitude_map: torch.Tensor,
        wavelength: float,
    ) -> None:
        self.n_size = n_size
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_coordinate = z_coordinate
        self.z_next = z_next

        # initialize a size x size matrix and instantiate all elements as
        # Parameters. The amplitude must be always smaller than 1.
        self.complex_amplitude_map = complex_amplitude_map

        # the wavelength of light
        self.wavelength = wavelength

    def plot_complex_amplitude_map(self) -> None:
        """Plots the intensity and phase map of the weights."""
        # make a numpy copy of complex amplitude map
        np_map = self.complex_amplitude_map.detach().cpu().numpy()

        # plot the amplitude3 and phase map
        plot_complex_amplitude_map(
            complex_amplitude_map=np_map,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.y_coordinates,
        )

    def forward(self) -> torch.Tensor:
        """Forward model based on the angular spectrum propagation method.

        Returns:
            Tensor representing the output of this layer at the given points
            of the next layer. The output is simply the complex map measured at
            the next layer.
        """
        # find the propagated complex amplitude map at z
        propagated_complex_amplitude_map = propagate_complex_amplitude_map(
            complex_amplitude_map=self.complex_amplitude_map,
            x_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            distance=self.z_next - self.z_coordinate,
        )

        # return the propagated complex amplitude map
        return propagated_complex_amplitude_map


if __name__ == "__main__":
    from utils import create_square_grid_pattern

    # create a square grid pattern centred on [0, 0] with pixel size 1 um
    # and pixel number 20 (400 pixels in total)
    square_grid_pattern = create_square_grid_pattern(
        center_coordinates=np.array([0, 0]),
        pixel_length=0.8e-6,
        pixel_number=100,
        pixel_separation=0.0,
        grid_z_coordinate=0,
    )

    # the x and y coordinates
    debug_x_coordinates = square_grid_pattern[1]
    debug_y_coordinates = square_grid_pattern[2]

    # retain only the x coordinates of the pattern (necessary for the
    # meshgrid)
    # used only for testing and debugging
    # try the forward pass
    debug_layer = InputDiffractiveLayer(
        n_size=100,
        x_coordinates=debug_x_coordinates,
        y_coordinates=debug_y_coordinates,
        z_coordinate=0.0,
        z_next=1.0,
        complex_amplitude_map=torch.ones(size=(100, 100)),
        wavelength=1.55e-6,
    )

    # plot the amplitude and phase map
    debug_layer.plot_complex_amplitude_map()
