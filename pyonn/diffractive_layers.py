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
from pyonn.angular_spectrum_propagation_method import (
    propagate_complex_amplitude_map,
)
from pyonn.utils import plot_complex_amplitude_map
from typing import Optional
from matplotlib import pyplot as plt


class DiffractiveLayerCommon(torch.nn.Module):
    """Base class for all types of diffractive layers. All Diffractive Layers
    must inherit from this class and overwrite the forward pass.

    Diffractive Layer with the architecture based on the Lin2018 paper built
    using the PyTorch backend and propagation based on the angular spectrum
    method. The diffractive layer's weights represent a matrix with complex
    valued elements that have an amplitude (always between 0 and 1) and phase
    (always between 0 and 2*pi). The forward pass is based on the Angular
    Spectrum method.

    The weights can be either static (non-trainable) or dynamic
    (trainable). If the weights are trainable, then they will always be
    initialized randomly. To avoid confusion (and keep with PyTorch) notation
    the dynamic weights are simply called weights and the static weights
    are called weights_static.

    Attributes:
        n_size: The numbers of neurons in a given column or row (the total
            number of neurons will be n_size x n_size). The values of the
            neurons is complex, and it will always have the form:
            e^(j*phase).
        x_coordinates: The x coordinates of all neurons. Must be a numpy
            array of shape (n_size, ).
        y_coordinates: The x coordinates of all neurons. Must be a numpy
            array of shape (n_size, ).
        z_coordinate: The z coordinated of the layer implemented (corresponding
            to the physical implementation). Keep in mind that all neurons will
            have this z coordinates as their position.
        weights_static: torch.Tensor object containing a size x size
            matrix with the phase valued elements. These values are static
            are non-trainable. These values are always given as input.
        weights: torch.nn.Parameter object containing a size x size
            matrix with the phase valued elements.These values are dynamic,
            and they are the trainable parameter of the network. These values
            are always initialized randomly.
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
        weights_static: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        # the parameters characterizing the architecture of the layer.
        self.n_size = n_size
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_coordinate = z_coordinate
        self.z_next = z_next
        self.weights_static = weights_static

        # initialize a size x size matrix and instantiate all elements as
        # Parameters. The amplitude must be always smaller than 1.
        weights = torch.randn(size=(n_size, n_size), dtype=torch.complex64)
        self.weights = torch.nn.Parameter(
            torch.divide(weights, torch.max(torch.abs(weights)))
        )

        # the wavelength of light
        self.wavelength = wavelength

    def _plot_complex_map(self, weights_map: torch.Tensor) -> None:
        """Plots the amplitude and phase map of the weights.

        Args:
            weights_map: torch.Tensor representing either weights or
                weight_static.
        """
        # copy the weights to a numpy array
        numpy_weights = weights_map.detach().cpu().numpy()

        # plot the amplitude3 and phase map
        plot_complex_amplitude_map(
            complex_amplitude_map=numpy_weights,
            x_coordinates=self.x_coordinates,
            y_coordinates=self.y_coordinates,
        )

    def forward(self, x: torch.Tensor) -> None:
        """This method represents the forward pass, and it always should
        be overwritten."""
        pass


class StaticDiffractiveLayer(DiffractiveLayerCommon):
    """Diffractive Layer but the weights are kept as static.

    For a full documentation about attributes see DiffractiveLayerCommon class.

     Attributes:
        weights_static: Given torch.Tensor. Keep in ind that this is
            static and it cannot be trained.
    """

    def __init__(
        self,
        n_size: int,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        z_coordinate: float,
        z_next: float,
        wavelength: float,
        weights_static: torch.Tensor,
    ) -> None:
        super().__init__(
            n_size,
            x_coordinates,
            y_coordinates,
            z_coordinate,
            z_next,
            wavelength,
            weights_static,
        )
        self.n_size = n_size
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_coordinate = z_coordinate
        self.z_next = z_next

        # initialize a size x size matrix and instantiate all elements as
        # Parameters. The amplitude must be always smaller than 1.
        self.weights_static = weights_static

        # the wavelength of light
        self.wavelength = wavelength

    def plot_weights_map(self) -> None:
        """Plots the intensity and phase map of the weights."""
        self._plot_complex_map(weights_map=self.weights_static)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model based on the angular spectrum propagation method.

        Args:
            x: Tensor representing the values of the output of the layer before
                at the given points where the weights are (has to be the same
                size as the weights). All values must represent a complex
                amplitude (a spatial spectrum).

        Returns:
            Tensor representing the output of this layer at the given points
            of the next layer. The output is simply the complex map measured at
            the next layer.
        """
        # multiply each complex amplitude by the value of a neuron
        initial_spatial_spectrum = torch.mul(self.weights_static, x)

        # find the propagated complex amplitude map at z
        propagated_spacial_spectrum = propagate_complex_amplitude_map(
            complex_amplitude_map=initial_spatial_spectrum,
            x_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            distance=self.z_next - self.z_coordinate,
        )

        # return the propagated complex amplitude map
        return propagated_spacial_spectrum


class InputDiffractiveLayer(DiffractiveLayerCommon):
    """Input layer for Diffractive Networks.

    This layer does not require weights and it can be considered as the
    'source of light' in a Deep Diffractive Neural Network.
    For a full documentation about attributes see the
    DiffractiveLayerCommon class.
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
        super().__init__(
            n_size,
            x_coordinates,
            y_coordinates,
            z_coordinate,
            z_next,
            wavelength,
        )

        # the parameters characterizing the architecture of the layer.
        self.n_size = n_size
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_coordinate = z_coordinate
        self.z_next = z_next

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model based on the angular spectrum propagation method.

        Args: Should be a complex amplitude map that is considered as
            the source of light.

        Returns:
            Tensor representing the output of this layer at the given points
            of the next layer. The output is simply the complex map measured at
            the next layer.
        """
        # find the propagated complex amplitude map at z
        propagated_spatial_spectrum = propagate_complex_amplitude_map(
            complex_amplitude_map=x,
            x_coordinates=self.x_coordinates,
            wavelength=self.wavelength,
            distance=self.z_next - self.z_coordinate,
        )

        # return the propagated complex amplitude map
        return propagated_spatial_spectrum


class DiffractiveLayer(DiffractiveLayerCommon):
    """Generic Diffractive Layer used for training models.

    use this class when training a model. For a full documentation about
    all the attributes see the DiffractiveLayerCommon class.
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
        # the parameters characterizing the architecture of the layer.
        super().__init__(
            n_size,
            x_coordinates,
            y_coordinates,
            z_coordinate,
            z_next,
            wavelength,
        )
        self.n_size = n_size
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_coordinate = z_coordinate
        self.z_next = z_next

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

        # plot the map
        self._plot_complex_map(weights_map=clipped_weights)

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


class DetectorLayer(torch.nn.Module):
    """Simply return the intensity map from a given complex amplitude map.

    For a full documentation about attributes see DiffractiveLayer class.

     Attributes:
        n_size: The numbers of pixels used to detect (equivalent to the
            number of neurons in the layer before). Used only for checking
            that the size of the input complex field is correct.
        x_coordinates: The x coordinates of all pixels. Must be a numpy
            array of shape (n_size, ). Used only for plotting.
        y_coordinates: The x coordinates of all pixels. Must be a numpy
            array of shape (n_size, ). Used only for plotting.
        z_coordinate: Z coordinate of all the pixels.
    """

    def __init__(
        self,
        n_size: int,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        z_coordinate: float,
    ) -> None:
        super().__init__()
        self.n_size = n_size
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.z_coordinate = z_coordinate

        # the intensity map is initialized as None
        self.intensity_map = None

    def forward(self, x: torch.tensor) -> torch.Tensor:
        """Forward model based on the angular spectrum propagation method.

        x: Tensor representing the values of the output of the layer before
           at the given points where the weights are (has to be the same
           size as the weights). All values must represent a complex
           amplitude (a spatial spectrum).

        Returns:
            Torch tensor representing the intensity map of x.
        """
        # check that the size of the detector and input complex map is the same
        # if (x.shape[1], x.shape[2]) != (self.n_size, self.n_size):
        #  raise Exception(
        #     "InputError! Shape of complex map input"
        #      "and Detector are different"
        # )

        # save the intensity_map
        self.intensity_map = torch.square(torch.abs(x))

        # normalize the intensity map
        self.intensity_map = self.intensity_map / torch.max(self.intensity_map)

        # return the intensity map
        return self.intensity_map

    def plot_intensity_map(self) -> None:
        """Plot the intensity map of the given field."""
        # make a copy of the intensity map in numpy
        intensity_map_np = self.intensity_map.detach().cpu().numpy()

        # make a mesh for the plot
        x_mesh, y_mesh = np.meshgrid(self.x_coordinates, self.y_coordinates)

        # create the figure
        figure = plt.figure(figsize=(12, 8))

        # plot the intensity map
        plt.title("Intensity Map Observed at the Detector")
        plot_map = plt.pcolormesh(x_mesh, y_mesh, intensity_map_np, cmap="jet")
        plt.xlabel("$x$ [m]")
        plt.ylabel("$y$ [m]")
        figure.colorbar(mappable=plot_map)
        plt.show()


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
        wavelength=1.55e-6,
    )

    # plot the amplitude and phase map
    debug_layer.plot_complex_amplitude_map()
