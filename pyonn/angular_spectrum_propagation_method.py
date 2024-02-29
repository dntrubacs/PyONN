"""
Created by Daniel-Iosif Trubacs on 15 February 2024. The purpose of this
module is to create equation to help with the propagation of light.
In this case, light is propagated using the angular spectrum propagation
method and the backend for calculations is PyTorch. For more information
check:
https://docs.google.com/file/d/0B78A_rsP6RDSS3VRWk12Y2FUcVk/edit?resourcekey=0-EdJQY3UFbqEiJnqV8YDPNA
https://github.com/lukepolson/youtube_channel/blob/3642cdd80f9200a5db4e622a3fe2c1a8f6868ecd/Python%20Metaphysics%20Series/vid29.ipynb
"""  # noqa: C901
import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import pickle

# Device configuration (used always fore very torch tensor declared)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def propagate_complex_amplitude_map(
    complex_amplitude_map: torch.Tensor,
    x_coordinates: np.ndarray,
    wavelength: float,
    distance: float,
    folder_name: str = None,
    data_name: str = "propagated_phase_map",
) -> torch.Tensor:
    """Propagates a given phase map for a given distance.

    Each values (or entry) in the complex amplitude map represents a pixel
    who has complex amplitude (amplitude and phase). The map must represent
    square grid who is centred at [0, 0] and placed.

    Also, keep in mind that the complex_amplitude_map is always considers to be
    at coordinate z = 0 and the place where the propagation is calculated will
    be placed at z = distance.

    Args:
        complex_amplitude_map: Torch tensor of shape (n_pixels, n_pixels)
            where each element represents the complex amplitude.
        x_coordinates: Numpy array representing the x coordinates of all
             pixels. Must be of shape (n_pixels, ). As the square is
             centred on [0, 0], the y_coordinates must be the exact same as
             x_coordinates.
        wavelength: The wavelength of light.
        distance: Distance at which the propagated map is calculated (must be
            greater than 0).
        folder_name: Name of the folder where the data will be saved. If None
            is given, the data will not be saved. The data saved will be the
            x_mesh, y_mesh and the numpy array representing the complex
            amplitude map.
        data_name: name of the data to be saved. Defaults to
            'propagated_map'.

    Returns:
        Torch tensor representing the propagate complex amplitude map
        at z = distance.
    """
    # compute th fourier transform of the initial complex amplitude map
    u_0 = torch.fft.fft2(complex_amplitude_map)

    # get the fourier space (fx and fy represent the spacial frequencies)
    fx = torch.fft.fftfreq(n=len(x_coordinates), d=np.diff(x_coordinates)[0])

    # create a mesh grid of fx and fy frequencies. fx and fy are the same
    # as the original grid is a square centred on 0.
    fx_mesh, fy_mesh = torch.meshgrid(fx, fx, indexing="ij")

    # work out the terms that are need in the integral
    # alpha and beta are simply the spacial frequency mesh multiplied by the
    # wavelength
    alpha = wavelength * fx_mesh
    beta = wavelength * fy_mesh

    # the squared term represent the value under the sqrt term.
    # the positive value must be taken such that no negative values are in the
    # square root
    squared_term = torch.abs(1 - torch.square(alpha) - torch.square(beta))

    # take sqrt of the squared term and multiply by k (wave vector)
    # simply take the sqrt of the square
    sqrt_term = (2 * torch.pi / wavelength) * torch.sqrt(squared_term)

    # find the transfer function term and move it to cuda if available
    transfer_term = torch.exp(1j * distance * sqrt_term)
    transfer_term = transfer_term.to(device)

    # propagated complex amplitude map
    propagated_complex_amplitude_map = torch.fft.ifft2(u_0 * transfer_term)

    # save the data if necessary
    if folder_name is not None:
        # generate the mesh grid necessary for saving the data
        x_mesh, y_mesh = np.meshgrid(x_coordinates, x_coordinates)

        # make a directory inside the folder with file name
        folder_save = os.path.join(folder_name, data_name)
        os.mkdir(folder_save)

        # make a numpy copy of the propagated_complex_amplitude_map
        propagated_map_np = propagated_complex_amplitude_map.detach().numpy()

        # save the data inside the folder with pickle
        with open(os.path.join(folder_save, "x_mesh"), "wb") as handle:
            pickle.dump(x_mesh, handle)
        with open(os.path.join(folder_save, "y_mesh"), "wb") as handle:
            pickle.dump(y_mesh, handle)
        with open(
            os.path.join(folder_save, "complex_amplitude_map"), "wb"
        ) as handle:
            pickle.dump(propagated_map_np, handle)

    return propagated_complex_amplitude_map


def find_real_maps(
    complex_amplitude_map: torch.Tensor, normalized: bool = False
) -> tuple:
    """Calculate the intensity and phase map.

    Keep in mind that this function returns a numpy array from a torch tensor.

    Args:
        complex_amplitude_map: Torch tensor representing a complex amplitude
            map (amplitude and phase).
        normalized: Whether to return the intensity map normalized (all values
            are between 0 and 1)

    Returns:
        Tuple containing 2 numpy arrays representing the intensity and
        phase map.
    """
    # make a numpy copy of the complex_amplitude map
    u_np = complex_amplitude_map.detach().cpu().numpy()

    # make the intensity and phase map
    intensity_map = np.square(np.abs(u_np))
    phase_map = np.angle(u_np)

    # normalize the intensity_map if required
    if normalized:
        intensity_map = intensity_map / np.max(intensity_map)

    # return the intensity and phase maps
    return intensity_map, phase_map


def plot_real_maps(
    complex_amplitude_map: torch.Tensor,
    x_coordinates: np.ndarray,
    intensity_map_title: str = "Intensity Map",
    phase_map_title: str = "Phase Map",
    normalized: bool = False,
) -> None:
    """Plots the intensity and phase maps.

    Args:
        complex_amplitude_map: Torch tensor of shape (n_pixels, n_pixels).
            The gird coordinates must represent a square grid centred at
            [0, 0].
        x_coordinates: Numpy array representing the x coordinates of all
             pixels. Must be of shape (n_pixels, ). As the square is
             centred on [0, 0], the y_coordinates must be the exact same as
             x_coordinates.
        intensity_map_title: String representing the title of the intensity
            map figure.
        phase_map_title: String representing the title of the phase
            map figure.
        normalized: Whether to normalize or not the intensity data
    """
    # generate the mesh grid necessary for plotting
    x_mesh, y_mesh = np.meshgrid(x_coordinates, x_coordinates)

    # get the intensity and phase maps
    intensity_map, phase_map = find_real_maps(
        complex_amplitude_map=complex_amplitude_map, normalized=normalized
    )
    # rotate the arrays by 180 degrees (better for plotting)
    intensity_map = np.rot90(intensity_map, k=2)
    phase_map = np.rot90(phase_map, k=2)

    # create the figure
    figure, axis = plt.subplots(1, 2, figsize=(20, 8))

    # plot the intensity map
    axis[0].set_title(intensity_map_title)
    a = axis[0].pcolormesh(x_mesh, y_mesh, intensity_map, cmap="jet")
    axis[0].set_xlabel("$x$ [m]")
    axis[0].set_ylabel("$y$ [m]")
    figure.colorbar(mappable=a)

    # plot the phase map
    axis[1].set_title(phase_map_title)
    b = axis[1].pcolormesh(x_mesh, y_mesh, phase_map, cmap="inferno")
    axis[1].set_xlabel("$x$ [m]")
    axis[1].set_ylabel("$y$ [m]")
    figure.colorbar(mappable=b)
    plt.show()


def find_phase_change(
    n_1: float,
    thickness: float,
    wavelength: float,
    n_2: float = 1.0,
) -> float:
    """Finds the phase change of light passing through a material from
    air.

    This represents the physical value of the phase neuron weight. The phase
    term is simply the exp(j*phi) of the neuron where phi is the phase change
    of light.

    Args:
        n_1: Refractive index of the material. Can be a
            scalar or a matrix containing multiple entries.
        n_2: Refractive index of the material from which the light is coming
            (must be constant)
        thickness: Thickness of the material (measured in meters).
        wavelength: Wavelength of light passing though the material (measured
            in meters).

    Returns:
        Torch tensor representing the phase change in light.
    """
    # squeeze the
    # find out the phase change
    phase_change = (2 * np.pi / wavelength) * (n_1 - n_2) * thickness

    # get the phase change between 0 and 2pi
    phase_change = phase_change % (2 * torch.pi)

    # transfer to [-pi, pi] interval
    phase_change = phase_change - torch.pi

    return phase_change
