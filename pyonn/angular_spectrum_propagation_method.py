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


def propagate_complex_amplitude_map(
    complex_amplitude_map: torch.Tensor,
    x_coordinates: np.ndarray,
    wavelength: float,
    distance: float,
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

    Returns:
        Torch tensor representing the propagate complex amplitude map
        at z = distance.
    """
    # compute th fourier transform of the initial complex amplitude map
    u_0 = torch.fft.fft2(complex_amplitude_map)

    # get the fourier space (fx and fy represent the spacial frequencies)
    fx = torch.fft.fftfreq(n=len(x_coordinates), d=np.diff(x_coordinates)[0])

    # create a meshgrid of fx and fy frequencies. fx and fy are the same
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

    # find the transfer function term
    transfer_term = torch.exp(1j * distance * sqrt_term)

    return torch.fft.ifft2(u_0 * transfer_term)


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
    """
    # generate the meshgrid necessary for plotting
    x_mesh, y_mesh = np.meshgrid(x_coordinates, x_coordinates)

    # get the intensity and phase maps
    intensity_map, phase_map = find_real_maps(
        complex_amplitude_map=complex_amplitude_map, normalized=True
    )

    # create the figure
    figure, axis = plt.subplots(1, 2, figsize=(20, 8))

    # plot the intensity map
    axis[0].set_title(intensity_map_title)
    a = axis[0].pcolormesh(x_mesh, y_mesh, intensity_map, cmap="jet")
    axis[0].set_xlabel("$x$ [mm]")
    axis[0].set_ylabel("$y$ [mm]")
    figure.colorbar(mappable=a)

    # plot the phase map
    axis[1].set_title(phase_map_title)
    b = axis[1].pcolormesh(x_mesh, y_mesh, phase_map, cmap="inferno")
    axis[1].set_xlabel("$x$ [mm]")
    axis[1].set_ylabel("$y$ [mm]")
    figure.colorbar(mappable=b)
    plt.show()


if __name__ == "__main__":
    from utils import create_square_grid_pattern
    from diffraction_equations import find_phase_change

    # wavelength
    debug_wavelength = 1.55e-6

    # create a square grid pattern centred on [0, 0] with pixel size 1 um
    # and pixel number 20 (400 pixels in total)
    square_grid_pattern = create_square_grid_pattern(
        center_coordinates=np.array([0, 0]),
        pixel_length=0.8e-6,
        pixel_number=100,
        pixel_separation=0.0,
        grid_z_coordinate=0,
    )

    # retain only the x coordinates of the pattern (necessary for the
    # meshgrid)
    debug_x_coordinates = square_grid_pattern[1]
    pattern = square_grid_pattern[0]

    # find the phase change for crystalline and amorphous Sb2Se3 film of
    # thickness 1 um
    amorphous_phase_change = find_phase_change(
        n_1=3.28536, thickness=1e-6, wavelength=1.55e-6, n_2=1.0
    )
    crystalline_phase_change = find_phase_change(
        n_1=4.04933, thickness=1e-6, wavelength=1.55e-6, n_2=1.0
    )

    # generate a phase map that represents a single silt
    # all pixels have amplitude 1 but different phase
    debug_map = np.zeros(shape=(100, 100), dtype=np.float64) + 1.0

    # create a complex amplitude map for the phase change material
    # e^(j*phase)
    debug_map = debug_map * np.exp(1j * amorphous_phase_change)
    debug_map[30:70, 50] = np.exp(1j * crystalline_phase_change)

    # move from numpy to torch tensors
    debug_map = torch.from_numpy(debug_map)

    # show the initial phase map
    plot_real_maps(
        complex_amplitude_map=debug_map,
        x_coordinates=debug_x_coordinates,
        intensity_map_title="Initial Intensity Map",
        phase_map_title="Initial Phase Map",
    )

    # find the propagated complex amplitude map for different distance
    for dist in [0.1, 1, 2, 5, 10, 20, 50]:
        resized_dist = dist * 1e-6

        # get the propagated phase map at 1 um
        propagated_phase_map = propagate_complex_amplitude_map(
            complex_amplitude_map=debug_map,
            x_coordinates=debug_x_coordinates,
            wavelength=debug_wavelength,
            distance=resized_dist,
        )

        plot_real_maps(
            complex_amplitude_map=propagated_phase_map,
            x_coordinates=debug_x_coordinates,
            intensity_map_title=f"Propagated intensity map at: "
            f"{round(dist, 3)} $\mu$ m",  # noqa W605
            phase_map_title=f"Propagated phase map at: "
            f"{round(dist, 3)} $\mu$ m",  # noqa W605
        )
