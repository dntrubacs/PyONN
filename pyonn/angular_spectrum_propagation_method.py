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
from utils import create_square_grid_pattern
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


if __name__ == "__main__":
    # wavelength
    wavelength = 1.55e-6

    # create a square grid pattern centred on [0, 0] with pixel size 1 um
    # and pixel number 20 (400 pixels in total)
    square_grid_pattern = create_square_grid_pattern(
        center_coordinates=np.array([0, 0]),
        pixel_length=0.8e-6,
        pixel_number=100,
        pixel_separation=0.0,
        grid_z_coordinate=0,
    )

    # generate the meshgrid necessary for the coordinates of the pixels
    pattern, x_coordinates, y_coordinates, z_coordinate = square_grid_pattern
    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)

    # generate a phase map that represents a single silt
    # all pixels have amplitude 1 but different phase
    phase_map = np.zeros(shape=(100, 100), dtype=np.float64) + 1.0
    phase_map = phase_map * np.exp(1j * (-0.1607))
    phase_map[30:70, 50] = np.exp(1j * 2.9361)

    # show the initial phase map
    plt.figure(figsize=(12, 8))
    plt.title("Initial Phase Map")
    plt.pcolormesh(x_mesh, y_mesh, np.angle(phase_map), cmap="jet")
    plt.xlabel("X-Position [m]")
    plt.ylabel("Y-Position [m]")
    plt.colorbar()
    plt.show()

    # move from numpy to torch tensors
    phase_map = torch.from_numpy(phase_map)

    # find the propagated complex amplitude map for different distance
    for dist in [0.1, 1, 2, 5, 10, 20, 50]:
        distance = dist * 1e-6

        # get the propagated phase map at 1 um
        propagated_phase_map = propagate_complex_amplitude_map(
            complex_amplitude_map=phase_map,
            x_coordinates=x_coordinates,
            wavelength=wavelength,
            distance=distance,
        )

        # make a copy of u as a numpy array
        u_np = propagated_phase_map.detach().numpy()

        # plot the intensity and phase map
        figure, axis = plt.subplots(1, 2, figsize=(20, 8))
        axis[0].set_title(
            f"Propagated intensity map at:"
            f" "
            f""
            f"{round(dist, 3)} $\mu$ m"  # noqa W605
        )
        a = axis[0].pcolormesh(x_mesh, y_mesh, np.square(np.abs(u_np)),
                               cmap="jet")
        axis[0].set_xlabel("$x$ [mm]")
        axis[0].set_ylabel("$y$ [mm]")
        figure.colorbar(mappable=a)

        axis[1].set_title(
            f"Propagated phase map at:"
            f" {round(dist, 3)} $\mu$m"  # noqa: W605
        )  # noqa W605
        b = axis[1].pcolormesh(x_mesh, y_mesh, np.angle(u_np), cmap="inferno")
        axis[1].set_xlabel("$x$ [mm]")
        axis[1].set_ylabel("$y$ [mm]")
        figure.colorbar(mappable=b)
        plt.show()
