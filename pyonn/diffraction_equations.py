"""
Created by Daniel-Iosif Trubacs on 3 January 2024 for the UoS Integrated
Nanophotonics Group. The purpose of this module is to create certain
diffraction equations based on the Rayleigh-Sommerfeld model. The equation
for the optical w at a position (x,y,z) generated from a source at position
(x_i, z_i, z_i) and wavelength lambda is:
w = (z-z_i)/r**2 * (1/(2*pi*r) + 1/(lambda*j))*exp(j*2*pi*r/lambda) (1)
  = factor*(inverse_distance + inverse_wavelength)*exponential_term

where r = sqrt((x-x_i)**2 + (y-y_i)**2 + (z-z_i)**2) and j is the imaginary
number sqrt(-1). If the calculation take into account the source optical
mode s (complex number representing amplitude and phase), then the optical
new optical mode n will simply be:
n = w*s

All the calculations are done using the torch backend as this integrates easier
with neural network architecture.

For more information about the mathematical algorithms behind the optical
neural network used please check the following References:

1. https://www.science.org/doi/10.1126/science.aat8084#supplementary-materials
"""

import torch

# define device as a global variable to use the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_optical_modes(
        source_positions: torch.Tensor,
        detector_positions: torch.Tensor,
        source_optical_modes: torch.Tensor,
        wavelength: torch.cfloat) -> torch.Tensor:
    """Finds the optical modes at multiple positions from multiple sources.

    Args:
        source_positions: Torch tensor representing the coordinates of the
            sources (x, y, z) having shape (M, N, 3).
        detector_positions: Torch tensor representing the coordinates of the
            detectors (x_i, y_i, z_i) for each source, having shape (K, L, 3).
        source_optical_modes: Torch tensor representing the source optical
            modes (complex number representing amplitude and phase) for each
            source, having shape (M, N).
        wavelength: Wavelength of the light.

    Returns:
        Torch tensor representing the optical modes at the detector positions.
        The optical mode is complex-valued and the returned shape is (N, M).
    """
    # complex unit 'j'
    complex_i = torch.tensor(1.j, dtype=torch.cfloat, device=device)

    # Broadcast detector positions to have the same shape as source positions
    # detector array of shape (K,L)
    # source array of shape (M,N)
    # shape will be (1, 1, M, N, 3)
    pos_src_broadcast = source_positions.unsqueeze(0).unsqueeze(
        0).to(device)
    # shape will be (K, L, 1, 1, 3)
    pos_det_broadcast = detector_positions.unsqueeze(2).unsqueeze(
        2).to(device)
    # shape will be (1, 1, M, N, 3)
    modes_src_broadcast = source_optical_modes.unsqueeze(0).unsqueeze(
        0).to(device)

    # Calculate the radial distance r for each source-detector pair
    squared_diff = (pos_src_broadcast - pos_det_broadcast) ** 2
    # sum along the last dimension (x, y, z)
    r = torch.sqrt(torch.sum(squared_diff,
                             dim=-1))

    # z-distance, z/r^2 factor, inverse distance, and inverse wavelength
    z_diff = pos_src_broadcast[..., 2] - pos_det_broadcast[..., 2]
    zr2_factor = z_diff / r ** 2
    inverse_distance = 1 / (2 * torch.pi * r)
    inverse_wavelength = 1 / (complex_i * wavelength)

    # phase term
    exponential_term = torch.exp(complex_i * 2 * torch.pi * r / wavelength)

    # contribution from all sources for each detector
    w = zr2_factor * (inverse_distance + inverse_wavelength) * exponential_term
    w_s = w * modes_src_broadcast

    # `w_s` is now a large array of shape (K, L, M, N), containing the complex
    # amplitude of every (MxN) source's contribution to every (KxL) detector

    # superpose fields: for each detector, sum over source positions the total,
    # local, complex amplitude
    # dimensions 0 and 1 are detector pos.
    total_amplitude = w_s.sum(
        dim=(2, 3))

    return total_amplitude


def find_intensity_map(optical_modes: torch.tensor) -> torch.tensor:
    """ Finds the normalized light intensities from a given optical mode map.

    Args:
        optical_modes: Torch tensor representing the matrix of optical
            modes.

    Returns:
        Torch tensor representing the light intensity of optical modes.
    """
    # maximum value of the optical modes value
    max_value = torch.max(torch.abs(optical_modes))

    # intensity map
    intensity_map = torch.abs(optical_modes)
    intensity_map = (torch.divide(intensity_map, max_value)) ** 2

    # return the intensity map
    return intensity_map


def find_phase_change(refractive_index: torch.Tensor, thickness: float,
                      wavelength: float) -> torch.Tensor:
    """ Finds the phase change of light passing through a material from
    air.

    This represents the physical value of the phase neuron weight. The phase
    term is simply the exp(j*phi) of the neuron where phi is the phase change
    of light.

    Args:
        refractive_index: Refractive index of the material. Can be a
            scalar or a matrix containing multiple entries.
        thickness: Thickness of the material (measured in meters).
        wavelength: Wavelength of light passing though the material (measured
            in meters).

    Returns:
        Torch tensor representing the phase change in light.
    """
    # squeeze the
    # find out the phase change
    phase_change = (2 * torch.pi / wavelength) * (
                refractive_index - 1) * thickness

    # get the phase change between o and 2pi
    phase_change = phase_change % (2*torch.pi)

    return phase_change


if __name__ == '__main__':
    debug_refractive_indices = torch.tensor([[3.28536, 4.0493, 3.28536],
                                             [3.28536, 4.0493, 3.28536]],
                                            device=device)
    debug_phase = find_phase_change(refractive_index=debug_refractive_indices,
                                    thickness=1E-6,
                                    wavelength=1.55E-6)
    print(debug_phase)
