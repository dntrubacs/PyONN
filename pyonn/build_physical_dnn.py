""" Created by Daniel-Iosif Trubacs on 30 January 2024. The purpose
of this module is to build physical DNN by transforming the phase and
transmission values of neurons to different refractive indices and depths. """

import torch
from matplotlib import pyplot as plt
from diffraction_equations import *
import numpy as np
from utils import create_square_grid_pattern

# define device as a global variable to use the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PhysicalDiffractiveLayer:
    """ Physical implementation of a diffractive layer.

    Each physical neuron represents a square of a certain material with
    a given refractive index. The layer will represent a square of
    neurons_number x neuron_number. This implementation is made
    to recreate the paper: Non-volatile Reconfigurable Digital Optical
    Diffractive Neural Network Based on Phase Change Material.

    Attributes:
        neuron_coordinates: Torch tensor that contains all cordinates of
            each neuron. The shape must
            (neuron_number, neuron_number, 3) where each entry represents
            the (x, y, z) coordinates. The z must be the same for
            all neurons.
        material_refractive_index: Torch tensor representing a matrix where
            each entry represents the refractive index of a neuron. keep in
            mind that each entry material_refractive_index[i][j] must
            correspond to neuron_coordinates[i][j]
        neuron_thickness: Thickness of one individual neuron.
        neuron_number: Number of neurons in one collumn (or row).
        neurons_z_coordinate: The z coordinate of all neurons.
        wavelength: Wavelength of light (in meters).
        phase_map: Phase value for each of the neurons.
        optical_modes: Optical mode of each neuron.
    """
    def __init__(self,
                 neuron_coordinates: torch.Tensor,
                 material_refractive_index: torch.Tensor,
                 neuron_thickness: float,
                 wavelength: float = 1.55E-6) -> None:
        # neuron features
        self.neuron_coordinates = neuron_coordinates
        self.material_refractive_index = material_refractive_index
        self.wavelength = wavelength
        self.neuron_thickness = neuron_thickness
        self.neuron_number = self.neuron_coordinates.shape[0]
        self.neurons_z_coordinate = self.neuron_coordinates[0][0][2]

        # phase  map and optical modes
        self.phase_map = self.find_phase_values()
        self.optical_modes = self.find_neuron_optical_modes()

    def find_phase_values(self) -> torch.Tensor:
        """Finds the phase values and optical modes of each neuron.

        For now the substrate is taken to be air.

        Returns:
            Torch tensor representing the phase of each neuron.
        """
        phase_map = find_phase_change(
            n_1=self.material_refractive_index,
            n_2=1,
            thickness=self.neuron_thickness,
            wavelength=self.wavelength
        )
        return phase_map

    def find_neuron_optical_modes(self) -> torch.Tensor:
        """ Find the corresponding optical mode of each neuron.

        The optical mode is simply e^(j*phase_change) for each neuron.
        For now, the amplitude transmission for each neuron is
        considered to be 1).

        Returns:
            Torch tensor representing the optical modes foe ach neuron
        """
        # complex number j
        complex_i = torch.tensor(1.j, dtype=torch.cfloat, device=device)

        # get the optical modes
        optical_modes = torch.exp(complex_i*self.phase_map)

        # return the optical modes
        return optical_modes

    def plot_phase_map(self) -> None:
        """Plots the phase map of all neurons."""
        # show the figure
        plt.figure(figsize=(12, 8))
        plt.title('Phase Map (measured in radians)')
        plt.imshow(X=self.phase_map.detach().cpu().numpy())
        plt.colorbar()
        plt.show()

    def find_output(self, n_pixels, z_distance) -> torch.Tensor:
        detector_positions = find_neuron_coordinates(
            first_neuron_coordinates=np.array([-1.6E-6, 1.6E-6]),
            neurons_number=330, neuron_length=0.01E-6, neurons_separation=0,
            z_coordinate=3E-6)

        detector_positions = torch.from_numpy(detector_positions).to(device)
        source_positions = torch.from_numpy(self.neuron_coordinates_matrix).to(device)

        output = find_optical_modes(source_positions=source_positions,
                                    detector_positions=detector_positions,
                                    source_optical_modes=self.optical_modes,
                                    wavelength=self.wavelength)
        intensity_map = find_intensity_map(output)
        plt.imshow(intensity_map.detach().cpu().numpy(), cmap='jet')
        plt.colorbar(cmap='jet')
        plt.show()


if __name__ == '__main__':
    c_sb2se3 = 4.04933
    a_sb2se3 = 3.285356

    # create a map of 3x3 refractive indices amorphous and crystalline Sb2Se3
    debug_refractive_indices = torch.tensor([[a_sb2se3, c_sb2se3, a_sb2se3],
                                             [c_sb2se3, a_sb2se3, c_sb2se3],
                                             [a_sb2se3, c_sb2se3, a_sb2se3]],
                                            device=device)

    # the coordinates for each neuron
    debug_neuron_coordinates = create_square_grid_pattern(
        center_coordinates=np.array([0, 0]),
        pixel_length=0.8E-6,
        pixel_separation=0.2E-6,
        pixel_number=5,
        grid_z_coordinate=1E-6
    )

    # create a torch tensor and move it to the device
    debug_neuron_coordinates = torch.from_numpy(
        debug_neuron_coordinates).to(device)

    debug_pdl = PhysicalDiffractiveLayer(
        neuron_coordinates=debug_neuron_coordinates,
        neuron_thickness=1E-6,
        wavelength=1.55E-6,
        material_refractive_index=debug_refractive_indices)

    debug_pdl.find_output(n_pixels=10, z_distance=3E-6)


