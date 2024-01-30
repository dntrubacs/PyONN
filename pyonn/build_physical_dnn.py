""" Created by Daniel-Iosif Trubacs on 30 January 2024. The purpose
of this module is to build physical DNN by transforming the phase and
transmission values of neurons to different refractive indices and depths. """

import torch
from matplotlib import pyplot as plt
from diffraction_equations import find_phase_change, find_optical_modes, find_intensity_map
import numpy as np
from utils import find_neuron_coordinates, find_coordinate_matrix

# define device as a global variable to use the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PhysicalDiffractiveLayer:
    """ Physical implementation of a diffractive layer.

    Each physical neuron represents a square of a certain material with
    a given refractive index. The layer will represent a square of
    neurons_number x neuron_number.

    Attributes:
        neuron_length: The physical length of the neuron (length of one side
            of the square in meters).
        neurons_number: The number of neuron in one row (or collumn) of the
            grid.
        neuron_separation: The distance between two consecutive neurons
            (in meters)
        neuron_thickness: The thickness of each neuron.
        material_refractive_index: Torch tensor representing a matrix where
            each entry represents the refractive index of a neuron.
        first_neuron_x_coordinate: The x coordinate of the first neuron
            (the neuron on bottom right).
        first_neuron_y_coordinate: The y coordinate of the first neuron
            (the neuron on bottom right).
        neurons_z_coordinate: The z coordinate of all neurons.
        wavelength: Wavelength of light (in meters).
        phase_map: Phase value for each of the neurons.
        optical_modes: Optical mode of each neuron.
        neuron_coordinates_matrix: the coordinates for each neuron
    """
    def __init__(self, neuron_length: float, neurons_number: int,
                 neuron_separation: float,
                 neuron_thickness: float,
                 material_refractive_index: torch.Tensor,
                 first_neuron_x_coordinate: float = 0.0,
                 first_neuron_y_coordinate: float = 0.0,
                 neurons_z_coordinate: float = 0.0,
                 wavelength: float = 1.55E-6) -> None:
        self.neuron_length = neuron_length
        self.neurons_number = neurons_number
        self.neuron_thickness = neuron_thickness
        self.neuron_separation = neuron_separation
        self.first_neuron_x_coordinate = first_neuron_x_coordinate
        self.first_neuron_y_coordinate = first_neuron_y_coordinate
        self.first_neuron_coordinates = np.array(
            [self.first_neuron_x_coordinate,
             self.first_neuron_y_coordinate])
        self.neurons_z_coordinate = neurons_z_coordinate
        self.material_refractive_index = material_refractive_index
        self.wavelength = wavelength
        self.phase_map = self.find_phase_values()
        self.optical_modes = self.find_neuron_optical_modes()
        self.neuron_coordinates_matrix = find_neuron_coordinates(
            neuron_length=self.neuron_length,
            neurons_separation=self.neuron_separation,
            neurons_number=self.neurons_number,
            z_coordinate=self.neurons_z_coordinate,
            first_neuron_coordinates=self.first_neuron_coordinates)

    def find_phase_values(self) -> torch.Tensor:
        """Finds the phase values and optical modes of each neuron.

        Returns:
            Torch tensor representing the phase of each neuron.
        """
        phase_map = find_phase_change(
            refractive_index=self.material_refractive_index,
            thickness=self.neuron_thickness,
            wavelength=self.wavelength
        )
        return phase_map

    def find_neuron_optical_modes(self) -> torch.Tensor:
        """ Find the corresponding optical mode of each neuron.

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
        plt.imshow(intensity_map.detach().cpu().numpy())
        plt.colorbar()
        plt.show()



if __name__ == '__main__':
    c_sb2se3 = 4.04933
    a_sb2se3 = 3.285356
    debug_refractive_indices = torch.tensor([[a_sb2se3, c_sb2se3, a_sb2se3],
                                             [c_sb2se3, a_sb2se3, c_sb2se3],
                                             [a_sb2se3, c_sb2se3, a_sb2se3]],
                                            device=device)

    debug_pdl = PhysicalDiffractiveLayer(
        first_neuron_x_coordinate=-1E-6,
        first_neuron_y_coordinate=1E-6,
        neuron_length=0.8E-6,
        neurons_number=3,
        neuron_separation=0.2E-6,
        neuron_thickness=1E-6,
        neurons_z_coordinate=2E-6,
        material_refractive_index=debug_refractive_indices)

    debug_pdl.find_output(n_pixels=10, z_distance=3E-6)


