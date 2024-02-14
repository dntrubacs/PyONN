""" Created by Daniel-Iosif Trubacs on 30 January 2024. The purpose
of this module is to build physical DNN by transforming the phase and
transmission values of neurons to different refractive indices and depths. """

import numpy as np
from matplotlib import pyplot as plt

from diffraction_equations import *
from utils import create_square_grid_pattern, plot_square_grid_pattern

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
        neuron_coordinates: Torch tensor that contains all coordinates of
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
        print(phase_map)
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
        # create the figure
        plt.figure(figsize=(12, 8))

        # title and axis labels
        plt.title(f'Phase Map (measured in radians) placed at'
                  f' z={self.neurons_z_coordinate}')
        plt.ylabel('y coordinate (m)')
        plt.xlabel('x coordinate (m)')

        # copy neuron_coordinates in a numpy array
        np_neuron_coordinates = self.neuron_coordinates.detach().cpu().numpy()

        # x_ticks and y_ticks represent the possible values for x and
        # coordinates
        x_ticks = []
        y_ticks = []
        for i in range(np_neuron_coordinates.shape[0]):
            x_ticks.append(np_neuron_coordinates[i][0][0])
        for i in range(np_neuron_coordinates.shape[0]):
            y_ticks.append(np_neuron_coordinates[0][i][1])

        # get the limits for phase map
        limits = [min(x_ticks), max(x_ticks), min(y_ticks), max(y_ticks)]

        # plot the phase map
        plt.imshow(X=self.phase_map.detach().cpu().numpy(),
                   extent=limits)
        plt.xticks(x_ticks)
        plt.yticks(y_ticks)
        plt.colorbar()
        plt.show()

    def find_output(self, n_pixels: int, z_coordinate: float,
                    detector_size: float, center_coordinates: np.ndarray,
                    plot: bool = True) -> torch.Tensor:
        """ Find the output optical modes of the neurons at a detector.

        The detector will be placed at z_coordinate and have a center of
        center_coordinates. The accuracy (the numbers of pixels that record
        the intensity) will be n_pixels x n_pixels.

        Args:
            n_pixels: number of pixels.
            z_coordinate: The z coordinate at which the detector will be
                placed.
            detector_size: The size of the detector of the detector (one
                length of it).
            center_coordinates: X and Y coordinates of the center.
            plot: Boolean representing whether to plot the result.
        """
        # the required size of each pixel
        pixel_size = detector_size/n_pixels

        # numpy array of detector positions
        np_detector_positions = create_square_grid_pattern(
            center_coordinates=center_coordinates,
            pixel_length=pixel_size,
            pixel_number=n_pixels,
            pixel_separation=0,
            grid_z_coordinate=z_coordinate
        )

        # plot the detector positions if necessary
        if plot:
            plot_square_grid_pattern(pattern=np_detector_positions)

        # move to torch tensor for detector positions
        detector_positions = torch.from_numpy(np_detector_positions).to(device)

        # find the output of optical modes
        # output optical modes
        output_optical_modes = find_optical_modes(
            source_positions=self.neuron_coordinates,
            detector_positions=detector_positions,
            source_optical_modes=self.optical_modes,
            wavelength=self.wavelength
        )

        # find the intensity map corresponding to the intensity of light
        intensity_map = find_intensity_map(output_optical_modes)

        # if necessary plot the intensity map
        if plot:
            # create the figure
            plt.figure(figsize=(12, 8))

            # title and axis labels
            plt.title(f'Intensity map placed at'
                      f' z={z_coordinate}')
            plt.ylabel('y coordinate (m)')
            plt.xlabel('x coordinate (m)')

            # get the correct x_ticks and y_ticks
            x_ticks = []
            y_ticks = []
            for i in range(np_detector_positions.shape[0]):
                x_ticks.append(np_detector_positions[i][0][0])
            for i in range(np_detector_positions.shape[0]):
                y_ticks.append(np_detector_positions[0][i][1])

            # get the limits for phase map
            limits = [min(x_ticks), max(x_ticks), min(y_ticks), max(y_ticks)]
            print(limits)
            # keep only 15 values for x_ticks and y_ticks (don't overcrowd
            # the plot)
            if len(x_ticks) > 10:
                x_ticks = np.linspace(start=min(x_ticks), stop=max(x_ticks),
                                      num=10)
                y_ticks = np.linspace(start=min(y_ticks), stop=max(y_ticks),
                                      num=10)

            # show the map
            plt.imshow(intensity_map.detach().cpu().numpy(), cmap='jet',
                       extent=limits)
            plt.xticks(x_ticks)
            plt.yticks(y_ticks)
            plt.colorbar(cmap='jet')
            plt.show()

        # return the intensity map
        return intensity_map


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
        pixel_separation=0.0,
        pixel_number=3,
        grid_z_coordinate=0
    )

    # create a torch tensor and move it to the device
    debug_neuron_coordinates = torch.from_numpy(
        debug_neuron_coordinates).to(device)

    debug_pdl = PhysicalDiffractiveLayer(
        neuron_coordinates=debug_neuron_coordinates,
        neuron_thickness=1E-6,
        wavelength=1.55E-6,
        material_refractive_index=debug_refractive_indices)

    debug_pdl.plot_phase_map()

    debug_pdl.find_output(n_pixels=150,
                          z_coordinate=50E-6,
                          detector_size=2.8E-6,
                          center_coordinates=np.array([0, 0]),
                          plot=True
                          )
