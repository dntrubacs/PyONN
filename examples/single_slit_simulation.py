""" Created by Daniel-Iosif Trubacs on 22 February 2024. The purpose
of this module is to show an example on how to use Diffractive Layers to
simulate a single slit diffraction pattern.

The material simulated in this case represents a surface of amorphous
Sb2Se3 with a written crystalline Sb2Se3 slit.
"""

from pyonn.utils import create_square_grid_pattern
from pyonn.diffraction_equations import find_phase_change
from pyonn.diffractive_layers import InputDiffractiveLayer, DiffractiveLayer
from pyonn.angular_spectrum_propagation_method import plot_real_maps
import numpy as np
import torch

# wavelength of light
wavelength = 1.55e-6

# create a square grid pattern centred on [0, 0] with pixel size 0.8 um
# and pixel number 100 (10000 pixels in total)
square_grid_pattern = create_square_grid_pattern(
    center_coordinates=np.array([0, 0]),
    pixel_length=0.8e-6,
    pixel_number=120,
    pixel_separation=0.0,
    grid_z_coordinate=0,
)

# retain only the x coordinates of the pattern (necessary for the
# meshgrid)
x_coordinates = square_grid_pattern[1]
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
phase_map = np.zeros(shape=(120, 120), dtype=np.float64) + 1.0

# create a complex amplitude map for the phase change material
# e^(j*phase)
# set the background of the map to amorphous phase change
phase_map = phase_map * np.exp(1j * amorphous_phase_change)

# set the slit to crystalline phase change
phase_map[25:95, 60] = np.exp(1j * crystalline_phase_change)

# move from numpy to torch tensors
initial_phase_map = torch.from_numpy(phase_map)

# find the propagated complex amplitude map for different distance
for dist in [0.1, 1, 2, 5, 10, 20, 50]:
    resized_dist = dist * 1e-6

    # create the InputDiffractiveLayer
    single_slit_input = InputDiffractiveLayer(
            n_size=120, x_coordinates=x_coordinates,
            y_coordinates=x_coordinates,
            z_coordinate=0.0, z_next=resized_dist,
            complex_amplitude_map=initial_phase_map,
            wavelength=1.55E-6
    )

    # create a DiffractiveLayer with random weights
    diffractive_layer = DiffractiveLayer(
        n_size=120, x_coordinates=x_coordinates,
        y_coordinates=x_coordinates,
        z_coordinate=resized_dist, z_next=resized_dist,
        wavelength=1.55E-6
    )

    # do the forward pass (propagate the light)
    propagated_map = single_slit_input.forward()

    # plot the propagated real maps
    plot_real_maps(
        complex_amplitude_map=propagated_map,
        x_coordinates=x_coordinates,
        intensity_map_title=f"Propagated intensity map at: "
                            f"{round(dist, 3)} $\mu$ m",  # noqa W605
        phase_map_title=f"Propagated phase map at: "
                        f"{round(dist, 3)} $\mu$ m",  # noqa W605
    )

    diffractive_layer.plot_weights_map()
    output = diffractive_layer.forward(x=propagated_map)

    # plot the propagated real maps
    plot_real_maps(
            complex_amplitude_map=output,
            x_coordinates=x_coordinates,
            intensity_map_title=f"Output at: "
            f"{round(dist, 3)} $\mu$ m",  # noqa W605
            phase_map_title=f"Output at: "
            f"{round(dist, 3)} $\mu$ m",  # noqa W605
    )
