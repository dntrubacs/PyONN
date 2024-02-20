import numpy as np
import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN")

# the folder where the data with results from scalar simulation
scalar_results = "results/scalar/single_slit/propagated map at distance 1 um"

# load the data for scalar simulation
scalar_x_mesh = np.load(
    os.path.join(scalar_results, "x_mesh"), allow_pickle=True
)
scalar_y_mesh = np.load(
    os.path.join(scalar_results, "y_mesh"), allow_pickle=True
)
scalar_complex_amplitude_map = np.load(
    os.path.join(scalar_results, "complex_amplitude_map"), allow_pickle=True
)

# generate the intensity map from the complex amplitude map
scalar_intensity_map = np.square(np.abs(scalar_complex_amplitude_map))


# the folder where the data with results from fdtd simulation
fdtd_results = "results/fdtd/single_slit/propagated map at distance 10 um"

# load the data for scalar simulation
fdtd_x_mesh = np.load(os.path.join(fdtd_results, "x_mesh"), allow_pickle=True)
fdtd_y_mesh = np.load(os.path.join(fdtd_results, "y_mesh"), allow_pickle=True)
fdtd_intensity_map = np.load(
    os.path.join(fdtd_results, "intensity_map"), allow_pickle=True
)


plt.pcolormesh(fdtd_x_mesh, fdtd_y_mesh, fdtd_intensity_map, cmap="jet")
plt.colorbar()
plt.show()
