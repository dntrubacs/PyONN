import numpy as np
import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN")

# the folder where the data with results from scalar simulation
scalar_results = "results/scalar/single_slit/propagated map at distance 1 um"

# load the data for scalar simulation
x_mesh = np.load(os.path.join(scalar_results, "x_mesh"), allow_pickle=True)
y_mesh = np.load(os.path.join(scalar_results, "y_mesh"), allow_pickle=True)
complex_amplitude_map = np.load(
    os.path.join(scalar_results, "complex_amplitude_map"), allow_pickle=True
)

# generate the intensity map from the complex amplitude map
intensity_map = np.square(np.abs(complex_amplitude_map))


plt.pcolormesh(x_mesh, y_mesh, intensity_map, cmap="jet")
plt.colorbar()
plt.show()
