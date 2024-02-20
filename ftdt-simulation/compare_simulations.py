import numpy as np
import os
from matplotlib import pyplot as plt

os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN")

# distance at which the intensity mao has been measured
distance = 50

# the folder where the data with results from scalar simulation
scalar_results = (f"results/scalar/single_slit/propagated map at "
                  f"distance {distance} um")

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
fdtd_results = (f"results/fdtd/single_slit/propagated map at distance "
                f"{distance} um")

# load the data for scalar simulation
fdtd_x_mesh = np.load(os.path.join(fdtd_results, "x_mesh"), allow_pickle=True)
fdtd_y_mesh = np.load(os.path.join(fdtd_results, "y_mesh"), allow_pickle=True)
fdtd_intensity_map = np.load(
    os.path.join(fdtd_results, "intensity_map"), allow_pickle=True
)

# normalize both intensity maps (better for comparison)
scalar_intensity_map = scalar_intensity_map/np.max(scalar_intensity_map)
fdtd_intensity_map = fdtd_intensity_map/np.max(fdtd_intensity_map)

# plot the scalar and fdtd simulations in the same figure
figure, axis = plt.subplots(1, 2, figsize=(20, 8))
figure.suptitle(f'Simulation for monitor placed at distance {distance}'
                f'$\mu$m.',
                fontsize=20)

# plot the scalar intensity map
axis[0].set_title('Scalar results')
scalar_map = axis[0].pcolormesh(scalar_x_mesh, scalar_y_mesh,
                                scalar_intensity_map, cmap="jet")
axis[0].set_xlabel("$x$ [mm]")
axis[0].set_ylabel("$y$ [mm]")
figure.colorbar(mappable=scalar_map)

# plot the fdtd intensity map
axis[1].set_title('FDTD results')
fdtd_map = axis[1].pcolormesh(fdtd_x_mesh, fdtd_y_mesh,
                              fdtd_intensity_map, cmap="jet")
axis[1].set_xlabel("$x$ [mm]")
axis[1].set_ylabel("$y$ [mm]")
figure.colorbar(mappable=fdtd_map)
plt.savefig(f'results/pictures/Single slit comparison for '
            f'distance {distance} um.png')
plt.show()
