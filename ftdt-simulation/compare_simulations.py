import numpy as np
import os
from matplotlib import pyplot as plt
import cv2 as cv

# the current directory
os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN")

# distance at which the intensity map has been measured
distance = 50

# the folder where the data with results from scalar simulation
scalar_results = (
    f"results/grid/scalar/propagated map at " f"distance {distance} um"
)

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
fdtd_results = (
    f"results/grid/fdtd/propagated map at distance " f"{distance} um"
)

# load the data for scalar simulation
fdtd_x_mesh = np.load(os.path.join(fdtd_results, "x_mesh"), allow_pickle=True)
fdtd_y_mesh = np.load(os.path.join(fdtd_results, "y_mesh"), allow_pickle=True)
fdtd_intensity_map = np.load(
    os.path.join(fdtd_results, "intensity_map"), allow_pickle=True
)

# normalize both intensity maps (better for comparison)
scalar_intensity_map = scalar_intensity_map / np.max(scalar_intensity_map)


# resize the fdtd intensity to map the pixel accuracy of the scalar one
# and make a copy of it
fdtd_intensity_map_resized = cv.resize(
    fdtd_intensity_map, dsize=scalar_intensity_map.shape
)

# normalize both fdtd intensity maps
fdtd_intensity_map = fdtd_intensity_map / np.max(fdtd_intensity_map)
fdtd_intensity_map_resized = fdtd_intensity_map_resized / np.max(
    fdtd_intensity_map_resized
)

# calculate the squared error between the 2 maps
error_map = (fdtd_intensity_map_resized - scalar_intensity_map) ** 2

# mean squared error
mean_squared_error = round(
    np.sum(error_map) / (error_map.shape[0] * error_map.shape[1]), 4
)


# plot the scalar and fdtd simulations in the same figure
figure, axis = plt.subplots(2, 2, figsize=(30, 25))
figure.suptitle(
    f"Simulation for monitor placed at distance "
    f"{distance} $\mu$m.",  # noqa: W605
    fontsize=20,
)

# plot the scalar intensity map
axis[0][0].set_title("Scalar result")
scalar_map = axis[0][0].pcolormesh(
    scalar_x_mesh, scalar_y_mesh, scalar_intensity_map, cmap="jet"
)
axis[0][0].set_xlabel("$x$ [mm]")
axis[0][0].set_ylabel("$y$ [mm]")
figure.colorbar(mappable=scalar_map)

# plot the fdtd intensity map
axis[0][1].set_title("FDTD result")
fdtd_map = axis[0][1].pcolormesh(
    fdtd_x_mesh, fdtd_y_mesh, fdtd_intensity_map, cmap="jet"
)
axis[0][1].set_xlabel("$x$ [mm]")
axis[0][1].set_ylabel("$y$ [mm]")
figure.colorbar(mappable=fdtd_map)

# plot the fdtd intensity map
axis[1][0].set_title("Resized FDTD result")
fdtd_map = axis[1][0].pcolormesh(
    scalar_x_mesh, scalar_y_mesh, fdtd_intensity_map_resized, cmap="jet"
)
axis[1][0].set_xlabel("$x$ [mm]")
axis[1][0].set_ylabel("$y$ [mm]")
figure.colorbar(mappable=fdtd_map)

# plot the fdtd intensity map
axis[1][1].set_title(f"Error Map. MSE = {mean_squared_error}")
error_map = axis[1][1].pcolormesh(scalar_x_mesh, scalar_y_mesh, error_map)
axis[1][1].set_xlabel("$x$ [mm]")
axis[1][1].set_ylabel("$y$ [mm]")
figure.colorbar(mappable=error_map)


plt.savefig(
    f"results/grid/pictures of comparison/distance " f"{distance} um.png"
)
plt.show()
