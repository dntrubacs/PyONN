"""
Created by Daniel-Iosif Trubacs on 20 February 2024. the purpose of this module
is to process the simulated patterns (neural networks) with Lumerical FDTD
and compare with the ones obtained through scalar methods.
"""

import numpy as np


def process_ftd_data(fdtd_path: str, max_x_value: float, max_y_value) -> tuple:
    """Processes the FDTD simulation raw data and returns the electric field
    values.

    Args:
        fdtd_path: Path to the txt file containing the raw data. It must be
            generated by the Lumerical FDTD model.
        max_x_value: Maximum value for the x coordinates. Only data
            in the rage (-max_x_value, max_x_value) will be kept.
        max_y_value: Maximum value for the x coordinates. Only data
            in the rage (-max_y_value, max_y_value) will be kept.

    Returns:
        Tuple containing the x and y numpy mesh grids plus a numpy array
        containing the value of the electric field (x_mesh, y_mesh, e).

    """
    # open the file with all the data
    fdtd_data = open(fdtd_path, "r")

    # split the file into lines
    lines = fdtd_data.read().splitlines()

    # use these variables as a checkpoint of saving data when reading lines
    # in the file
    append_x = False
    append_y = False
    append_intensity = False

    # lists for xy coordinates and intensity measurements
    x = []
    y = []
    intensity = []

    # go through each line in the data file (neglect the first one as it is
    # only a test)
    for line in lines[1:]:
        # ignore empty lines
        if len(line) == 0:
            pass

        else:
            # append the x values
            if append_x is True:
                # this signals the end of x values
                if line[0] == "y":
                    append_x = False
                else:
                    # else append the values
                    x.append(float(line))
            # this signal the start of x values
            if line[0] == "x":
                append_x = True

            # append the y values
            if append_y is True:
                # this signals the end of y values
                if line[0] == "m":
                    append_y = False
                else:
                    # else append the values
                    y.append(float(line))
            # this signal the start of y values
            if line[0] == "y":
                append_y = True

            # append all intensity values
            if append_intensity:
                for intensity_value in line.split(" "):
                    # try to see whether the value is a float
                    try:
                        intensity.append(float(intensity_value))
                    except ValueError:
                        pass

            # this signals the start of intensity values
            if line[0] == "m":
                append_intensity = True

    # convert x, y and intensity list to numpy arrays
    x = np.array(x)
    y = np.array(y)
    intensity = np.array(intensity)

    # reshape intensity to have the same shape as (len(x), len(y))
    intensity = np.reshape(intensity, newshape=(x.shape[0], y.shape[0]))

    # initialize index min and max for both x and y as None
    # this is used to keep only the part we are interested in
    index_min_x = None
    index_max_x = None
    index_min_y = None
    index_max_y = None

    # remember the indices that will create the necessary
    for i in range(len(x)):
        if x[i] > max_x_value and index_max_x is None:
            index_max_x = i
        if x[i] > -1 * max_x_value and index_min_x is None:
            index_min_x = i

    for i in range(len(y)):
        if y[i] > max_y_value and index_max_y is None:
            index_max_y = i
        if y[i] > -1 * max_y_value and index_min_y is None:
            index_min_y = i

    # keep only the neccessary part of x and y
    x = x[index_min_x:index_max_x]
    y = y[index_min_y:index_max_y]

    # create a mesh grid  of x and y
    x_mesh, y_mesh = np.meshgrid(x, y)

    # keep only the necessary part of intensity array
    intensity = intensity[index_min_x:index_max_x, index_min_y:index_max_y]
    intensity = intensity.T

    # return the x, y and intensity arrays
    return x_mesh, y_mesh, intensity


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    fdtd_file = "single slit 10 um distance.txt"
    debug_x, debug_y, debug_intensity = process_ftd_data(
        fdtd_path=fdtd_file, max_x_value=3.96e-5, max_y_value=3.96e-5
    )

    plt.pcolormesh(debug_x, debug_y, debug_intensity, cmap="jet")
    plt.colorbar()
    plt.show()
