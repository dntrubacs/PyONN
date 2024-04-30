import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def save_model_metric(
    train_losses: np.ndarray,
    validation_losses: np.ndarray,
    save_folder: str,
    train_accuracies: np.ndarray = None,
    validation_accuracies: np.ndarray = None,
) -> None:
    """

    Args:
        train_accuracies:
        validation_accuracies:
        train_losses:
        validation_losses:
        save_folder:

    Returns:

    """
    # save the training accuracies
    if train_accuracies is not None:
        with open(
            os.path.join(save_folder, "train_accuracies"), "wb"
        ) as handle:
            pickle.dump(train_accuracies, handle)

    # save the validation accuracies
    if validation_accuracies is not None:
        with open(
            os.path.join(save_folder, "validation_accuracies"), "wb"
        ) as handle:  # noqa
            pickle.dump(validation_accuracies, handle)

    # save the training losses
    with open(os.path.join(save_folder, "train_losses"), "wb") as handle:
        pickle.dump(train_losses, handle)

    # save the validation losses
    with open(os.path.join(save_folder, "validation_losses"), "wb") as handle:
        pickle.dump(validation_losses, handle)


def find_coordinate_matrix(
    n_size: int, n_length: float, z_coordinate: float
) -> np.ndarray:
    """Find the position coordinates of elements in a physical representation
    of an n_size x n_size matrix located at position z.

    Keep in mind that the position of the elements is always considered to be
    in the middle. For example, if n_length=1, the element in the matrix at
    position [1, 2] will have coordinates [1+1/2, 2+1/2].

    Args:
        n_size: Size of the matrix (there will be n_size x n_size elements in
            the matrix).
        n_length: Length of the physical matrix (squared).
        z_coordinate: Z coordinate where the physical matrix is placed.

    Returns:
        Numpy array containing the coordinates of elements (n_size, n_size, 3)
        where the last entry represents: (x, y, z).
    """
    # the matrix containing all coordinates
    matrix = np.zeros(shape=(n_size, n_size, 3))

    # length of one pixel (or physical element of the matrix)
    pixel_length = n_length / n_size

    # go through each element in the matrix and assign its value
    for i in range(n_size):
        for j in range(n_size):
            matrix[i][j] = np.array(
                [
                    pixel_length * (i + 0.5),
                    pixel_length * (j + 0.5),
                    z_coordinate,
                ]
            )

    return matrix


def create_square_grid_pattern(
    center_coordinates: np.ndarray,
    pixel_length: float,
    pixel_number: int,
    pixel_separation: float,
    grid_z_coordinate: float,
) -> tuple:
    """Finds the coordinates of all pixels from a square grid.

    Each feature of the grid (cna be square, circle, etc.) will be placed in
    a square 'pixel' of size pixel_length x pixel_length at a distance
    of pixel-separation from all other pixels around. Keep in mind that
    the total length of the physical grid will be:
    pixel_length*pixel_number + pixel_separation*(pixel_number-1)

    Args:
        center_coordinates: X and Y coordinates of the center. If pixel_number
            is odd the center will be a pixel and if pixel_number is even,
            the center will be placed between 4 pixels.
        pixel_length: The physical length of the pixel (length of one side
            of the square in meters).
        pixel_number: The number of pixels in one row (or collumn) of the
            grid.
        pixel_separation: The distance between two consecutive pixels
            (in meters)
        grid_z_coordinate: The z coordinate of all pixels.

    Returns:
        Tuple containing:
            -Numpy array containing the coordinates of elements
            (pixel_number, pixel_numbers, 3)
            where the last entry represents the coordinates (x, y, z).
            - Numpy array containing x-coordinates of all the pixels.
            - Numpy array containing y-coordinates of all the pixels.
            - Float representing the z coordinate of all the pixels.
    """
    # the matrix containing all coordinates
    pixel_matrix = np.zeros(shape=(pixel_number, pixel_number, 3))

    # distance between the centre of 2 pixels
    distance = pixel_length + pixel_separation

    # possible values for x and y
    # the center square will have center_coordinates for ood pixel number
    if pixel_number % 2 == 1:
        x_coordinates = (
            distance
            * np.arange(
                start=-int(pixel_number / 2),
                stop=int(pixel_number / 2) + 1,
                step=1,
            )
            + center_coordinates[0]
        )
        y_coordinates = (
            distance
            * np.arange(
                start=-int(pixel_number / 2),
                stop=int(pixel_number / 2) + 1,
                step=1,
            )
            + center_coordinates[1]
        )

    # center_coordinates will be between for ood pixel number
    else:
        x_coordinates = (
            distance
            * np.arange(
                start=-pixel_number / 2 + 0.5,
                stop=pixel_number / 2 + 0.5,
                step=1,
            )
            + center_coordinates[0]
        )
        y_coordinates = (
            distance
            * np.arange(
                start=-pixel_number / 2 + 0.5,
                stop=pixel_number / 2 + 0.5,
                step=1,
            )
            + center_coordinates[1]
        )

    # go through each pixel
    for i in range(len(x_coordinates)):
        for j in range(len(y_coordinates)):
            # set all z coordinates to grid_z_coordinates
            pixel_matrix[i][j][2] = grid_z_coordinate

            # set the x and y coordinates
            pixel_matrix[i][j][0] = x_coordinates[i]
            pixel_matrix[i][j][1] = y_coordinates[j]

    # return the square grid pattern found
    return pixel_matrix, x_coordinates, y_coordinates, grid_z_coordinate


def plot_square_grid_pattern(pattern: np.ndarray) -> None:
    """Plots the squared grid pattern in XY plane.

    Args:
        pattern: Numpy array representing the pixel matrix generated by the
            function create_square_grid_pattern
    """
    # create the figure
    plt.figure(figsize=(12, 8))

    # figure title and axis labels
    plt.title(f"Square grid pattern at z={pattern[0][0][2]}")
    plt.ylabel("y coordinate (m)")
    plt.xlabel("x coordinate (m)")

    # plot each pixel
    for i in range(pattern.shape[0]):
        for j in range(pattern.shape[1]):
            plt.plot(
                pattern[i][j][0],
                pattern[i][j][1],
                marker="s",
                markersize=20,
                color="blue",
            )

    plt.show()


def circ_function(x: np.ndarray) -> np.ndarray:
    """Returns 1 for all values smaller than 1 and 0 otherwise.

    Args:
        x: Numpy array of general shape

    Returns:
        Numpy array containing elements of 1 and 0.

    """
    circ_result = np.where(x > 1, 0, x)

    print(circ_result)

    # divide by itself to get only values
    circ_result = np.where(circ_result != 0, 1, circ_result)

    # return the result
    return circ_result


def optical_relu_plotting(x, alpha):
    if x < alpha:
        return 0
    else:
        return x


def optical_inverse_relu_plotting(x, beta):
    if x > beta:
        return beta
    else:
        return x


def plot_optical_activation_function(
    mode: str, alpha: float = None, beta: float = None
) -> None:
    """Plots optical activation functions such as ReLU or Inverse Relu.

    Args:
        mode:
        alpha:
    """
    # neuron amplitude range
    amplitude_range = np.arange(start=0, stop=1, step=0.0001)
    output = None

    # apply the optical activation function
    if mode == "relu":
        output = [
            optical_relu_plotting(x=x, alpha=alpha) for x in amplitude_range
        ]

    if mode == "inverse_relu":
        output = [
            optical_inverse_relu_plotting(x=x, beta=beta)
            for x in amplitude_range
        ]

    # plot the output map
    plt.figure(figsize=(12, 8))
    if mode == "relu":
        plt.title(f"Optical ReLU for $alpha$={alpha}")
        plt.plot(
            amplitude_range,
            output,
            label="Equivalent Neuron Amplitude",
            color="blue",
        )
    if mode == "inverse_relu":
        plt.title(f"Optical Inverse ReLU for $beta$={beta}")
        plt.plot(
            amplitude_range,
            output,
            label="Equivalent Neuron Amplitude",
            color="red",
        )
    plt.xlabel("Amplitude")
    plt.ylabel("Transmitted Amplitude")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    plot_optical_activation_function(mode="inverse_relu", beta=0.8)
