import numpy as np


def find_coordinate_matrix(n_size: int, n_length: float,
                           z_coordinate: float) -> np.ndarray:
    """ Find the position coordinates of elements in a physical representation
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
    pixel_length = n_length/n_size

    # go through each element in the matrix and assign its value
    for i in range(n_size):
        for j in range(n_size):
            matrix[i][j] = np.array([pixel_length*(i+0.5),
                                     pixel_length*(j+0.5),
                                     z_coordinate])

    return matrix


def find_neuron_coordinates(first_neuron_coordinates: np.ndarray,
                            neuron_length: float, neurons_number: int,
                            neurons_separation: float,
                            z_coordinate: float) -> np.ndarray:
    """ Finds the coordinates of all neurons from a given grid.

    Args:
        first_neuron_coordinates: X and Y coordinates of the first neuron
            (top left). The coordinate represent the center of the neuron.
        neuron_length: The physical length of the neuron (length of one side
            of the square in meters).
        neurons_number: The number of neuron in one row (or collumn) of the
            grid.
        neurons_separation: The distance between two consecutive neurons
            (in meters)
        z_coordinate: The z coordinate of all neurons

    Returns:
        Numpy array containing the coordinates of elements
        (neurons_number, neuron_numbers, 3)
        where the last entry represents the coordinates (x, y, z).
    """
    # the matrix containing all coordinates
    coordinate_matrix = np.zeros(shape=(neurons_number, neurons_number, 3))

    # go through each row and columns
    for i in range(neurons_number):
        for j in range(neurons_number):
            # total_separation between neurons
            distance = neurons_separation+neuron_length
            # the x position of the neuron
            coordinate_matrix[i][j][0] = (
                    first_neuron_coordinates[0] + i * distance)
            # the y position of the neuron
            coordinate_matrix[i][j][1] = (
                    first_neuron_coordinates[1] - j * distance)
            # the z position of the neuron
            coordinate_matrix[i][j][2] = z_coordinate

    return coordinate_matrix


if __name__ == '__main__':
    # used only for testing and debugging
    debug_matrix = find_neuron_coordinates(
        first_neuron_coordinates=np.array([-1E-6, 1E-6]),
        neuron_length=0.8E-6,
        neurons_separation=0.2E-6,
        neurons_number=3,
        z_coordinate=1E-6)
