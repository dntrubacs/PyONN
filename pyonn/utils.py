import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import torch
from pyonn_data.processing import convert_optical_label
from pyonn_data.datasets import OpticalImageDataset, HybridImageDataset


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


def plot_complex_amplitude_map(
    complex_amplitude_map: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> None:
    """Plots the amplitude and phase map for a complex amplitude map.

    Args:
        complex_amplitude_map: Numpy array representing the complex valued
            elements (amplitude and phase). Must have shape
            (n_pixels, n_pixels) where n_pixels**2 is the number of elements
            in the map.
        x_coordinates: Numpy array representing the x coordinates of all
             pixels. Must be of shape (n_pixels, ).
        y_coordinates: Numpy array representing the y coordinates of all
             pixels. Must be of shape (n_pixels, ).
    """
    # generate the mesh grid necessary for plotting
    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)

    # get the amplitude and phase maps
    amplitude_map = np.abs(complex_amplitude_map)
    phase_map = np.angle(complex_amplitude_map)

    # create the figure
    figure, axis = plt.subplots(1, 2, figsize=(20, 8))

    # plot the intensity map
    axis[0].set_title("Amplitude Map")
    amplitude_map = axis[0].pcolormesh(
        x_mesh, y_mesh, amplitude_map, cmap="jet"
    )
    axis[0].set_xlabel("$x$ [mm]")
    axis[0].set_ylabel("$y$ [mm]")
    figure.colorbar(mappable=amplitude_map)

    # plot the phase map
    axis[1].set_title("Phase Map")
    phase_map = axis[1].pcolormesh(x_mesh, y_mesh, phase_map, cmap="inferno")
    axis[1].set_xlabel("$x$ [mm]")
    axis[1].set_ylabel("$y$ [mm]")
    figure.colorbar(mappable=phase_map)
    plt.show()


def test_model_on_image(
    model: torch.nn.Module,
    optical_image: torch.Tensor,
    optical_label: torch.Tensor,
) -> tuple:
    """Tests a model on a given optical image and optical label.

    Args:
        model: Deep diffractive neural network. Must be made as a torch model.
            (see the examples module).
        optical_image: Optical image used for the model prediction.
            (must be an optical image generated with the
            pyonn_data.processing.create_optical_images function).
        optical_label: Optical label used to find the real label
            (must be an optical image generated with the
            pyonn_data.processing.create_optical_labels function).

    Returns:
        Tuple containing the optical image, prediction and optical label
        (converted to numpy arrays) and the predicted and real labels.
    """
    # get the prediction of the model
    predicted_label = model(optical_image)

    # convert all tensors to numpy arrays
    np_optical_image = optical_image.detach().cpu().numpy()
    np_predicted_label = predicted_label.detach().cpu().numpy()
    np_optical_label = optical_label.detach().cpu().numpy()

    # get the 'real' labels (integer not optical images
    real_prediction = convert_optical_label(optical_label=np_predicted_label)[
        0
    ]
    real_label = convert_optical_label(optical_label=np_optical_label)[0]

    # return the given image, label and predictions
    return (
        np_optical_image,
        np_predicted_label,
        np_optical_label,
        real_prediction,
        real_label,
    )


def test_model_on_optical_dataset(
    model: torch.nn.Module, dataset: OpticalImageDataset
) -> float:
    """Test the model on a given dataset.

    Args:
        model: Deep diffractive neural network. Must be made as a torch model.
            (see the examples module).
        dataset: Optical Dataset (must be made with pyonn_data.datasets
        module).

    Returns:
        The accuracy of the model on the given dataset.
    """
    # number of samples in the dataset
    n_samples = dataset.__len__()

    # number of correct predictions
    n_correct = 0

    # go through each example and find the accuracy
    for n_sample in range(n_samples):
        optical_image, real_optical_label = dataset[n_sample]
        output_train = test_model_on_image(
            model=model,
            optical_image=optical_image,
            optical_label=real_optical_label,
        )
        predicted_label = output_train[3]
        real_label = output_train[4]

        if predicted_label == real_label:
            n_correct += 1
        if n_sample % 1000 == 0 and n_sample > 1:
            print(
                f"Sample number: {n_sample}. "
                f"Correct predictions until now: {n_correct}. "
                f"Accuracy until now: {n_correct / n_sample * 100} %"
            )

    # return the accuracy
    return n_correct / n_samples


def test_model_on_hybrid_dataset(
    model: torch.nn.Module, dataset: HybridImageDataset
) -> float:
    """tests the model on a given dataset.

    Args:
        model:  Optical encoder. Must be made as a torch model.
            (see the examples module).
        dataset: Hybrid Dataset (must be made with pyonn_data.datasets
        module).

    Returns:
        The accuracy of the model on the given dataset.
    """
    # number of samples in the dataset
    n_samples = dataset.__len__()

    # number of correct predictions
    n_correct = 0

    # go through each example and find the accuracy
    for n_sample in range(n_samples):
        optical_image, real_label = dataset[n_sample]
        output = torch.nn.functional.softmax(model(optical_image))

        # get the predicted label
        output = output.detach().cpu().numpy()
        predicted_label = np.argmax(output)

        if predicted_label == real_label:
            n_correct += 1
        if n_sample % 1000 == 0 and n_sample > 1:
            print(
                f"Sample number: {n_sample}. "
                f"Correct predictions until now: {n_correct}. "
                f"Accuracy until now: {n_correct / n_sample * 100} %"
            )

    # return the accuracy
    return n_correct / n_samples


def plot_optical_encoder(
    model: torch.nn.Module,
    optical_image: torch.tensor,
    label: float,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    image_title: str = None,
    save_path: str = None,
) -> None:
    """Plot the images of an optical encoder.

    The images plotted are the original optical image, the detector intensity
    map and the image after the max pool layer.

    Args:
        model: Optical Encoder moder, must be created with OpticalEncoder
            from pyonn.prebuilts.
        optical_image: Input image in the model (must be an optical image
            generated with the pyonn_data.processing.create_optical_images
            function).
        label: Label of the input image (must be a scalar label e.g. int from 0
            9).
        x_coordinates: Numpy array representing the x coordinates of all
             pixels. Must be of shape (n_pixels, ).
        y_coordinates: Numpy array representing the y coordinates of all
             pixels. Must be of shape (n_pixels, ).
        image_title: Title of the optical image plot. If none is given the
            title will simply state the prediction and label
        save_path: The path where to save the figure.
    """
    output = torch.nn.functional.softmax(model(optical_image))

    # get the predicted label
    output = output.detach().cpu().numpy()
    predicted_label = np.argmax(output)

    detector_intensity_map = model.detector_layer.intensity_map
    max_pool = model.max_pool
    max_pool_out = max_pool(detector_intensity_map.view(1, 120, 120))
    max_pool_out = max_pool_out.view(40, 40)

    # create a meshgrid for plotting
    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)

    # create a meshgrid for plotting the max pool layer (40x40)
    reduced_x_coordinates = np.linspace(
        start=min(x_coordinates), stop=max(x_coordinates), num=40
    )
    reduced_y_coordinates = np.linspace(
        start=min(y_coordinates), stop=max(y_coordinates), num=40
    )
    reduced_x_mesh, reduced_y_mesh = np.meshgrid(
        reduced_x_coordinates, reduced_y_coordinates
    )

    # create the figure
    figure, axis = plt.subplots(1, 3, figsize=(30, 8))

    # plot the input image
    if image_title is None:
        image_title = f"Prediction: {predicted_label}. Label: {label}"
    axis[0].set_title(image_title)
    input_image_map = axis[0].pcolormesh(
        x_mesh, y_mesh, optical_image.detach().cpu().numpy(), cmap="jet"
    )
    axis[0].set_xlabel("$x$ [m]")
    axis[0].set_ylabel("$y$ [m]")
    figure.colorbar(mappable=input_image_map)

    # plot the predicted image
    axis[1].set_title("Detector Output")
    predicted_image_map = axis[1].pcolormesh(
        x_mesh,
        y_mesh,
        detector_intensity_map.detach().cpu().numpy(),
        cmap="inferno",
    )

    axis[1].set_xlabel("$x$ [m]")
    axis[1].set_ylabel("$y$ [m]")
    figure.colorbar(mappable=predicted_image_map)

    # plot the image after max pool
    axis[2].set_title("Max Pooling")
    axis[2].set_xlabel("$x$ ")
    axis[2].set_ylabel("$y$ ")
    max_pool_image_map = axis[2].pcolormesh(
        reduced_x_mesh,
        reduced_y_mesh,
        max_pool_out.detach().cpu().numpy(),
        cmap="inferno",
    )

    axis[2].set_xlabel("$x$ [m]")
    axis[2].set_ylabel("$y$ [m]")
    figure.colorbar(mappable=max_pool_image_map)

    # if required, save the figure
    if save_path is not None:
        plt.savefig(save_path)

    # show the figure
    plt.show()


def plot_model_testing(
    input_image: np.ndarray,
    predicted_image: np.ndarray,
    label_image: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    input_image_title: str = "Input image",
    predicted_image_title: str = "Predicted",
    label_image_title: str = "Label",
    save_path: Optional[str] = None,
) -> None:
    """Plot the input image, prediction and label.

    Args:
        input_image: Input image in the model (must be an optical image
            generated with the pyonn_data.processing.create_optical_images
            function).
        predicted_image: Predicted image of the model.
        label_image: Label of the input image (must be an optical label
            generated with the pyonn_data.processing.create_optical_labels
            function).
        x_coordinates: Numpy array representing the x coordinates of all
             pixels. Must be of shape (n_pixels, ).
        y_coordinates: Numpy array representing the x coordinates of all
             pixels. Must be of shape (n_pixels, ).
        input_image_title: Title for the figure containing the input image.
        predicted_image_title: Title for the figure containing the predicted
            image.
        label_image_title: Title for the figure containing the label.
        save_path: The path where to save the figure.
    """
    # create a meshgrid for plotting
    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)

    # create the figure
    figure, axis = plt.subplots(1, 3, figsize=(30, 8))

    # plot the input image
    axis[0].set_title(input_image_title)
    input_image_map = axis[0].pcolormesh(
        x_mesh, y_mesh, input_image, cmap="jet"
    )
    axis[0].set_xlabel("$x$ [m]")
    axis[0].set_ylabel("$y$ [m]")
    figure.colorbar(mappable=input_image_map)

    # plot the predicted image
    axis[1].set_title(predicted_image_title)
    predicted_image_map = axis[1].pcolormesh(
        x_mesh, y_mesh, predicted_image, cmap="inferno"
    )
    axis[1].set_xlabel("$x$ [m]")
    axis[1].set_ylabel("$y$ [m]")
    figure.colorbar(mappable=predicted_image_map)

    # plot the label
    axis[2].set_title(label_image_title)
    label_image_map = axis[2].pcolormesh(
        x_mesh, y_mesh, label_image, cmap="inferno"
    )
    axis[2].set_xlabel("$x$ [m]")
    axis[2].set_ylabel("$y$ [m]")
    figure.colorbar(mappable=label_image_map)

    # if required, save the figure
    if save_path is not None:
        plt.savefig(save_path)

    # show the figure
    plt.show()


def plot_training_histogram(
    training_losses: list, validation_losses: Optional[list] = None
) -> None:
    """Plots the training histogram.

    Args:
        training_losses: Losses of the model during each epoch.
        validation_losses: Validations losses of the model during each epoch.
    """
    # the number of all epochs numbers
    epochs = np.arange(1, len(training_losses) + 1, step=1)

    # plot the histogram
    plt.figure(figsize=(12, 8))
    plt.title("Loss vs Epochs")
    plt.ylabel("Mean squared Error")
    plt.xlabel("Epochs")
    # plot all the epochs number only if the number is smaller than 10
    if len(epochs) < 10:
        plt.xticks(epochs)
    plt.plot(epochs, training_losses, color="green", label="training")
    if validation_losses is not None:
        plt.plot(epochs, validation_losses, color="blue", label="validation")
    plt.legend()
    plt.show()
