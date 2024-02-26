""" Created by Daniel-Iosif Trubacs on 26 February 2024. The purpose of this
module is to create torch Datasets an d Dataloaders  containing optical images
that can be used to train deep diffractive networks."""
import numpy as np
from torch.utils.data import Dataset
import torch


class OpticalImageDataset(Dataset):
    """Optical image dataset for training deep diffractive neural network.

    Attributes:
        optical_images: Numpy array representing optical images. Must be
            made with the processing.create_optical_images function (see
            its documentation for more information).
        optical_labels: Numpy array representing optical labels. Must be
            made with the processing.create_optical_labels function (see
            its documentation for more information).
    """

    def __init__(
        self, optical_images: np.ndarray, optical_labels: np.ndarray
    ) -> None:
        # get the number of samples
        self.n_samples = optical_images.shape[0]

        # convert the numpy array to torch tensors
        self.optical_images = torch.from_numpy(optical_images)
        self.optical_labels = torch.from_numpy(optical_labels)

    # get item method
    def __getitem__(self, index) -> tuple:
        return self.optical_images[index], self.optical_labels[index]

    # get len method
    def __len__(self) -> int:
        return self.n_samples


if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader

    # load the data (must be optical images and labels)
    os.chdir("C:/Users/dit1u20/PycharmProjects/PyONN/data")
    images = np.load("mnist_processed_data/train_images", allow_pickle=True)
    labels = np.load("mnist_processed_data/train_labels", allow_pickle=True)

    # generate an optical image dataset
    debug_dataset = OpticalImageDataset(
        optical_images=images, optical_labels=labels
    )

    # create a data loader
    debug_loader = DataLoader(
        dataset=debug_dataset, batch_size=12, shuffle=True, num_workers=0
    )
    dataiter = iter(debug_loader)
    data = next(dataiter)
    features, labels = data
    print(features, labels)
