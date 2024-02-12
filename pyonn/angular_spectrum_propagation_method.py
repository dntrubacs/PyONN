"""
Equations to help with the angular spectrum propagation method.
"""
import numpy as np
from utils import (create_square_grid_pattern, plot_square_grid_pattern,
                   create_pattern_mesh_grid)
from matplotlib import pyplot as plt
import scipy

def get_u(u_0, z, k, kx_mesh, ky_mesh):
    """ Finds the propagated phase map after a distance z"""
    sqrt_term = np.sqrt(np.abs(k**2-kx_mesh**2-ky_mesh**2))
    exp_term = np.exp(1j*z*sqrt_term)
    return scipy.fft.ifft2(u_0*exp_term)

if __name__ == '__main__':
    # wavelength
    wavelength = 1.55E-6
    k = 2 * np.pi / wavelength
    print(k)

    # create a square grid pattern centred on [0, 0] with pixel size 1 um
    # and pixel number 20 (400 pixels in total)
    square_grid_pattern = create_square_grid_pattern(
        center_coordinates=np.array([0, 0]),
        pixel_length=1E-6,
        pixel_number=40, pixel_separation=0.0,
        grid_z_coordinate=0)

    # generate the meshgrid necessary for the
    pattern, x_coordinates, y_coordinates, z_coordinate = square_grid_pattern
    x_mesh, y_mesh = create_pattern_mesh_grid(x_coordinates=x_coordinates,
                                              y_coordinates=y_coordinates)

    # generate a phase map that represents the digit 0
    phase_map = np.zeros(shape=(40, 40), dtype=np.float64)
    phase_map[4, 6:14] += 1.0
    phase_map[16, 6:14] += 1.0
    phase_map[5:16, 6] += 1.0
    phase_map[5:16, 13] += 1.0

    # show the phase map
    plt.figure(figsize=(12, 8))
    plt.title('Initial Phase Map')
    plt.pcolormesh(x_mesh, y_mesh, phase_map, cmap='jet')
    plt.xlabel('X-Position [m]')
    plt.ylabel('Y-Position [m]')
    plt.colorbar()
    plt.show()

    # compute the fourier transform of the initial phase map
    u_0 = scipy.fft.fft2(phase_map)

    # get the fourier space (kx and ky are the same)
    kx = scipy.fft.fftfreq(len(x_coordinates),
                           np.diff(x_coordinates)[0])*2*np.pi

    kx_mesh, ky_mesh = np.meshgrid(kx, kx)
    print(np.abs(k**2-kx_mesh**2-ky_mesh**2))

    # shift the fourier space to be centred on 0
    kx_shift = scipy.fft.fftshift(kx_mesh)
    ky_shift = scipy.fft.fftshift(ky_mesh)
    u_0_shift = scipy.fft.fftshift(u_0)

    # plot the fourier transform (u_0)
    plt.figure(figsize=(12, 8))
    plt.title('Fourier transform')
    plt.pcolormesh(kx_shift, ky_shift,
                   np.abs(u_0_shift), cmap='jet')
    plt.xlabel('$k_x$ [m$^{-1}$]')
    plt.ylabel('$k_y$ [m$^{-1}$]')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.colorbar()
    plt.show()



    distance = 50E-6

    # get the propagated phase map at 1 um
    u = get_u(u_0=u_0, z=distance, k=k, kx_mesh=kx_mesh, ky_mesh=ky_mesh)
    plt.figure(figsize=(12, 8))
    plt.title(f'Propagated phase map at: {distance} m')
    plt.pcolormesh(x_mesh, y_mesh, np.abs(u), cmap='jet')
    plt.xlabel('$x$ [mm]')
    plt.ylabel('$y$ [mm]')
    plt.colorbar()
    plt.show()









