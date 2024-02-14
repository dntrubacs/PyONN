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

    # create a square grid pattern centred on [0, 0] with pixel size 1 um
    # and pixel number 20 (400 pixels in total)
    square_grid_pattern = create_square_grid_pattern(
        center_coordinates=np.array([0, 0]),
        pixel_length=0.8E-6,
        pixel_number=100, pixel_separation=0.0,
        grid_z_coordinate=0)

    # generate the meshgrid necessary for the
    pattern, x_coordinates, y_coordinates, z_coordinate = square_grid_pattern
    plot_square_grid_pattern(pattern)



    x_mesh, y_mesh = create_pattern_mesh_grid(x_coordinates=x_coordinates,
                                              y_coordinates=y_coordinates)

    # generate a phase map that represents the digit 0
    phase_map = np.zeros(shape=(100, 100), dtype=np.float64) + 1.0
    phase_map = phase_map*np.exp(1j*(-0.1607))
    phase_map[30:70, 50] = np.exp(1j*2.9361)
  #  phase_map[80, 30:70] = np.exp(1j*2.9361)
  #  phase_map[21:80, 30] = np.exp(1j*2.9361)
  #  phase_map[20:81, 70] = np.exp(1j*2.9361)

    f = open('pattern.txt', 'w')

    for i in range(100):
        for j in range(100):
            if phase_map[i][j] == np.exp(1j*2.9361):
                material = 1
            else:
                material = 0
            x_value = pattern[i][j][0]
            y_value = pattern[i][j][1]

            print('x_value: ', x_value,
                  'y_value: ', y_value,
                  'material: ', material)
            f.write(f'{x_value} {y_value} {material} \n')
    f.close()

    # show the phase map
    plt.figure(figsize=(12, 8))
    plt.title('Initial Phase Map')
    plt.pcolormesh(x_mesh, y_mesh, np.angle(phase_map), cmap='jet')
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

    i = 0
    for dist in [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]:
        distance = dist*1E-6

        # get the propagated phase map at 1 um
        u = get_u(u_0=u_0, z=distance, k=k, kx_mesh=kx_mesh, ky_mesh=ky_mesh)

        # plot the intensity and phase map
        figure, axis = plt.subplots(1, 2,
                                    figsize=(20, 8))

        axis[0].set_title(f'Propagated intensity map at: '
                          f'{round(dist, 3)} $\mu$m')
        a = axis[0].pcolormesh(x_mesh, y_mesh, np.square(np.abs(u)),
                           cmap='jet')
        axis[0].set_xlabel('$x$ [mm]')
        axis[0].set_ylabel('$y$ [mm]')
        figure.colorbar(mappable=a)

        axis[1].set_title(f'Propagated phase map at: {round(dist, 3)} $\mu$m')
        b = axis[1].pcolormesh(x_mesh, y_mesh, np.angle(u),
                               cmap='inferno')
        axis[1].set_xlabel('$x$ [mm]')
        axis[1].set_ylabel('$y$ [mm]')
        figure.colorbar(mappable=b)
        plt.savefig(f'diffracted patterns/{round(dist, 3)}.png')
        plt.show()











