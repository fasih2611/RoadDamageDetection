import numpy as np
import cv2

def anisotropic_diffusion(image, iterations, kappa, gamma=0.1):

    image = image.astype('float32')

    for _ in range(iterations):

        north = np.roll(image, -1, axis=0)
        south = np.roll(image, 1, axis=0)
        west = np.roll(image, -1, axis=1)
        east = np.roll(image, 1, axis=1)


        north[-1, :], south[0, :], west[:, -1], east[:, 0] = 0, 0, 0, 0

        delta_n = image - north
        delta_s = image - south
        delta_w = image - west
        delta_e = image - east

        # Calculate conduction
        c_n = np.exp(-(delta_n / kappa)**2)
        c_s = np.exp(-(delta_s / kappa)**2)
        c_w = np.exp(-(delta_w / kappa)**2)
        c_e = np.exp(-(delta_e / kappa)**2)


        image += gamma * (c_n * delta_n + c_s * delta_s + c_w * delta_w + c_e * delta_e)

    return image

depth_map = cv2.imread('depthmap_Depth.png', cv2.IMREAD_UNCHANGED)
iterations = 30
kappa = 50

filled_depth_map = anisotropic_diffusion(depth_map, iterations, kappa)

# Save or display the filled depth map
cv2.imwrite('filled_depth_map.png', filled_depth_map)
