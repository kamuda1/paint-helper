from skimage.io import imread, imsave
from skimage import io
from skimage.color import rgb2hsv
from numpy import quantile
import numpy as np
import os


class MainClass:
    def __init__(self,
                 base_image_name: str,
                 image_folder: str = r"C:\GitRepos\paint-helper\images",
                 n_quantiles: int = 8
                 ):
        self.image_folder = image_folder
        self.base_image_name = base_image_name

        base_image_fname = os.path.join(self.image_folder, "00106-2904245478.png")
        self.base_image_rgb = imread(base_image_fname)
        self.base_image_hsv = rgb2hsv(self.base_image_rgb)

        self.n_quantiles = n_quantiles
        self.make_value_maps(n_quantiles=n_quantiles)

    def show_base_image(self):
        io.imshow(self.base_image_rgb)
        io.show()

    def make_value_maps(self, n_quantiles: int = 4):

        quantile_ranges = [(quantile(self.base_image_hsv[:, :, 2], 1/n_quantiles * i),
                            quantile(self.base_image_hsv[:, :, 2], 1/n_quantiles * (i + 1)))
                           for i in range(n_quantiles)]
        self.value_maps = {}
        self.value_map_masks = {}

        for i, quantile_range in enumerate(quantile_ranges):
            mask = np.multiply(self.base_image_hsv[:, :, 2] > quantile_range[0],
                               self.base_image_hsv[:, :, 2] < quantile_range[1])
            tmp_value_map = np.full_like(self.base_image_rgb, 50)
            tmp_value_map[mask] = self.base_image_rgb[mask]

            self.value_maps[i] = tmp_value_map
            self.value_map_masks[i] = mask

            imsave(os.path.join(self.image_folder, f'value_map_{i}.png'), tmp_value_map)
            imsave(os.path.join(self.image_folder, f'value_map_mask_{i}.png'), mask)


MainClass(base_image_name="00106-2904245478.png")
