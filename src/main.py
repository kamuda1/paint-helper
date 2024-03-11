from skimage.io import imread, imsave
from skimage import io
from skimage.color import rgb2hsv, lab2rgb, rgb2lab, deltaE_ciede94
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
        self.background_rgb = None

    def set_background_rgb(self, value: np.ndarray):
        self.background_rgb = value

    def get_background_rgb(self):
        if self.background_rgb is None:
            raise TypeError("Background Not Initialized")
        return self.background_rgb

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

    def compare_images(self, live_image_rgb: np.ndarray, base_image_rgb: np.ndarray) -> np.ndarray:
        """
        Compares two images, returns an image showing where they don't match
        """
        live_image_mask = live_image_rgb[:, :, :3]

        live_image_lab = rgb2lab(live_image_rgb[:, :, :3])
        base_image_lab = rgb2lab(base_image_rgb[:, :, :3])

        image_difference_lab = base_image_lab - live_image_lab
        return image_difference_lab


if __name__ == "__main__":
    main_class = MainClass(base_image_name="00106-2904245478.png")
    main_class.set_background_rgb(imread(os.path.join(main_class.image_folder, "image_0_live_background.png")))
    pass

base_image_fname = os.path.join(main_class.image_folder, "value_map_0.png")
base_image_rgb = imread(base_image_fname)

live_image_fname = os.path.join(main_class.image_folder, "image_0_live_value_map_0.png")
live_image_rgb = imread(live_image_fname)

test_image_lab = main_class.compare_images(live_image_rgb=live_image_rgb, base_image_rgb=base_image_rgb)
test_image_lab_0 = test_image_lab[:, :, 0]
test_image_lab_1 = test_image_lab[:, :, 1]
test_image_lab_2 = test_image_lab[:, :, 2]

io.imshow(test_image_lab_2)
io.show()
