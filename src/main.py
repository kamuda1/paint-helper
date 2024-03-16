import time
from typing import List

import gradio.components.image
from PIL.Image import Image
from skimage.io import imread, imsave
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv, lab2rgb, rgb2lab, deltaE_ciede94, hsv2rgb
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid
from numpy import quantile
import numpy as np
import os
import gradio as gr

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

    def modify_hsv_from_rgb(self, image_rgb: np.ndarray, s_aug: float = 1.0, v_aug: float = 1.0) -> np.ndarray:
        image_hsv = rgb2hsv(image_rgb)
        image_hsv[:, :, 1] *= s_aug
        image_hsv[:, :, 2] *= v_aug

        image_mod_rgb = hsv2rgb(image_hsv)
        image_mod_rgb = (image_mod_rgb * 255).astype(np.uint8)

        return image_mod_rgb

    def show_progress(self, live_image_rgb: np.ndarray, curr_layer_idx: int = 0, image_choice: str = "Base Image", s_aug: float = 0.75, v_aug: float = 0.5) -> List[np.ndarray]:
        """
        Shows where the current live image matches the current layer. Match is checked both spatially, does the pixel
        fall on the layer's mask, and if the color matches the layer.
        """
        if self.get_background_rgb() is None:
            return None

        live_image_rgb = resize(live_image_rgb, self.base_image_rgb.shape)
        live_image_rgb = (live_image_rgb*255).astype(np.uint8)

        background_mask = self.background_mask(live_image_rgb)
        valuemap_minus_painted_mask = self.value_map_masks[curr_layer_idx] * background_mask[:, :, 0]
        valuemap_minus_painted_mask_3d = np.repeat(np.expand_dims(valuemap_minus_painted_mask, 2), 3, 2)

        # ================
        live_image_rgb_flat = live_image_rgb.reshape(
            [live_image_rgb.shape[0] * live_image_rgb.shape[1], live_image_rgb.shape[2]])[:, :3]
        pred_live_image_clusters_flat = self.cluster_map_discriminator.predict(live_image_rgb_flat)
        pred_live_image_clusters = pred_live_image_clusters_flat.reshape(live_image_rgb.shape[0],
                                                                         live_image_rgb.shape[1])


        correct_paint_mask = (pred_live_image_clusters == curr_layer_idx) * self.value_map_masks[curr_layer_idx]
        correct_paint_mask_3d = np.repeat(np.expand_dims(correct_paint_mask, 2), 3, 2)

        wrong_paint_mask = (pred_live_image_clusters != curr_layer_idx) * self.value_map_masks[curr_layer_idx] * (
                    pred_live_image_clusters != -1)
        wrong_paint_mask_3d = np.repeat(np.expand_dims(wrong_paint_mask, 2), 3, 2)

        outside_curr_layer_paint_mask = (pred_live_image_clusters == curr_layer_idx) * ~self.value_map_masks[
            curr_layer_idx]
        outside_curr_layer_paint_mask_3d = np.repeat(np.expand_dims(outside_curr_layer_paint_mask, 2), 3, 2)

        test_image_rgb = np.where(correct_paint_mask_3d,
                                  live_image_rgb[:, :, :3],
                                  self.value_maps[curr_layer_idx],
                                  )
        value_map_mod = self.modify_hsv_from_rgb(self.value_maps[curr_layer_idx], 0.5, 0.5)
        live_image_mod_rgb = self.modify_hsv_from_rgb(live_image_rgb[:, :, :3], 0.5, 0.5)

        wrong_paint_image_rgb = np.where(wrong_paint_mask_3d,
                                         live_image_rgb[:, :, :3],
                                         live_image_mod_rgb
                                         )
        correct_paint_image_rgb = np.where(correct_paint_mask_3d,
                                           self.background_rgb[:, :, :3],
                                           self.value_maps[curr_layer_idx],
                                           )
        # imsave(os.path.join(self.image_folder, f'valuemap_minus_painted_mask_3d.png'), valuemap_minus_painted_mask_3d)
        # imsave(os.path.join(self.image_folder, f'correct_paint_mask_3d.png'), correct_paint_mask_3d)
        # imsave(os.path.join(self.image_folder, f'wrong_paint_mask_3d.png'), wrong_paint_mask_3d)
        # imsave(os.path.join(self.image_folder, f'outside_curr_layer_paint_mask_3d.png'),
        #        outside_curr_layer_paint_mask_3d)
        #
        # imsave(os.path.join(self.image_folder, f'test_image_rgb.png'), test_image_rgb)
        # imsave(os.path.join(self.image_folder, f'live_image_rgb.png'), live_image_rgb)
        # imsave(os.path.join(self.image_folder, f'wrong_paint_image_rgb.png'), wrong_paint_image_rgb)
        # imsave(os.path.join(self.image_folder, f'correct_paint_image_rgb.png'), correct_paint_image_rgb)
        # ================

        if image_choice == 'Good Paint':
            return correct_paint_image_rgb
        if image_choice == 'Bad Paint':
            return wrong_paint_image_rgb
        if image_choice == 'Current Image':
            return live_image_rgb
        else:
            return self.base_image_rgb

    def set_background_rgb(self, image: np.ndarray):
        """
        Sets initial painting background and trains a gaussian mixture model to identify background in future images.
        Also defines a minimum log likelihood for background, anything that scores less than that is likely not
        background.
        """
        self.background_rgb = image
        background_pixel_rgb = image.reshape([image.shape[0] * image.shape[1], image.shape[2]])[:, :3]
        self.background_gm_trained = GaussianMixture().fit(background_pixel_rgb)
        self.min_loglikelihood_background = min(self.background_gm_trained.score_samples(background_pixel_rgb))
        print('Background Saved')

    def create_value_map_discriminator(self):
        """
        Trains a model to predict which value map a pixel belongs to.
        """
        background_pixels = self.background_rgb[:, :, :3].reshape(
            self.background_rgb.shape[0] * self.background_rgb.shape[1], 3)
        x = list(background_pixels)
        y = [-1] * len(background_pixels)

        for index in range(len(self.value_maps)):
            tmp_x = self.value_maps[index][self.value_map_masks[index]]
            tmp_y = [index] * len(tmp_x)

            x.extend(tmp_x)
            y.extend(tmp_y)
        self.cluster_map_discriminator = NearestCentroid().fit(x, y)

    def get_background_rgb(self):
        # if self.background_rgb is None:
        #     raise TypeError("Background Not Initialized")
        return self.background_rgb

    def background_mask(self, image: np.ndarray, likeihood_factor: float = 0.1):
        live_pixel_rgb = image.reshape([image.shape[0] * image.shape[1], image.shape[2]])[:, :3]
        loglikelihood_pixels = self.background_gm_trained.score_samples(live_pixel_rgb)
        background_mask = loglikelihood_pixels.reshape(
            image.shape[0], image.shape[1], 1) > likeihood_factor * self.min_loglikelihood_background
        return background_mask

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

    def update_value_maps_with_background(self):
        for key in self.value_maps:
            value_map_3d = np.repeat(np.expand_dims(self.value_map_masks[key], 2), 3, 2)
            tmp_value_map = np.where(value_map_3d,
                                     self.value_maps[key],
                                     self.background_rgb[:, :, :3])
            self.value_maps[key] = tmp_value_map
            imsave(os.path.join(self.image_folder, f'value_map_{key}.png'), tmp_value_map)

    def compare_images(self, live_image_rgb: np.ndarray, base_image_rgb: np.ndarray) -> np.ndarray:
        """
        Compares two images, returns an image showing where they don't match
        """
        live_image_lab = rgb2lab(live_image_rgb[:, :, :3])
        base_image_lab = rgb2lab(base_image_rgb[:, :, :3])

        image_difference_lab = base_image_lab - live_image_lab
        return image_difference_lab


if __name__ == "__main__":
    main_class = MainClass(base_image_name="00106-2904245478.png")
    main_class.set_background_rgb(imread(os.path.join(main_class.image_folder, "image_0_live_background.png")))
    main_class.update_value_maps_with_background()
    main_class.create_value_map_discriminator()

    def show_progress(live_image, curr_layer_idx, image_choice):
        return main_class.show_progress(live_image, curr_layer_idx, image_choice)

    with gr.Blocks() as demo:
        with gr.Tab("Live Image") as live_image_tab:
            with gr.Row():
                gr.Image(main_class.base_image_rgb, label='Target Image')
                live_image = gr.Image(label='Live Image',
                                      sources=['webcam'],
                                      streaming=True,
                                      mirror_webcam=False,
                                      height=895,
                                      width=1192,
                                      )
            # button_save_background = gr.Button('Save Background')
            # button_save_background.click(fn=main_class.set_background_rgb, inputs=live_image)

        with gr.Tab("Paint Guide") as paint_guide_tab:
            value_map_int = gradio.Number(label='Value Map', value=0, precision=0, minimum=0,
                                          maximum=len(main_class.value_maps) - 1)
            # good_image_bool = gr.Checkbox(True, visible=True)
            image_choice = gr.Radio(['Base Image', 'Good Paint', 'Bad Paint', 'Current Image'],
                                    value='Base Image', visible=True)
            # bad_image_bool = gr.Checkbox(False, visible=False)
            gr.Interface(fn=show_progress, inputs=[live_image, value_map_int, image_choice], outputs="image", live=True)
            # gr.Interface(fn=show_progress, inputs=[live_image, value_map_int, bad_image_bool], outputs="image", live=True)


    demo.launch()

    pass

# base_image_fname = os.path.join(main_class.image_folder, "value_map_0.png")
# base_image_rgb = imread(base_image_fname)
#
# live_image_fname = os.path.join(main_class.image_folder, "image_0_live_value_map_0.png")
# live_image_rgb = imread(live_image_fname)
#
# test_image_lab = main_class.compare_images(live_image_rgb=live_image_rgb, base_image_rgb=base_image_rgb)
# test_image_lab_0 = test_image_lab[:, :, 0]
# test_image_lab_1 = test_image_lab[:, :, 1]
# test_image_lab_2 = test_image_lab[:, :, 2]
#
#
# background_mask = main_class.background_mask(live_image_rgb)
#
# valuemap_painted_match = main_class.value_map_masks[0] * ~background_mask[:, :, 0]
# valuemap_minus_painted = main_class.value_map_masks[0] * background_mask[:, :, 0]
#
#
# io.imshow(main_class.show_progress(live_image_rgb, curr_layer_idx=0))
# io.show()
