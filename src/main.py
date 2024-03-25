import time
from typing import List

import gradio.components.image
import sklearn.metrics
from PIL.Image import Image
from skimage.exposure import equalize_adapthist
from skimage.io import imread, imsave
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv, lab2rgb, rgb2lab, deltaE_ciede94, hsv2rgb
from sklearn.mixture import GaussianMixture
from skimage.color import rgb2gray
from skimage.filters import meijering, sato, frangi
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from sklearn.metrics.pairwise import euclidean_distances
from skimage import transform
from sklearn.neighbors import NearestCentroid
from numpy import quantile
import numpy as np
import os
import gradio as gr

class MainClass:
    def __init__(self,
                 base_image_name: str,
                 image_folder: str = r"D:\paint-helper\images",
                 n_quantiles: int = 8,
                 image_width_pix: int = 1000,
                 image_hegiht_pix: int = 800,
                 ):
        self.image_width_pix = image_width_pix,
        self.image_hegiht_pix = image_hegiht_pix
        self.image_folder = image_folder
        self.base_image_name = base_image_name

        base_image_fname = os.path.join(self.image_folder, "00106-2904245478.png")
        self.base_image_rgb = imread(base_image_fname)
        # self.base_image_rgb = resize(self.base_image_rgb,
        #                              (image_hegiht_pix, image_width_pix, self.base_image_rgb.shape[2]),
        #                              preserve_range=True).astype(int)
        self.base_image_hsv = rgb2hsv(self.base_image_rgb[:, :, :3])
        # self.base_image_hsv = rgb2hsv(self.base_image_rgb[:, :, :3])

        self.n_quantiles = n_quantiles
        self.make_value_maps(n_quantiles=n_quantiles)
        self.background_rgb = None
        self.corner_dict = {'ul': [],
                            'ur': [],
                            'lr': [],
                            'll': []}

    def modify_hsv_from_rgb(self, image_rgb: np.ndarray, s_aug: float = 1.0, v_aug: float = 1.0) -> np.ndarray:
        image_hsv = rgb2hsv(image_rgb)
        image_hsv[:, :, 1] *= s_aug
        image_hsv[:, :, 2] *= v_aug

        image_mod_rgb = hsv2rgb(image_hsv)
        image_mod_rgb = (image_mod_rgb * 255).astype(np.uint8)

        return image_mod_rgb

    def get_canvas_edge_candidates(self,
                                   image_gray,
                                   filter_sigmas=[5],
                                   harris_sigma=5,
                                   min_distance=2,
                                   threshold_rel=0.01,
                                   black_ridges=True):

        filter = sato
        edge = filter(image_gray, sigmas=filter_sigmas, black_ridges=black_ridges)

        harris_output = corner_harris(edge, sigma=harris_sigma)
        coords = corner_peaks(harris_output, min_distance=min_distance, threshold_rel=threshold_rel)

        return coords

    def update_canvas_edge_coords(self, coords):
        """
        Updates the internal coords of the canvas in the webcam image.
        """
        self.corner_dict

    def register_image(self, image, image_pct: float = 0.1, distance_thresh: float = 5):
        """
        image_pct is the percent distance from the edges we consider points as possible edge candidates.
        distance_thresh is the distance from the
        """
        image_gray = rgb2gray(image)
        img_adapteq = equalize_adapthist(image_gray, clip_limit=0.01)

        coords_dict = {f'black_ridges_{("true" if black_ridges == True else "false")}':
            self.get_canvas_edge_candidates(
                img_adapteq,
                filter_sigmas=[5],
                harris_sigma=5,
                min_distance=5,
                threshold_rel=0.01,
                black_ridges=black_ridges) for black_ridges in [True, False]}

        ul_coord_options_br_true = [(y, x) for (x, y) in coords_dict['black_ridges_true'] if
                            x < image.shape[0] * image_pct and y < image.shape[1] * image_pct]
        ur_coord_options_br_true = [(y, x) for (x, y) in coords_dict['black_ridges_true'] if
                            x < image.shape[0] * image_pct and y > image.shape[1] * (1 - image_pct)]
        lr_coord_options_br_true = [(y, x) for (x, y) in coords_dict['black_ridges_true'] if
                            x > image.shape[0] * (1 - image_pct) and y > image.shape[1] * (1 - image_pct)]
        ll_coord_options_br_true = [(y, x) for (x, y) in coords_dict['black_ridges_true'] if
                            x > image.shape[0] * (1 - image_pct) and y < image.shape[1] * image_pct]

        if len(ul_coord_options_br_true) == 0:
            ul_coord_options_br_true = [[0, 0]]
        if len(ur_coord_options_br_true) == 0:
            ur_coord_options_br_true = [[image.shape[1], 0]]
        if len(lr_coord_options_br_true) == 0:
            lr_coord_options_br_true = [[image.shape[1], image.shape[0]]]
        if len(ll_coord_options_br_true) == 0:
            ll_coord_options_br_true = [[0, image.shape[0]]]

        ul_coord_options_br_false = [
            (y, x) for (x, y) in coords_dict['black_ridges_false'] if
            euclidean_distances(np.array((y, x)).reshape(1, -1),
                                np.average(ul_coord_options_br_true, axis=0).reshape(1, -1)) < distance_thresh]
        ur_coord_options_br_false = [
            (y, x) for (x, y) in coords_dict['black_ridges_false'] if
            euclidean_distances(np.array((y, x)).reshape(1, -1),
                                np.average(ur_coord_options_br_true, axis=0).reshape(1, -1)) < distance_thresh]
        lr_coord_options_br_false = [
            (y, x) for (x, y) in coords_dict['black_ridges_false'] if
            euclidean_distances(np.array((y, x)).reshape(1, -1),
                                np.average(lr_coord_options_br_true, axis=0).reshape(1, -1)) < distance_thresh]
        ll_coord_options_br_false = [
            (y, x) for (x, y) in coords_dict['black_ridges_false'] if
            euclidean_distances(np.array((y, x)).reshape(1, -1),
                                np.average(ll_coord_options_br_true, axis=0).reshape(1, -1)) < distance_thresh]

        ul_coord_options = [x for x in ul_coord_options_br_true] + [x for x in ul_coord_options_br_false]
        ur_coord_options = [x for x in ur_coord_options_br_true] + [x for x in ur_coord_options_br_false]
        lr_coord_options = [x for x in lr_coord_options_br_true] + [x for x in lr_coord_options_br_false]
        ll_coord_options = [x for x in ll_coord_options_br_true] + [x for x in ll_coord_options_br_false]

        x_padding = 10
        y_padding = 10

        dst = np.array([
            list(np.average(ul_coord_options, axis=0) + np.array([-x_padding, -y_padding])),
            list(np.average(ur_coord_options, axis=0) + np.array([x_padding, -y_padding])),
            list(np.average(lr_coord_options, axis=0) + np.array([x_padding, y_padding])),
            list(np.average(ll_coord_options, axis=0) + np.array([-x_padding, y_padding])),
        ])
        src = np.array([
            [0, 0],  # ul
            [self.base_image_rgb.shape[1], 0],  # ur
            [self.base_image_rgb.shape[1], self.base_image_rgb.shape[0]],  # lr
            [0, self.base_image_rgb.shape[0]]  # ll
        ])

        tform3 = transform.ProjectiveTransform()
        tform3.estimate(src, dst)
        live_image_warped_rgb = transform.warp(image[:, :, :3], tform3, output_shape=(
            self.base_image_rgb.shape[0], self.base_image_rgb.shape[1]))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=plt.cm.gray)
        ax.plot(coords_dict['black_ridges_true'][:, 1], coords_dict['black_ridges_true'][:, 0], color='cyan', marker='o',
                linestyle='None', markersize=6)
        plt.savefig(os.path.join(self.image_folder, f'test_test.png'))

        return live_image_warped_rgb

    def show_progress(self, live_image_rgb: np.ndarray, curr_layer_idx: int = 0, image_choice: str = "Base Image", s_aug: float = 0.75, v_aug: float = 0.5) -> List[np.ndarray]:
        """
        Shows where the current live image matches the current layer. Match is checked both spatially, does the pixel
        fall on the layer's mask, and if the color matches the layer.
        """
        if self.get_background_rgb() is None:
            return None

        live_image_registered_rgb = self.register_image(live_image_rgb)

        # live_image_rgb = resize(live_image_rgb, self.base_image_rgb.shape, preserve_range=True)
        live_image_registered_rgb = (live_image_registered_rgb*255).astype(np.uint8)

        background_mask = self.background_mask(live_image_registered_rgb)
        valuemap_minus_painted_mask = self.value_map_masks[curr_layer_idx] * background_mask[:, :, 0]
        valuemap_minus_painted_mask_3d = np.repeat(np.expand_dims(valuemap_minus_painted_mask, 2), 3, 2)

        # ================
        live_image_rgb_flat = live_image_registered_rgb.reshape(
            [live_image_registered_rgb.shape[0] * live_image_registered_rgb.shape[1], live_image_registered_rgb.shape[2]])[:, :3]
        pred_live_image_clusters_flat = self.cluster_map_discriminator.predict(live_image_rgb_flat)
        pred_live_image_clusters = pred_live_image_clusters_flat.reshape(live_image_registered_rgb.shape[0],
                                                                         live_image_registered_rgb.shape[1])


        correct_paint_mask = (pred_live_image_clusters == curr_layer_idx) * self.value_map_masks[curr_layer_idx]
        correct_paint_mask_3d = np.repeat(np.expand_dims(correct_paint_mask, 2), 3, 2)

        wrong_paint_mask = (pred_live_image_clusters != curr_layer_idx) * self.value_map_masks[curr_layer_idx] * (
                    pred_live_image_clusters != -1)
        wrong_paint_mask_3d = np.repeat(np.expand_dims(wrong_paint_mask, 2), 3, 2)

        outside_curr_layer_and_paint_match_inside_mask = (pred_live_image_clusters == curr_layer_idx) * ~self.value_map_masks[
            curr_layer_idx]
        outside_curr_layer_and_paint_match_inside_mask_3d = np.repeat(
            np.expand_dims(outside_curr_layer_and_paint_match_inside_mask, 2), 3, 2)

        inside_curr_layer_and_paint_match_inside_mask = (pred_live_image_clusters == curr_layer_idx) * \
            self.value_map_masks[curr_layer_idx]
        inside_curr_layer_and_paint_match_inside_mask_3d = np.repeat(
            np.expand_dims(inside_curr_layer_and_paint_match_inside_mask, 2), 3, 2)

        test_image_rgb = np.where(correct_paint_mask_3d,
                                  live_image_registered_rgb[:, :, :3],
                                  self.value_maps[curr_layer_idx],
                                  )
        value_map_mod = self.modify_hsv_from_rgb(self.value_maps[curr_layer_idx], 0.5, 0.5)
        live_image_hsv_mod_rgb = self.modify_hsv_from_rgb(live_image_registered_rgb[:, :, :3], 0.8, 0.5)

        wrong_paint_image_rgb = np.where(wrong_paint_mask_3d,
                                         self.background_rgb[:, :, :3], # background to see that it's good, fill in the rest, not live_image_rgb[:, :, :3],
                                         live_image_hsv_mod_rgb[:, :, :3], # inside is actual live area, not live_image_mod_rgb
                                         )
        correct_paint_image_rgb = np.where(inside_curr_layer_and_paint_match_inside_mask_3d,
                                           live_image_hsv_mod_rgb[:, :, :3],
                                           live_image_registered_rgb[:, :, :3
                                           ], # self.value_maps[curr_layer_idx],
                                           )

        image_diff_rgb = np.where(
            inside_curr_layer_and_paint_match_inside_mask_3d,
            live_image_hsv_mod_rgb[:, :, :3],
            (self.value_maps[curr_layer_idx][:, :, :3] * 0.5 - live_image_registered_rgb[:, :, :3] * 0.5).astype(np.uint8),
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
        if image_choice == 'wrong_paint_mask_3d':
            return (wrong_paint_mask_3d*255).astype(np.uint8)
        if image_choice == 'outside_curr_layer_and_paint_match_inside_mask_3d':
            return (inside_curr_layer_and_paint_match_inside_mask_3d*255).astype(np.uint8)
        if image_choice == 'image_diff_rgb':
            return image_diff_rgb
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
                                     self.value_maps[key][:, :, :3],
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
    # image_folder = 'images'
    main_class = MainClass(
        # image_folder=image_folder,
        base_image_name="00106-2904245478.png")
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
                                      height=800,
                                      width=1000,
                                      )
            # button_save_background = gr.Button('Save Background')
            # button_save_background.click(fn=main_class.set_background_rgb, inputs=live_image)

        with gr.Tab("Paint Guide") as paint_guide_tab:
            value_map_int = gradio.Number(label='Value Map', value=0, precision=0, minimum=0,
                                          maximum=len(main_class.value_maps) - 1)
            # good_image_bool = gr.Checkbox(True, visible=True)
            image_choice = gr.Radio(['Base Image', 'Good Paint', 'Bad Paint', 'Current Image', 'wrong_paint_mask_3d', 'outside_curr_layer_and_paint_match_inside_mask_3d',
                                     'image_diff_rgb'],
                                    value='Base Image', visible=True)
            # bad_image_bool = gr.Checkbox(False, visible=False)
            gr.Interface(title='', fn=show_progress, inputs=[live_image, value_map_int, image_choice], outputs="image", live=True)
            # gr.Interface(fn=show_progress, inputs=[live_image, value_map_int, bad_image_bool], outputs="image", live=True)


    demo.launch(
        server_port=8000,
        server_name="0.0.0.0"
    )
    # demo.launch()

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
