from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy import number

from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization
import medpy.filter.smoothing as medpy_smooth
from scipy import stats
from scipy import optimize


class Clip_Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        image = image.astype(self.target_dtype)
        print(self.intensityproperties)
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = -150
        upper_bound = 350
        image = np.clip(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image


class ComputeILP(object):
    """Compute intensity-based lesion probability (ILP) for each voxel

    Args:
        target_int_file
    """

    def __init__(self, hu_range=[-350, 150], shift=0, \
                 target_int_file='/home/compai/code/data/nnunet_raw/nnunet_dataset/Dataset020_LNQ/ilp/tumor_int_hist_list_sbct.npy'):

        assert isinstance(hu_range, (tuple, list))
        self.hu_range = np.array(hu_range)  # only HU values within this range will be considered, otherwise 0
        self.shift = shift
        if self.shift != 0:
            self.hu_range = self.hu_range + shift

        tumor_int_hist_list = np.load(target_int_file)

        ### histogram ###
        if 'sbct' in target_int_file:
            hist_bin_min = -350
            hist_bin_max = 150
            hist_bin_size = 10
            self.denois_niter = 5
            self.resize_factor = 1
        else:
            raise NotImplementedError

        bins = list(range(hist_bin_min, hist_bin_max + 1, hist_bin_size))

        cum_hist = np.zeros(tumor_int_hist_list.shape[1])
        for idx in range(tumor_int_hist_list.shape[0]):
            norm_hist = tumor_int_hist_list[idx] / np.sum(tumor_int_hist_list[idx])
            cum_hist = cum_hist + norm_hist
        cum_hist = cum_hist / tumor_int_hist_list.shape[0]

        ### KDE ###
        N = 1000
        data = []
        for idx in range(len(cum_hist)):
            cur_ub = bins[idx + 1]
            cur_data = [cur_ub - hist_bin_size * 0.5] * int(np.round(N * cum_hist[idx]))
            data += cur_data

        kde = stats.gaussian_kde(data)

        # check out the maximum
        opt = optimize.minimize_scalar(lambda x: -kde(x))
        # print('max : %.5f @ %.5f'%(-opt.fun[0],opt.x[0]))

        self.ilp_func = kde
        self.ilp_scale = -opt.fun[0]

    def __call__(self, sample):

        hu_range = [-350, 150]

        sample_deformed = {}
        orig_shape = cur_data.shape

        cur_data = medpy_smooth.anisotropic_diffusion(cur_data, \
                                                      niter=self.denois_niter, kappa=50, gamma=0.1,
                                                      voxelspacing=None, option=3)

        mask = (cur_data >= hu_range[0]) & (cur_data <= hu_range[1])
        in_val = cur_data[mask]
        out_val = self.ilp_func(in_val)
        out_val = out_val / self.ilp_scale
        ilp = np.zeros(orig_shape)
        ilp[mask] = out_val
        return ilp