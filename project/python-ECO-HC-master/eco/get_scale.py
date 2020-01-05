#参照matlab代码中ECO-master\scale_filter文件夹中内容编写
import numpy as np
import scipy
import cv2
from numpy.fft import fft, ifft
from scipy import signal
import sys
sys.path.append("./eco")
from config import hc_config
config = hc_config.HCConfig()
from fourier_tools import resize_dft
from features import fhog


class get_scale:
    def __init__(self, target_sz, ): #初始化用于搜索范围的滤波器
        init_target_sz = target_sz
        num_scales = config.number_of_scales_filter
        scale_step = config.scale_step_filter
        scale_sigma = config.number_of_interp_scales * config.scale_sigma_factor

        scale_exp = np.arange(-np.floor(num_scales - 1)/2,
                              np.ceil(num_scales-1)/2+1,
                              dtype=np.float32) * config.number_of_interp_scales / num_scales
        scale_exp_shift = np.roll(scale_exp, (0, -int(np.floor((num_scales-1)/2))))

        interp_scale_exp = np.arange(-np.floor((config.number_of_interp_scales-1)/2),
                                     np.ceil((config.number_of_interp_scales-1)/2)+1,
                                     dtype=np.float32)
        interp_scale_exp_shift = np.roll(interp_scale_exp, [0, -int(np.floor(config.number_of_interp_scales-1)/2)])

        self.scale_size_factors = scale_step ** scale_exp
        self.interp_scale_factors = scale_step ** interp_scale_exp_shift

        ys = np.exp(-0.5 * (scale_exp_shift ** 2) / (scale_sigma ** 2))
        self.yf = np.real(fft(ys))[np.newaxis, :]
        self.window = signal.hann(ys.shape[0])[np.newaxis, :].astype(np.float32)

        if config.scale_model_factor**2 * np.prod(init_target_sz) > config.scale_model_max_area:
            scale_model_factor = np.sqrt(config.scale_model_max_area / np.prod(init_target_sz))
        else:
            scale_model_factor = config.scale_model_factor

        self.scale_model_sz = np.maximum(np.floor(init_target_sz * scale_model_factor), np.array([8, 8]))
        self.max_scale_dim = config.s_num_compressed_dim == 'MAX'
        if self.max_scale_dim:
            self.s_num_compressed_dim = len(self.scale_size_factors)

        self.num_scales = num_scales
        self.scale_step = scale_step
        self.scale_factors = np.array([1])

    def track(self, im, pos, base_target_sz, current_scale_factor): #基于预测中心点位置进行范围搜索

        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz)

        # 多分辨率特征图
        xs = self.basis.dot(xs) * self.window

        # 得到得分
        xsf = fft(xs, axis=1)
        scale_responsef = np.sum(self.sf_num * xsf, 0) / (self.sf_den + config.lamBda)
        interp_scale_response = np.real(ifft(resize_dft(scale_responsef, config.number_of_interp_scales)))
        recovered_scale_index = np.argmax(interp_scale_response)
        id1 = (recovered_scale_index - 1) % config.number_of_interp_scales
        id2 = (recovered_scale_index + 1) % config.number_of_interp_scales
        poly_x = np.array([self.interp_scale_factors[id1], self.interp_scale_factors[recovered_scale_index], self.interp_scale_factors[id2]])
        poly_y = np.array([interp_scale_response[id1], interp_scale_response[recovered_scale_index], interp_scale_response[id2]])
        poly_A = np.array([[poly_x[0]**2, poly_x[0], 1],
                           [poly_x[1]**2, poly_x[1], 1],
                           [poly_x[2]**2, poly_x[2], 1]], dtype=np.float32)
        poly = np.linalg.inv(poly_A).dot(poly_y.T)
        scale_change_factor = - poly[1] / (2 * poly[0])
        return scale_change_factor

    def update(self, im, pos, base_target_sz, current_scale_factor): #对于范围滤波器进行更新

        scales = current_scale_factor * self.scale_size_factors
        xs = self._extract_scale_sample(im, pos, base_target_sz, scales, self.scale_model_sz)

        first_frame = not hasattr(self, 's_num')

        if first_frame:
            self.s_num = xs
        else:
            self.s_num = (1 - config.scale_learning_rate) * self.s_num + config.scale_learning_rate * xs
        if self.max_scale_dim:
            self.basis, _ = scipy.linalg.qr(self.s_num, mode='economic')
            scale_basis_den, _ = scipy.linalg.qr(xs, mode='economic')
        else:
            U, _, _ = np.linalg.svd(self.s_num)
            self.basis = U[:, :self.s_num_compressed_dim]
        self.basis = self.basis.T


        feat_proj = self.basis.dot(self.s_num) * self.window
        sf_proj = fft(feat_proj, axis=1)
        self.sf_num = self.yf * np.conj(sf_proj)

        xs = scale_basis_den.T.dot(xs) * self.window
        xsf = fft(xs, axis=1)
        new_sf_den = np.sum(np.real(xsf * np.conj(xsf)), 0)
        if first_frame:
            self.sf_den = new_sf_den
        else:
            self.sf_den = (1 - config.scale_learning_rate) * self.sf_den + config.scale_learning_rate * new_sf_den

    def _extract_scale_sample(self, im, pos, base_target_sz, scale_factors, scale_model_sz): #提取范围滤波时的样本
        num_scales = len(scale_factors)

        scale_sample = []
        for idx, scale in enumerate(scale_factors):
            patch_sz = np.floor(base_target_sz * scale)

            xs = np.floor(pos[1]) + np.arange(0, patch_sz[1]+1) - np.floor(patch_sz[1]/2)
            ys = np.floor(pos[0]) + np.arange(0, patch_sz[0]+1) - np.floor(patch_sz[0]/2)
            xmin = max(0, int(xs.min()))
            xmax = min(im.shape[1], int(xs.max()))
            ymin = max(0, int(ys.min()))
            ymax = min(im.shape[0], int(ys.max()))

            im_patch = im[ymin:ymax, xmin:xmax :]

            left = right = top = down = 0
            if xs.min() < 0:
                left = int(abs(xs.min()))
            if xs.max() > im.shape[1]:
                right = int(xs.max() - im.shape[1])
            if ys.min() < 0:
                top = int(abs(ys.min()))
            if ys.max() > im.shape[0]:
                down = int(ys.max() - im.shape[0])
            if left != 0 or right != 0 or top != 0 or down != 0:
                im_patch = cv2.copyMakeBorder(im_patch, top, down, left, right, cv2.BORDER_REPLICATE)

            im_patch_resized = cv2.resize(im_patch,
                                          (int(scale_model_sz[0]),int(scale_model_sz[1])),
                                          cv2.INTER_CUBIC)
            scale_sample.append(fhog(im_patch_resized, 4)[:, :, :31].reshape((-1, 1)))
        scale_sample = np.concatenate(scale_sample, axis=1)
        return scale_sample
