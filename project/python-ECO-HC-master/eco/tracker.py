#参考了matlab代码ECO-master\implementation部分tracker.m文件
import numpy as np
import cv2
import scipy
import time


from scipy import signal
import sys
sys.path.append("./eco")
from config import hc_config
config = hc_config.HCConfig()
from features import FHogFeature, TableFeature
from fourier_tools import cfft2, interpolate_dft, shift_sample, full_fourier_coeff,\
        cubic_spline_fourier, compact_fourier_coeff, ifft2, fft2, sample_fs
from maximize_score import maximize_score
from sample_space_model import GMM
from train import train_joint, train_filter
from get_scale import get_scale

class ECOTracker:
    def __init__(self, is_color):
        self._is_color = is_color
        self._frame_num = 0
        self._frames_since_last_train = 0

    def select_sample(self, x, P):
        return [np.matmul(P_.T, x_) for x_, P_ in zip(x, P)]

    def _cosine_window(self, size): #构造余弦窗
        cos_window = np.hanning(int(size[0]+2))[:, np.newaxis].dot(np.hanning(int(size[1]+2))[np.newaxis, :])
        cos_window = cos_window[1:-1, 1:-1][:, :, np.newaxis, np.newaxis].astype(np.float32)
        return cos_window

    def _get_interp_fourier(self, sz): #计算插值函数的傅立叶级数
        f1 = np.arange(-(sz[0]-1) / 2, (sz[0]-1)/2+1, dtype=np.float32)[:, np.newaxis] / sz[0]
        interp1_fs = np.real(cubic_spline_fourier(f1, config.interp_bicubic_a) / sz[0])
        f2 = np.arange(-(sz[1]-1) / 2, (sz[1]-1)/2+1, dtype=np.float32)[np.newaxis, :] / sz[1]
        interp2_fs = np.real(cubic_spline_fourier(f2, config.interp_bicubic_a) / sz[1])
        if config.interp_centering:
            f1 = np.arange(-(sz[0]-1) / 2, (sz[0]-1)/2+1, dtype=np.float32)[:, np.newaxis]
            interp1_fs = interp1_fs * np.exp(-1j*np.pi / sz[0] * f1)
            f2 = np.arange(-(sz[1]-1) / 2, (sz[1]-1)/2+1, dtype=np.float32)[np.newaxis, :]
            interp2_fs = interp2_fs * np.exp(-1j*np.pi / sz[1] * f2)

        if config.interp_windowing:
            win1 = np.hanning(sz[0]+2)[:, np.newaxis]
            win2 = np.hanning(sz[1]+2)[np.newaxis, :]
            interp1_fs = interp1_fs * win1[1:-1]
            interp2_fs = interp2_fs * win2[1:-1]

        return (interp1_fs[:, :, np.newaxis, np.newaxis],
                    interp2_fs[:, :, np.newaxis, np.newaxis])
        
    def _get_reg_filter(self, sz, target_sz, reg_window_edge): #构造空间归一化函数，以使用用于优化响应滤波器操作
        if config.use_reg_window:
            reg_scale = 0.5 * target_sz

            # 构造梯度
            wrg = np.arange(-(sz[0]-1)/2, (sz[1]-1)/2+1, dtype=np.float32)
            wcg = np.arange(-(sz[0]-1)/2, (sz[1]-1)/2+1, dtype=np.float32)
            wrs, wcs = np.meshgrid(wrg, wcg)

            # 构造归一化窗
            reg_window = (reg_window_edge - config.reg_window_min) * (np.abs(wrs/reg_scale[0])**config.reg_window_power + \
                            np.abs(wcs/reg_scale[1])**config.reg_window_power) + config.reg_window_min

            # 计算DFT并增强稀疏性（阈值收缩）
            reg_window_dft = fft2(reg_window) / np.prod(sz)
            reg_window_dft[np.abs(reg_window_dft) < config.reg_sparsity_threshold* np.max(np.abs(reg_window_dft.flatten()))] = 0

            # 做反变换，修正窗最小值
            reg_window_sparse = np.real(ifft2(reg_window_dft))
            reg_window_dft[0, 0] = reg_window_dft[0, 0] - np.prod(sz) * np.min(reg_window_sparse.flatten()) + config.reg_window_min
            reg_window_dft = np.fft.fftshift(reg_window_dft).astype(np.complex64)

            # 去除零点以找到正则化滤波器
            row_idx = np.logical_not(np.all(reg_window_dft==0, axis=1))
            col_idx = np.logical_not(np.all(reg_window_dft==0, axis=0))
            mask = np.outer(row_idx, col_idx)
            reg_filter = np.real(reg_window_dft[mask]).reshape(np.sum(row_idx), -1)
        else:
            # 否则使用比例恒等矩阵
            reg_filter = config.reg_window_min

        return reg_filter.T
        

    def _init_proj_matrix(self, init_sample, compressed_dim, proj_method):  
        # 初始化投影矩阵
        x = [np.reshape(x, (-1, x.shape[2])) for x in init_sample]
        x = [z - z.mean(0) for z in x]
        proj_matrix_ = []
        if config.proj_init_method == 'pca':
            for x_, compressed_dim_  in zip(x, compressed_dim):
                proj_matrix, _, _ = np.linalg.svd(x_.T.dot(x_))
                proj_matrix = proj_matrix[:, :compressed_dim_]
                proj_matrix_.append(proj_matrix)
        elif config.proj_init_method == 'rand_uni':
            for x_, compressed_dim_ in zip(x, compressed_dim):
                proj_matrix = np.random.uniform(size=(x_.shape[1], compressed_dim_))
                proj_matrix /= np.sqrt(np.sum(proj_matrix**2, axis=0, keepdims=True))
                proj_matrix_.append(proj_matrix)
        return proj_matrix_

    def init(self, frame, bbox, total_frame=np.inf):
        self._pos = np.array([bbox[1]+(bbox[3]-1)/2., bbox[0]+(bbox[2]-1)/2.], dtype=np.float32)
        self._target_sz = np.array([bbox[3], bbox[2]])
        self._num_samples = min(config.num_samples, total_frame)
        #np = np

        # 计算搜索区域并初始化比例尺
        search_area = np.prod(self._target_sz * config.search_area_scale)
        if search_area > config.max_image_sample_size:
            self._current_scale_factor = np.sqrt(search_area / config.max_image_sample_size)
        elif search_area < config.min_image_sample_size:
            self._current_scale_factor = np.sqrt(search_area / config.min_image_sample_size)
        else:
            self._current_scale_factor = 1.

        # 在初始化比例尺下的目标尺寸
        self._base_target_sz = self._target_sz / self._current_scale_factor

        # 目标大小（将padding考虑在内）
        self._img_sample_sz = np.ones((2), dtype=np.float32) * np.sqrt(np.prod(self._base_target_sz * config.search_area_scale))
        
        features = [feature for feature in config.features
                if ("use_for_color" in feature and feature["use_for_color"] == self._is_color) or
                    "use_for_color" not in feature]

        self._features = []
        for idx, feature in enumerate(features):
            if feature['fname'] == 'cn' or feature['fname'] == 'ic':
                self._features.append(TableFeature(**feature))
            elif feature['fname'] == 'fhog':
                self._features.append(FHogFeature(**feature))
            else:
                raise("unimplemented features")
        self._features = sorted(self._features, key=lambda x:x.min_cell_size)

        # 计算图像采样尺寸
        cell_size = [x.min_cell_size for x in self._features]
        self._img_sample_sz = self._features[0].init_size(self._img_sample_sz, cell_size)

        for idx, feature in enumerate(self._features):
            feature.init_size(self._img_sample_sz)

        if config.use_projection_matrix:
            sample_dim = [ x for feature in self._features for x in feature._compressed_dim ]
        else:
            sample_dim = [ x for feature in self._features for x in feature.num_dim ]

        feature_dim = [ x for feature in self._features for x in feature.num_dim ]

        feature_sz = np.array([x for feature in self._features for x in feature.data_sz ], dtype=np.int32)

        # 为每个滤波层保存的傅立叶系数数量（注意应当为偶数）
        filter_sz = feature_sz + (feature_sz + 1) % 2

        # 标签函数DFT的尺寸，应和最大滤波器的尺寸相同
        self._k1 = np.argmax(filter_sz, axis=0)[0]
        self._output_sz = filter_sz[self._k1]

        self._num_feature_blocks = len(feature_dim)

        # 得到其他块的索引
        self._block_inds = list(range(self._num_feature_blocks))
        self._block_inds.remove(self._k1)

        # 计算padding的size，为了适应outputsize的尺寸
        self._pad_sz = [((self._output_sz - filter_sz_) / 2).astype(np.int32) for filter_sz_ in filter_sz]

        # 计算傅立叶级数和其转制
        self._ky = [np.arange(-np.ceil(sz[0]-1)/2, np.floor((sz[0]-1)/2)+1, dtype=np.float32)
                        for sz in filter_sz]
        self._kx = [np.arange(-np.ceil(sz[1]-1)/2, 1, dtype=np.float32)
                        for sz in filter_sz]

        # 用柏松公式构造高斯标签函数
        sig_y = np.sqrt(np.prod(np.floor(self._base_target_sz))) * config.output_sigma_factor * (self._output_sz / self._img_sample_sz)
        yf_y = [np.sqrt(2 * np.pi) * sig_y[0] / self._output_sz[0] * np.exp(-2 * (np.pi * sig_y[0] * ky_ / self._output_sz[0])**2)
                    for ky_ in self._ky]
        yf_x = [np.sqrt(2 * np.pi) * sig_y[1] / self._output_sz[1] * np.exp(-2 * (np.pi * sig_y[1] * kx_ / self._output_sz[1])**2)
                    for kx_ in self._kx]
        self._yf = [yf_y_.reshape(-1, 1) * yf_x_ for yf_y_, yf_x_ in zip(yf_y, yf_x)]

        # 构造余弦窗
        self._cos_window = [self._cosine_window(feature_sz_) for feature_sz_ in feature_sz]

        # 为插值函数计算傅立叶系数
        self._interp1_fs = []
        self._interp2_fs = []
        for sz in filter_sz:
            interp1_fs, interp2_fs = self._get_interp_fourier(sz)
            self._interp1_fs.append(interp1_fs)
            self._interp2_fs.append(interp2_fs)

        # 获得reg_window_edge的参数
        reg_window_edge = []
        for feature in self._features:
            if hasattr(feature, 'reg_window_edge'):#hasattr 对象是否包含已知的属性
                reg_window_edge.append(feature.reg_window_edge)
            else:
                reg_window_edge += [config.reg_window_edge for _ in range(len(feature.num_dim))]

        # 构造空间与归一化
        self._reg_filter = [self._get_reg_filter(self._img_sample_sz, self._base_target_sz, reg_window_edge_)
                                for reg_window_edge_ in reg_window_edge]

        # 计算滤波器能量
        self._reg_energy = [np.real(np.vdot(reg_filter.flatten(), reg_filter.flatten()))
                        for reg_filter in self._reg_filter]
        
        
        self.get_scale = get_scale(self._target_sz)
        self._num_scales = self.get_scale.num_scales
        self._scale_step = self.get_scale.scale_step
        self._scale_factor = self.get_scale.scale_factors

        if self._num_scales > 0:
            # 比例调整
            self._min_scale_factor = self._scale_step ** np.ceil(np.log(np.max(5 / self._img_sample_sz)) / np.log(self._scale_step))
            self._max_scale_factor = self._scale_step ** np.floor(np.log(np.min(frame.shape[:2] / self._base_target_sz)) / np.log(self._scale_step))

        # 共轭梯度选项
        init_CG_opts = {'CG_use_FR': True,
                        'tol': 1e-6,
                        'CG_standard_alpha': True
                       }
        self._CG_opts = {'CG_use_FR': config.CG_use_FR,
                         'tol': 1e-6,
                         'CG_standard_alpha': config.CG_standard_alpha
                        }
        if config.CG_forgetting_rate == np.inf or config.learning_rate >= 1:
            self._CG_opts['init_forget_factor'] = 0.
        else:
            self._CG_opts['init_forget_factor'] = (1 - config.learning_rate) ** config.CG_forgetting_rate

        # 初始化
        self._gmm = GMM(self._num_samples)
        self._samplesf = [[]] * self._num_feature_blocks

        for i in range(self._num_feature_blocks):
            self._samplesf[i] = np.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                sample_dim[i], config.num_samples), dtype=np.complex64)
        
        self._num_training_samples = 0

        # 提取样本，初始化投影矩阵
        sample_pos = np.round(self._pos)
        sample_scale = self._current_scale_factor
        xl = [x for feature in self._features
                for x in feature.get_features(frame, sample_pos, self._img_sample_sz, self._current_scale_factor) ]  

        xlw = [x * y for x, y in zip(xl, self._cos_window)]                                                          
        xlf = [cfft2(x) for x in xlw]                                                                                
        xlf = interpolate_dft(xlf, self._interp1_fs, self._interp2_fs)                                              
        xlf = compact_fourier_coeff(xlf)                                                                             
        shift_sample_ = 2 * np.pi * (self._pos - sample_pos) / (sample_scale * self._img_sample_sz)
        xlf = shift_sample(xlf, shift_sample_, self._kx, self._ky)
        self._proj_matrix = self._init_proj_matrix(xl, sample_dim, config.proj_init_method)
        xlf_proj = self.select_sample(xlf, self._proj_matrix)
        merged_sample, new_sample, merged_sample_id, new_sample_id = \
            self._gmm.update_sample_space_model(self._samplesf, xlf_proj, self._num_training_samples)
        self._num_training_samples += 1

        if config.update_projection_matrix:
            for i in range(self._num_feature_blocks):
                self._samplesf[i][:, :, :, new_sample_id:new_sample_id+1] = new_sample[i]

        # 训练tracker
        self._sample_energy = [np.real(x * np.conj(x)) for x in xlf_proj]

        # 初始化共轭梯度参数
        self._CG_state = None
        if config.update_projection_matrix:
            init_CG_opts['maxit'] = np.ceil(config.init_CG_iter / config.init_GN_iter)
            self._hf = [[[]] * self._num_feature_blocks for _ in range(2)]
            feature_dim_sum = float(np.sum(feature_dim))
            proj_energy = [2 * np.sum(np.abs(yf_.flatten())**2) / feature_dim_sum * np.ones_like(P)
                    for P, yf_ in zip(self._proj_matrix, self._yf)]
        else:
            self._CG_opts['maxit'] = config.init_CG_iter
            self._hf = [[[]] * self._num_feature_blocks]

        # 初始化滤波器
        for i in range(self._num_feature_blocks):
            self._hf[0][i] = np.zeros((int(filter_sz[i, 0]), int((filter_sz[i, 1]+1)/2),
                int(sample_dim[i]), 1), dtype=np.complex64)

        if config.update_projection_matrix:
            # 初始化高斯牛顿优化器和投影矩阵参数
            self._hf, self._proj_matrix = train_joint(
                                                  self._hf,
                                                  self._proj_matrix,
                                                  xlf,
                                                  self._yf,
                                                  self._reg_filter,
                                                  self._sample_energy,
                                                  self._reg_energy,
                                                  proj_energy,
                                                  init_CG_opts)
            # 插入训练样本重新规划
            xlf_proj = self.select_sample(xlf, self._proj_matrix)
            for i in range(self._num_feature_blocks):
                self._samplesf[i][:, :, :, 0:1] = xlf_proj[i]

            # 更新矩阵
            if config.distance_matrix_update_type == 'exact':
                new_train_sample_norm = 0.
                for i in range(self._num_feature_blocks):
                    new_train_sample_norm += 2 * np.real(np.vdot(xlf_proj[i].flatten(), xlf_proj[i].flatten()))
                self._gmm._gram_matrix[0, 0] = new_train_sample_norm
        self._hf_full = full_fourier_coeff(self._hf)

        if self._num_scales > 0:
            self.get_scale.update(frame, self._pos, self._base_target_sz, self._current_scale_factor)
        self._frame_num += 1

    def update(self, frame, train=True, vis=False):
        # 目标定位步骤
        pos = self._pos
        old_pos = np.zeros((2))
        for _ in range(config.refinement_iterations):
            if not np.allclose(old_pos, pos):
                old_pos = pos.copy()
                sample_pos = np.round(pos)
                sample_scale = self._current_scale_factor * self._scale_factor
                xt = [x for feature in self._features
                        for x in feature.get_features(frame, sample_pos, self._img_sample_sz, sample_scale) ]  

                xt_select = self.select_sample(xt, self._proj_matrix)                                            
                xt_select = [feat_map_ * cos_window_
                        for feat_map_, cos_window_ in zip(xt_select, self._cos_window)]                         
                xtf_select = [cfft2(x) for x in xt_select]                                                         
                xtf_select = interpolate_dft(xtf_select, self._interp1_fs, self._interp2_fs)                       

                # 计算每个特征块的卷积（通过频域）再加和
                scores_fs_feat = [[]] * self._num_feature_blocks#获取格式
                scores_fs_feat[self._k1] = np.sum(self._hf_full[self._k1] * xtf_select[self._k1], 2)
                scores_fs = scores_fs_feat[self._k1]

                
                for i in self._block_inds:
                    scores_fs_feat[i] = np.sum(self._hf_full[i] * xtf_select[i], 2)
                    scores_fs[self._pad_sz[i][0]:self._output_sz[0]-self._pad_sz[i][0],
                              self._pad_sz[i][1]:self._output_sz[0]-self._pad_sz[i][1]] += scores_fs_feat[i]

                # 获得position
                trans_row, trans_col, scale_idx = maximize_score(scores_fs, config.newton_iterations)

                # 测试得分
                if vis:
                    self.score = np.fft.fftshift(sample_fs(scores_fs[:,:,scale_idx],
                            tuple((10*self._output_sz).astype(np.uint32))))
                    self.crop_size = self._img_sample_sz * self._current_scale_factor

                # 计算像素坐标中的平移矢量并舍入到最接近整数像素
                translation_vec = np.array([trans_row, trans_col]) * (self._img_sample_sz / self._output_sz) * \
                                    self._current_scale_factor * self._scale_factor[scale_idx]
                scale_change_factor = self._scale_factor[scale_idx]

                # 更新position
                pos = sample_pos + translation_vec

                if config.clamp_position:
                    pos = np.maximum(np.array(0, 0), np.minimum(np.array(frame.shape[:2]), pos))

                # 用scale filter做跟踪
                if self._num_scales > 0:
                    scale_change_factor = self.get_scale.track(frame, pos, self._base_target_sz,
                           self._current_scale_factor)

                # 更新scale
                self._current_scale_factor *= scale_change_factor

                # 做调整确保规模不会过大或过小
                if self._current_scale_factor < self._min_scale_factor:
                    self._current_scale_factor = self._min_scale_factor
                elif self._current_scale_factor > self._max_scale_factor:
                    self._current_scale_factor = self._max_scale_factor

        # 更新整个模型
        if config.learning_rate > 0:
            sample_scale = sample_scale[scale_idx]
            xlf_proj = [xf[:, :(xf.shape[1]+1)//2, :, scale_idx:scale_idx+1] for xf in xtf_select]

            shift_sample_ = 2 * np.pi * (pos - sample_pos) / (sample_scale * self._img_sample_sz)
            xlf_proj = shift_sample(xlf_proj, shift_sample_, self._kx, self._ky)

        # 更新采样点
        merged_sample, new_sample, merged_sample_id, new_sample_id = \
                self._gmm.update_sample_space_model(self._samplesf, xlf_proj, self._num_training_samples)

        if self._num_training_samples < self._num_samples:
            self._num_training_samples += 1

        if config.learning_rate > 0:
            for i in range(self._num_feature_blocks):
                if merged_sample_id >= 0:
                    self._samplesf[i][:, :, :, merged_sample_id:merged_sample_id+1] = merged_sample[i]
                if new_sample_id >= 0:
                    self._samplesf[i][:, :, :, new_sample_id:new_sample_id+1] = new_sample[i]

        # 训练滤波器
        if self._frame_num < config.skip_after_frame or \
                self._frames_since_last_train >= config.train_gap:
            new_sample_energy = [np.real(xlf * np.conj(xlf)) for xlf in xlf_proj]
            self._CG_opts['maxit'] = config.CG_iter
            self._sample_energy = [(1 - config.learning_rate)*se + config.learning_rate*nse
                                for se, nse in zip(self._sample_energy, new_sample_energy)]

            # 滤波器共轭梯度优化
            self._hf, self._CG_state = train_filter(
                                                 self._hf,
                                                 self._samplesf,
                                                 self._yf,
                                                 self._reg_filter,
                                                 self._gmm.prior_weights,
                                                 self._sample_energy,
                                                 self._reg_energy,
                                                 self._CG_opts,
                                                 self._CG_state)
            # 构造完整的傅立叶级数
            self._hf_full = full_fourier_coeff(self._hf)
            self._frames_since_last_train = 0
        else:
            self._frames_since_last_train += 1
        self.get_scale.update(frame, pos, self._base_target_sz, self._current_scale_factor)

        # 更新目标尺寸
        self._target_sz = self._base_target_sz * self._current_scale_factor

        # 保存位置和当前帧数
        bbox = (pos[1] - self._target_sz[1]/2, # xmin
                pos[0] - self._target_sz[0]/2, # ymin
                pos[1] + self._target_sz[1]/2, # xmax
                pos[0] + self._target_sz[0]/2) # ymax
        self._pos = pos
        self._frame_num += 1
        return bbox
