#参考matlab代码ECO-master\sample_space_model编写
import numpy as np
import sys
sys.path.append("./eco")
from config import hc_config
config = hc_config.HCConfig()


class GMM:
    def __init__(self, num_samples): #用于初始化
        self._num_samples = num_samples
        self._distance_matrix = np.ones((num_samples, num_samples), dtype=np.float32) * np.inf
        self._gram_matrix = np.ones((num_samples, num_samples), dtype=np.float32) * np.inf
        self.prior_weights = np.zeros((num_samples, 1), dtype=np.float32)
        # 找出允许的最小采样的权重，drop权重下降的样本
        self.minimum_sample_weight = config.learning_rate * (1 - config.learning_rate)**(2*config.num_samples)



    def _find_gram_vector(self, samplesf, new_sample, num_training_samples): #用现有样本查找新样本的内积，用于距离计算
        gram_vector = np.inf * np.ones((config.num_samples))
        if num_training_samples > 0:
            ip = 0.
            for k in range(len(new_sample)):
                samplesf_ = samplesf[k][:, :, :, :num_training_samples]
                samplesf_ = samplesf_.reshape((-1, num_training_samples))
                new_sample_ = new_sample[k].flatten()
                ip += np.real(2 * samplesf_.T.dot(np.conj(new_sample_)))
            gram_vector[:num_training_samples] = ip
        return gram_vector

    def _merge_samples(self, sample1, sample2, w1, w2, sample_merge_type): #用于合并components
        alpha1 = w1 / (w1 + w2)
        alpha2 = 1 - alpha1
        if sample_merge_type == 'replace':
            merged_sample = sample1
        elif sample_merge_type == 'merge':
            num_feature_blocks = len(sample1)
            merged_sample = [alpha1 * sample1[k] + alpha2 * sample2[k] for k in range(num_feature_blocks)]
        return merged_sample

    def _update_distance_matrix(self, gram_vector, new_sample_norm, id1, id2, w1, w2): #用于更新样本的距离矩阵

        alpha1 = w1 / (w1 + w2)
        alpha2 = 1 - alpha1
        if id2 < 0:
            norm_id1 = self._gram_matrix[id1, id1]

            # 更新格拉姆矩阵
            if alpha1 == 0:
                self._gram_matrix[:, id1] = gram_vector
                self._gram_matrix[id1, :] = self._gram_matrix[:, id1]
                self._gram_matrix[id1, id1] = new_sample_norm
            elif alpha2 == 0:
                # 新样本8行
                pass
            else:
                # 合并新样本和现有样本
                self._gram_matrix[:, id1] = alpha1 * self._gram_matrix[:, id1] + alpha2 * gram_vector
                self._gram_matrix[id1, :] = self._gram_matrix[:, id1]
                self._gram_matrix[id1, id1] = alpha1 ** 2 * norm_id1 + alpha2 ** 2 * new_sample_norm + 2 * alpha1 * alpha2 * gram_vector[id1]

            # 更新距离矩阵
            self._distance_matrix[:, id1] = np.maximum(self._gram_matrix[id1, id1] + np.diag(self._gram_matrix) - 2 * self._gram_matrix[:, id1], 0)
            self._distance_matrix[id1, :] = self._distance_matrix[:, id1]
            self._distance_matrix[id1, id1] = np.inf
        else:
            if alpha1 == 0 or alpha2 == 0:
                raise("Error!")

            norm_id1 = self._gram_matrix[id1, id1]
            norm_id2 = self._gram_matrix[id2, id2]
            ip_id1_id2 = self._gram_matrix[id1, id2]

            # 处理现有样本并合并
            self._gram_matrix[:, id1] = alpha1 * self._gram_matrix[:, id1] + alpha2 * self._gram_matrix[:, id2]
            self._gram_matrix[id1, :] = self._gram_matrix[:, id1]
            self._gram_matrix[id1, id1] = alpha1 ** 2 * norm_id1 + alpha2 ** 2 * norm_id2 + 2 * alpha1 * alpha2 * ip_id1_id2
            gram_vector[id1] = alpha1 * gram_vector[id1] + alpha2 * gram_vector[id2]

            # 处理新样本
            self._gram_matrix[:, id2] = gram_vector
            self._gram_matrix[id2, :] = self._gram_matrix[:, id2]
            self._gram_matrix[id2, id2] = new_sample_norm

            # 更新距离矩阵
            self._distance_matrix[:, id1] = np.maximum(self._gram_matrix[id1, id1] + np.diag(self._gram_matrix) - 2 * self._gram_matrix[:, id1], 0)
            self._distance_matrix[id1, :] = self._distance_matrix[:, id1]
            self._distance_matrix[id1, id1] = np.inf
            self._distance_matrix[:, id2] = np.maximum(self._gram_matrix[id2, id2] + np.diag(self._gram_matrix) - 2 * self._gram_matrix[:, id2], 0)
            self._distance_matrix[id2, :] = self._distance_matrix[:, id2]
            self._distance_matrix[id2, id2] = np.inf

    def update_sample_space_model(self, samplesf, new_train_sample, num_training_samples):
        num_feature_blocks = len(new_train_sample)

        # 用现有样本找到新样本的内积
        gram_vector = self._find_gram_vector(samplesf, new_train_sample, num_training_samples)
        new_train_sample_norm = 0.

        for i in range(num_feature_blocks):
            new_train_sample_norm += np.real(2 * np.vdot(new_train_sample[i].flatten(), new_train_sample[i].flatten()))

        dist_vector = np.maximum(new_train_sample_norm + np.diag(self._gram_matrix) - 2 * gram_vector, 0)
        dist_vector[num_training_samples:] = np.inf

        merged_sample = []
        new_sample = []
        merged_sample_id = -1
        new_sample_id = -1

        if num_training_samples == config.num_samples:
            min_sample_id = np.argmin(self.prior_weights)
            min_sample_weight = self.prior_weights[min_sample_id]
            if min_sample_weight < self.minimum_sample_weight:
                # 如果任何先前权值小于最小权值
                # 用新样本替换
                # 更新距离矩阵和格里姆矩阵
                self._update_distance_matrix(gram_vector, new_train_sample_norm, min_sample_id, -1, 0, 1)

                # 先前的权值归一化，并将新样本获得的权值作为学习率
                self.prior_weights[min_sample_id] = 0
                self.prior_weights = self.prior_weights * (1 - config.learning_rate) / np.sum(self.prior_weights)
                self.prior_weights[min_sample_id] = config.learning_rate

                # 设置新样本和新样本位置
                new_sample_id = min_sample_id
                new_sample = new_train_sample
            else:
                #如果没有样本具有足够低的先验权重，那么我们要么合并新样本与现有样本，要么合并现有样本中的两个，并将新样本插入腾空位置。
                closest_sample_to_new_sample = np.argmin(dist_vector)
                new_sample_min_dist = dist_vector[closest_sample_to_new_sample]

                # 寻找现有样本中距离最近的一对
                closest_existing_sample_idx = np.argmin(self._distance_matrix.flatten())
                closest_existing_sample_pair = np.unravel_index(closest_existing_sample_idx, self._distance_matrix.shape)
                existing_samples_min_dist = self._distance_matrix[closest_existing_sample_pair[0], closest_existing_sample_pair[1]]
                closest_existing_sample1, closest_existing_sample2 = closest_existing_sample_pair
                if closest_existing_sample1 == closest_existing_sample2:
                    raise("Score matrix diagnoal filled wrongly")

                if new_sample_min_dist < existing_samples_min_dist:
                    #如果新样本到现有样本的最小距离小于任何现有样本之间的最小距离，则将新样本与最近的现有样本合并
                    self.prior_weights = self.prior_weights * (1 - config.learning_rate)

                    # 设置合并样本的位置
                    merged_sample_id = closest_sample_to_new_sample

                    # 提取现有样本并合并
                    existing_sample_to_merge = []
                    for i in range(num_feature_blocks):
                        existing_sample_to_merge.append(samplesf[i][:, :, :, merged_sample_id:merged_sample_id+1])

                    # 将新样本与现有样本合并
                    merged_sample = self._merge_samples(existing_sample_to_merge,
                                                      new_train_sample,
                                                      self.prior_weights[merged_sample_id],
                                                      config.learning_rate,
                                                      config.sample_merge_type)

                    # 更新距离矩阵和格里姆矩阵
                    self._update_distance_matrix(gram_vector,
                                                new_train_sample_norm,
                                                merged_sample_id,
                                                -1,
                                                self.prior_weights[merged_sample_id, 0],
                                                config.learning_rate)

                    # 更新先验权重
                    self.prior_weights[closest_sample_to_new_sample] = self.prior_weights[closest_sample_to_new_sample] + config.learning_rate

                else:
                    #如果现有样本中的最小距离小于新样本到现有样本的最小距离，则合并现有样本并将新样本插入腾空位置。
                    self.prior_weights = self.prior_weights * ( 1 - config.learning_rate)

                    if self.prior_weights[closest_existing_sample2] > self.prior_weights[closest_existing_sample1]:
                        tmp = closest_existing_sample1
                        closest_existing_sample1 = closest_existing_sample2
                        closest_existing_sample2 = tmp

                    sample_to_merge1 = []
                    sample_to_merge2 = []
                    for i in range(num_feature_blocks):
                        sample_to_merge1.append(samplesf[i][:, :, :, closest_existing_sample1:closest_existing_sample1+1])
                        sample_to_merge2.append(samplesf[i][:, :, :, closest_existing_sample2:closest_existing_sample2+1])

                    merged_sample = self._merge_samples(sample_to_merge1,
                                                      sample_to_merge2,
                                                      self.prior_weights[closest_existing_sample1],
                                                      self.prior_weights[closest_existing_sample2],
                                                      config.sample_merge_type)

                    self._update_distance_matrix(gram_vector,
                                                new_train_sample_norm,
                                                closest_existing_sample1,
                                                closest_existing_sample2,
                                                self.prior_weights[closest_existing_sample1, 0],
                                                self.prior_weights[closest_existing_sample2, 0])

                    self.prior_weights[closest_existing_sample1] = self.prior_weights[closest_existing_sample1] + self.prior_weights[closest_existing_sample2]
                    self.prior_weights[closest_existing_sample2] = config.learning_rate

                    merged_sample_id = closest_existing_sample1
                    new_sample_id = closest_existing_sample2

                    new_sample = new_train_sample
        else:
            # 如果没有存满，则将新样本插入下一个空位置
            sample_position = num_training_samples
            self._update_distance_matrix(gram_vector, new_train_sample_norm,sample_position, -1, 0, 1)

            if sample_position == 0:
                self.prior_weights[sample_position] = 1
            else:
                self.prior_weights = self.prior_weights * (1 - config.learning_rate)
                self.prior_weights[sample_position] = config.learning_rate

            new_sample_id = sample_position
            new_sample = new_train_sample

        if abs(1 - np.sum(self.prior_weights)) > 1e-5:
            raise("weights not properly udpated")

        return merged_sample, new_sample, merged_sample_id, new_sample_id
