#参考了matlab代码ECO-master\implementation\training部分
import numpy as np
import warnings

from scipy.signal import convolve
import sys
sys.path.append("./eco")
from fourier_tools import symmetrize_filter
from config import hc_config
config = hc_config.HCConfig()



def diag_precond(hf, M_diag):
    ret = []
    for x, y in zip(hf, M_diag):
        ret.append([x_ / y_ for x_, y_ in zip(x, y)])
    return ret

def inner_product_filter(xf, yf):
    #计算两个筛选器之间的内积
    ip = 0
    for i in range(len(xf[0])):
        ip += 2 * np.vdot(xf[0][i].flatten(), yf[0][i].flatten()) - np.vdot(xf[0][i][:, -1, :].flatten(), yf[0][i][:, -1, :].flatten())
    return np.real(ip)

def inner_product_joint(xf, yf):
    #计算两个滤波器与投影矩阵的联合内积
       
    ip = 0
    for i in range(len(xf[0])):
        ip += 2 * np.vdot(xf[0][i].flatten(), yf[0][i].flatten()) - np.vdot(xf[0][i][:, -1, :].flatten(), yf[0][i][:, -1, :].flatten())
        ip += np.vdot(xf[1][i].flatten(), yf[1][i].flatten())
    return np.real(ip)

def lhs_operation(hf, samplesf, reg_filter, sample_weights):
    #共轭梯度的左边运算
    num_features = len(hf[0])
    filter_sz = np.zeros((num_features, 2), np.int32)
    for i in range(num_features):
        filter_sz[i, :] = np.array(hf[0][i].shape[:2])

    # 尺寸最大的特征块的索引
    k1 = np.argmax(filter_sz[:, 0])

    block_inds = list(range(0, num_features))
    block_inds.remove(k1)
    output_sz = np.array([hf[0][k1].shape[0], hf[0][k1].shape[1]*2-1])
    
    # 对所有要素和要素块求和
    sh = np.matmul(hf[0][k1].transpose(0, 1, 3, 2), samplesf[k1])
    pad_sz = [[]] * num_features
    for i in block_inds:
        pad_sz[i] = ((output_sz - np.array([hf[0][i].shape[0], hf[0][i].shape[1]*2-1])) / 2).astype(np.int32)
        sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] += np.matmul(hf[0][i].transpose(0, 1, 3, 2), samplesf[i])

    # 各样本的权重
    sh = sample_weights.reshape(1, 1, 1, -1) * sh

    # 乘转置
    hf_out = [[]] * num_features
    hf_out[k1] = np.matmul(np.conj(samplesf[k1]), sh.transpose(0, 1, 3, 2))
    for i in block_inds:
        hf_out[i] = np.matmul(np.conj(samplesf[i]), sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :].transpose(0, 1, 3, 2))

    # 计算与正则化项相对应的运算（用w的DFT去卷积每个特征维数）正则化部分：w^H w f
    for i in range(num_features):
        reg_pad = min(reg_filter[i].shape[1] - 1, hf[0][i].shape[1]-1)

        # 卷积
        hf_conv = np.concatenate([hf[0][i], np.conj(np.rot90(hf[0][i][:, -reg_pad-1:-1, :], 2))], axis=1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            # 第一次卷积
            hf_conv = convolve(hf_conv, reg_filter[i][:,:,np.newaxis,np.newaxis])

            # 最后一次卷积并相加
            hf_out[i] += convolve(hf_conv[:, :-reg_pad, :], reg_filter[i][:,:,np.newaxis,np.newaxis], 'valid')
    return [hf_out]


def lhs_operation_joint(hf, samplesf, reg_filter, init_samplef, XH, init_hf, proj_reg):
    #这是共轭梯度中的左侧操作 
 
    hf_out = [[[]] * len(hf[0]) for _ in range(len(hf))]

    P = [np.real(hf_) for hf_ in hf[1]]
    hf = hf[0]

    # 获得尺寸
    num_features = len(hf)
    filter_sz = np.zeros((num_features, 2), np.int32)
    for i in range(num_features):
        filter_sz[i, :] = np.array(hf[i].shape[:2])

    # 尺寸最大的特征块
    k1 = np.argmax(filter_sz[:, 0])

    block_inds = list(range(0, num_features))
    block_inds.remove(k1)
    output_sz = np.array([hf[k1].shape[0], hf[k1].shape[1]*2-1])

    # 分块矩阵乘法：正则化项 A^H A f

    # 加和所有特征的特征快
    sh = np.matmul(samplesf[k1].transpose(0, 1, 3, 2), hf[k1])
    pad_sz = [[]] * num_features
    for i in block_inds:
        pad_sz[i] = ((output_sz - np.array([hf[i].shape[0], hf[i].shape[1]*2-1])) / 2).astype(np.int32)
        sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] += np.matmul(samplesf[i].transpose(0, 1, 3, 2), hf[i])

    hf_out1 = [[]] * num_features
    hf_out1[k1] = np.matmul(np.conj(samplesf[k1]), sh)
    for i in block_inds:
        hf_out1[i] = np.matmul(np.conj(samplesf[i]), sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :])

    # #添加正则化部分
    for i in range(num_features):
        reg_pad = min(reg_filter[i].shape[1] - 1, hf[i].shape[1]-1)

        hf_conv = np.concatenate([hf[i], np.conj(np.rot90(hf[i][:, -reg_pad-1:-1, :], 2))], axis=1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            hf_conv = convolve(hf_conv, reg_filter[i][:, :, np.newaxis, np.newaxis])
            hf_out1[i] += convolve(hf_conv[:, :-reg_pad, :], reg_filter[i][:, :, np.newaxis, np.newaxis], 'valid')
        
    # 投影矩阵
    BP_list = [np.matmul(init_hf_.transpose(0, 1, 3, 2), np.matmul(P_.T, init_samplef_))
            for init_samplef_, P_, init_hf_ in zip(init_samplef, P, init_hf)]
    BP = BP_list[k1]
    for i in block_inds:
        BP[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] += BP_list[i]

    hf_out[0][k1] = hf_out1[k1] + (np.conj(samplesf[k1]) * BP)

    fBP = [[]] * num_features
    fBP[k1] = (np.conj(init_hf[k1]) * BP).reshape((-1, init_hf[k1].shape[2]))

    # 计算proj矩阵
    shBP = [[]] * num_features
    shBP[k1] = (np.conj(init_hf[k1]) * sh).reshape((-1, init_hf[k1].shape[2]))

    for i in block_inds:
        hf_out[0][i] = hf_out1[i] + (BP[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :] * np.conj(samplesf[i]))

        fBP[i] = (np.conj(init_hf[i]) * BP[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :]).reshape((-1, init_hf[i].shape[2]))
        shBP[i] = (np.conj(init_hf[i]) * sh[pad_sz[i][0]:output_sz[0]-pad_sz[i][0], pad_sz[i][1]:, :, :]).reshape((-1, init_hf[i].shape[2]))

    for i in range(num_features):
        fi = hf[i].shape[0] * (hf[i].shape[1] - 1) 
        hf_out2 = 2 * np.real(XH[i].dot(fBP[i]) - XH[i][:, fi:].dot(fBP[i][fi:, :])) + proj_reg * P[i]
        hf_out[1][i] = hf_out2 + 2 * np.real(XH[i].dot(shBP[i]) - XH[i][:, fi:].dot(shBP[i][fi:, :]))
    return hf_out


def preconditioned_conjugate_gradient(A, b, opts, M1, M2, ip,x0, state=None):
    #执行预处理共轭梯度
    maxit  = int(opts['maxit'])

    if 'init_forget_factor' not in opts:
        opts['init_forget_factor'] = 1

    x = x0
    p = None
    rho = 1
    r_prev = None

    if state is None:
        state = {}
    else:
        if opts['init_forget_factor'] > 0:
            if 'p' in state:
                p = state['p']
            if 'rho' in state:
                rho = state['rho'] / opts['init_forget_factor']
            if 'r_prev' in state and not opts['CG_use_FR']:
                r_prev = state['r_prev']
    state['flag'] = 1

    r = []
    for z, y in zip(b, A(x)):
        r.append([z_- y_ for z_, y_ in zip(z, y)])

    resvec = []
    relres = []
    # 开始迭代
    for ii in range(maxit):
        if M1 is not None:
            y = M1(r)
        else:
            y = r

        if M2 is not None:
            z = M2(y)
        else:
            z = y

        rho1 = rho
        rho = ip(r, z)
        if rho == 0 or np.isinf(rho):
            state['flag'] = 4
            break

        if ii == 0 and p is None:
            p = z
        else:
            if opts['CG_use_FR']:
                #  Fletcher-Reeves法
                beta = rho / rho1
            else:
                # Polak-Ribiere法
                rho2 = ip(r_prev, z)
                beta = (rho - rho2) / rho1
            if beta == 0 or np.isinf(beta):
                state['flag'] = 4
                break
            beta = max(0, beta)
            tmp = []
            for zz, pp in zip(z, p):
                tmp.append([zz_ + beta * pp_ for zz_, pp_ in zip(zz, pp)])
            p = tmp

        q = A(p)
        pq = ip(p, q)
        if pq <= 0 or np.isinf(pq):
            state['flag'] = 4
            break
        else:
            if opts['CG_standard_alpha']:
                alpha = rho / pq
            else:
                alpha = ip(p, r) / pq
        if np.isinf(alpha):
            state['flag'] = 4
        if not opts['CG_use_FR']:
            r_prev = r

        # 新的跌倒迭代
        tmp = []
        for xx, pp in zip(x, p):
            tmp.append([xx_ + alpha * pp_ for xx_, pp_ in zip(xx, pp)])
        x = tmp
        if ii < maxit:
            tmp = []
            for rr, qq in zip(r, q):
                tmp.append([rr_ - alpha * qq_ for rr_, qq_ in zip(rr, qq)])
            r = tmp

    # 保存状态量
    state['p'] = p
    state['rho'] = rho
    if not opts['CG_use_FR']:
        state['r_prev'] = r_prev
    return x, resvec, state

def train_filter(hf, samplesf, yf, reg_filter, sample_weights, sample_energy, reg_energy, CG_opts, CG_state):

    # 构造右侧向量
    rhs_samplef = [np.matmul(xf, sample_weights) for xf in samplesf]
    rhs_samplef = [(np.conj(xf) * yf[:,:,np.newaxis,np.newaxis])
            for xf, yf in zip(rhs_samplef, yf)]

    # 构造前置条件
    diag_M = [(1 - config.precond_reg_param) * (config.precond_data_param * m + (1-config.precond_data_param)*np.mean(m, 2, keepdims=True))+ \
              config.precond_reg_param * reg_energy_ for m, reg_energy_ in zip(sample_energy, reg_energy)]
    hf, _, CG_state = preconditioned_conjugate_gradient(
            lambda x: lhs_operation(x, samplesf, reg_filter, sample_weights), # A
            [rhs_samplef],                                                    # b
            CG_opts,                                                          # opts
            lambda x: diag_precond(x, [diag_M]),                              # M1
            None,                                                             # M2
            inner_product_filter,
            [hf],
            CG_state)
    
    return hf[0], CG_state #res_norms, CG_state

def train_joint(hf, proj_matrix, xlf, yf, reg_filter, sample_energy, reg_energy, proj_energy, init_CG_opts):
    #滤波核和投影矩阵的初始化

    lf_ind = [x.shape[0] * (x.shape[1]-1) for x in hf[0]]
    init_samplef = xlf
    init_samplef_H = [np.conj(x.reshape((-1, x.shape[2]))).T for x in init_samplef]
    diag_M = [[], []]
    diag_M[0] = [(1 - config.precond_reg_param) * (config.precond_data_param * m + (1-config.precond_data_param)*np.mean(m, 2, keepdims=True))+ \
              config.precond_reg_param * reg_energy_ for m, reg_energy_ in zip(sample_energy, reg_energy)]
    diag_M[1] = [config.precond_proj_param * (m + config.projection_reg) for m in proj_energy]

    rhs_samplef = [[]] * len(hf[0])
    for iter_ in range(config.init_GN_iter):
        init_samplef_proj = [np.matmul(P.T, x) for x, P in zip(init_samplef, proj_matrix)]
        init_hf = hf[0]

        # 构造右边的向量
        rhs_samplef[0] = [np.conj(xf) * yf_[:,:,np.newaxis,np.newaxis] for xf, yf_ in zip(init_samplef_proj, yf)]

        # 构造右边向量
        fyf = [np.reshape(np.conj(f) * yf_[:,:,np.newaxis,np.newaxis], (-1, f.shape[2])) for f, yf_ in zip(hf[0], yf)] # matlab reshape
        rhs_samplef[1] =[ 2 * np.real(XH.dot(fyf_) - XH[:, fi:].dot(fyf_[fi:, :])) - config.projection_reg * P
            for P, XH, fyf_, fi in zip(proj_matrix, init_samplef_H, fyf, lf_ind)]

        # 初始化投影矩阵为0
        hf[1] = [np.zeros_like(P) for P in proj_matrix]

        # 共轭梯度法
        hf, _, _ = preconditioned_conjugate_gradient(
                lambda x: lhs_operation_joint(x, init_samplef_proj, reg_filter, init_samplef, init_samplef_H, init_hf, config.projection_reg), # A
                rhs_samplef,                                                                                                                   # b
                init_CG_opts,                                                                                                                  # opts
                lambda x: diag_precond(x, diag_M),                                                                                             # M1
                None,                                                                                                                          # M2
                inner_product_joint,
                hf)

        # 让滤波核对称
        hf[0] = symmetrize_filter(hf[0])

        # 计算投影矩阵
        proj_matrix = [x + y for x, y in zip(proj_matrix, hf[1])]

    # 提取滤波核
    hf = hf[0]
    return hf, proj_matrix
