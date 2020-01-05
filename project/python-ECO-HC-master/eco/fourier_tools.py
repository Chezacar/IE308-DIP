#参照matlab代码中ECO-master\fourier_tools文件夹中函数得出
import numpy as np
import sys
sys.path.append("/Users/chezacar/Desktop/图像处理与内容分析/大作业/pyECO-master/eco/config")
from config import hc_config
config = hc_config.HCConfig()

np.seterr(divide='ignore', invalid='ignore')

def fft2(x): #编写所需的二维fft变换
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)

def ifft2(x): #编写所需的二维fft逆变换
    return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)

def cfft2(x): #利用实数傅里叶谱的共轭对称性减少fft运算量
    in_shape = x.shape
    if in_shape[0] % 2 == 1 and in_shape[1] % 2 == 1:
        xf = np.fft.fftshift(np.fft.fftshift(fft2(x), 0), 1).astype(np.complex64)
    else:
        out_shape = list(in_shape)
        out_shape[0] =  in_shape[0] + (in_shape[0] + 1) % 2
        out_shape[1] =  in_shape[1] + (in_shape[1] + 1) % 2
        out_shape = tuple(out_shape)
        xf = np.zeros(out_shape, dtype=np.complex64)
        xf[:in_shape[0], :in_shape[1]] = np.fft.fftshift(np.fft.fftshift(fft2(x), 0), 1).astype(np.complex64)
        if out_shape[0] != in_shape[0]:
            xf[-1,:] = np.conj(xf[0,::-1])
        if out_shape[1] != in_shape[1]:
            xf[:,-1] = np.conj(xf[::-1,0])
    return xf

def cifft2(xf): #编写cfft所需的反变换函数
    x = np.real(ifft2(np.fft.ifftshift(np.fft.ifftshift(xf, 0),1))).astype(np.float32)
    return x

def compact_fourier_coeff(xf): #计算紧凑傅里叶系数
    if isinstance(xf, list):
        return [x[:, :(x.shape[1]+1)//2, :] for x in xf]
    else:
        return xf[:, :(xf.shape[1]+1)//2, :]

def cubic_spline_fourier(f, a): #计算三次样条插值的计算核对应的频域值
    bf = - ( - 12 * a + 12 * np.exp( - np.pi * f * 2j) + 12 * np.exp(np.pi * f * 2j) + 6 * a * np.exp(-np.pi * f * 4j) + \
        6 * a * np.exp(np.pi * f * 4j) + f * (np.pi * np.exp(-np.pi*f*2j)*12j) - f * (np.pi * np.exp(np.pi * f * 2j) * 12j) + \
        a*f*(np.pi*np.exp(-np.pi*f*2j)*16j) - a * f * (np.pi*np.exp(np.pi*f*2j)*16j) + \
        a*f*(np.pi*np.exp(-np.pi*f*4j)*4j) - a * f * (np.pi*np.exp(np.pi*f*4j)*4j)-24)
    bf /= (16 * f ** 4 * np.pi ** 4)
    bf[f == 0] = 1
    return bf

def full_fourier_coeff(xf): #根据共轭对称性补全频谱系数，进行频谱重构
    xf = [np.concatenate([xf_, np.conj(np.rot90(xf_[:, :-1,:], 2))], axis=1) for xf_ in xf]
    return xf

def interpolate_dft(xf, interp1_fs, interp2_fs): #在频域对于目标图像进行插值
    return [xf_ * interp1_fs_ * interp2_fs_
            for xf_, interp1_fs_, interp2_fs_ in zip(xf, interp1_fs, interp2_fs)]


def resize_dft(inputdft, desired_len): #将一维的DFT序列转化为指定长度
    input_len = len(inputdft)
    minsz = min(input_len, desired_len)

    scaling = desired_len / input_len

    resize_dft = np.zeros(desired_len, dtype=inputdft.dtype)

    mids = int(np.ceil(minsz / 2))
    mide = int(np.floor((minsz - 1) / 2))

    resize_dft[:mids] = scaling * inputdft[:mids]
    resize_dft[-mide:] = scaling * inputdft[-mide:]
    return resize_dft

def sample_fs(xf, grid_sz=None): #进行频域点乘操作
    sz = xf.shape[:2]
    if grid_sz is None or sz == grid_sz:
        x = sz[0] * sz[1] * cifft2(xf)
    else:
        sz = np.array(sz)
        grid_sz = np.array(grid_sz)
        if np.any(grid_sz < sz):
            raise("The grid size must be larger than or equal to the siganl size")
        tot_pad = grid_sz - sz
        pad_sz = np.ceil(tot_pad / 2).astype(np.int32)
        xf_pad = np.pad(xf, tuple(pad_sz), 'constant')
        if np.any(tot_pad % 2 == 1):
            xf_pad = xf_pad[:xf_pad.shape[0]-(tot_pad[0] % 2), :xf_pad.shape[1]-(tot_pad[1] % 2)]
        x = grid_sz[0] * grid_sz[1] * cifft2(xf_pad)
    return x

def shift_sample(xf, shift, kx, ky): #对于样本块进行位移操作
    shift_exp_y = [np.exp(1j * shift[0] * ky_).astype(np.complex64) for ky_ in ky]
    shift_exp_x = [np.exp(1j * shift[1] * kx_).astype(np.complex64) for kx_ in kx]
    xf = [xf_ * sy_.reshape(-1, 1, 1, 1) * sx_.reshape((1, -1, 1, 1))
            for xf_, sy_, sx_ in zip(xf, shift_exp_y, shift_exp_x)]
    return xf

def symmetrize_filter(hf): #确保滤波核是hermetian对称的
    for i in range(len(hf)):
        dc_ind = int((hf[i].shape[0]+1) / 2)
        hf[i][dc_ind:, -1, :] = np.conj(np.flipud(hf[i][:dc_ind-1, -1, :]))
    return hf
