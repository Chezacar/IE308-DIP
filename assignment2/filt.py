import numpy as np
import numpy.linalg as la
import math
import PIL.Image as im
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import fftpack as fft
#分块哈尔变换函数

#变换基本操作
def haar_operation(array):
    block_size = int(array.shape[0] / 2)
    result_array = np.zeros(2 * block_size)
    for j in range(0,block_size):
        result_array[j] = (array[2*j] + array[2*j + 1]) / 2
        result_array[j + block_size] = (array[2*j] - array[2*j+1]) / 2
    return result_array

#整体操作函数
def haar_transform_1d(array):
    block_size = array.shape[0]
    cal_times = int(math.log(block_size,2))
    result = np.zeros(block_size)
    for i in range(cal_times,0,-1):
        cal_lenth = pow(2,i)#单次变换序列长度
        """ cal_round = int(pow(2,block_size/cal_lenth-1))#该轮变换调用次数
        for j in range(0,cal_round): """
        result[0:cal_lenth] = haar_operation(array[0:cal_lenth])
    return result

#将haar变换拓展至二维
def haar_transform_2d(matrix):
    (x,y) = matrix.shape
    result = np.zeros((x,y))
    result += matrix
    cal_round = int(math.log(x,2))
    for step in range(cal_round,0,-1):
        length = pow(2,step)
        for i in range(0,length):
            result[i,0:length] = haar_operation(result[i,0:length])
        for j in range(0,length):    
            result[0:length,j] = haar_operation(result[0:length,j])
    return result

#haar反变换操作
def haar_inv_operation(array):
    block_size = array.shape[0]
    length = int(block_size / 2)
    cal_unit = int(block_size / 2)
    result_array = np.zeros(block_size)
    for i in range(cal_unit):
        result_array[2*i] = array[i] + array[i+length]
        result_array[2*i+1] = array[i] - array[i+length]
    return result_array

#二维haar反变换
def haar_inv_2d(matrix):
    (x,y) = matrix.shape
    result = np.zeros((x,y))
    result += matrix
    cal_round = int(math.log(x,2))
    for step in range(0,cal_round):
        length = pow(2,step+1)
        for j in range(0,length):    
            result[0:length,length-j-1] = haar_inv_operation(result[0:length,length-j-1])
        for i in range(0,length):
            result[length-i-1,0:length] = haar_inv_operation(result[length-i-1,0:length])
    return result

#三维哈尔变换正变换过程
def haar_transform_3d(matrix_3d):
    (x,y,z) = matrix_3d.shape
    result = np.zeros((x,y,z))
    result += matrix_3d
    cal_round = int(math.log(x,2))
    for step in range(0,cal_round):
        length = pow(2,cal_round-step)
        for i in range(0,x):
            for j in range(0,x):
                result[i,j,0:length] = haar_operation(result[i,j,0:length])
        for i in range(0,x):
            for j in range(0,x):
                result[i,0:length,j] = haar_operation(result[i,0:length,j])
        for i in range(0,x):
            for j in range(0,x):
                result[0:length,i,j] = haar_operation(result[0:length,i,j])
    return result

#三维哈尔变换反变换过程                               
def haar_inv_3d(matrix_3d):#周一19:55写到这里
    (x,y,z) = matrix_3d.shape
    result = np.zeros((x,y,z))
    result += matrix_3d
    cal_round = int(math.log(x,2))
    for step in range(0,cal_round):
        length = pow(2,step+1)
        for j in range(x-1,0,-1):
            for i in range(x-1,0,-1):
                result[i,j,0:length] = haar_inv_operation(result[i,j,0:length])
        for j in range(x-1,0,-1):
            for i in range(x-1,0,-1):
                result[i,0:length,j] = haar_inv_operation(result[i,0:length,j])
        for j in range(x-1,0,-1):
            for i in range(x-1,0,-1):
                result[0:length,i,j] = haar_inv_operation(result[0:length,i,j])
    return result


#阈值收缩
def Washrink(matrix,wavelet):
    result = np.zeros(matrix.shape)
    result = np.where(abs(matrix) < wavelet,0,matrix)
    return result

#二维哈尔阈值收缩滤波
def haar_filter_2d(pic,kernel_size,wavelet,stride):
    x_rounds = pic.shape[0] - kernel_size + 1
    y_rounds = pic.shape[1] - kernel_size + 1
    record_matrix = np.zeros((pic.shape[0],pic.shape[1],2))
    for i in range(0,x_rounds,stride):
        print(i,'/',pic.shape[0])
        for j in range(0,y_rounds,stride):
            haar_matrix = haar_transform_2d(pic[i:i+kernel_size,j:j+kernel_size])
            haar_matrix = Washrink(haar_matrix,wavelet)
            matrix = haar_inv_2d(haar_matrix)
            record_matrix[i:i+kernel_size,j:j+kernel_size,0] += matrix
            record_matrix[i:i+kernel_size,j:j+kernel_size,1] += np.ones((kernel_size,kernel_size))
    record_matrix[:,:,0] = record_matrix[:,:,0] / record_matrix[:,:,1]
    return record_matrix[:,:,0]

#3维小波阈值收缩的过程 先三维哈尔变换 在阈值收缩 再三维哈尔反变换
def haar_filter_3d(matrix_3d,wavelet):
    '''result_1 = haar_transform_3d(matrix_3d)
    result_2 = Washrink(result_1,0.05)
    result = haar_inv_3d(result_2)
    '''
    result_1 = np.fft.fftn(matrix_3d)
    result_2 = Washrink(result_1,wavelet)
    result = np.fft.ifftn(result_2)
    result = result.real
    return result

#分块距离匹配,这里用二范数表示两block之间距离：        
def dist_block(block1,block2):
    (x,y) = block1.shape
    dist = la.norm(block1 - block2) / (x * y)
    return dist

#分块函数
'''
def get_block(pic,x,y):
    num_x = pic.shape[0] / x
    num_y = pic.shape[1] / y
    num = num_x * num_y
    result = np.zeros((num_x,num_y,(x,y)))
    for i in range(num_x):
        for j in range(num_y):
            result[i,j,:] = pic[i*x:(i+1)*x,j*y,(j+1)*y]
    return result
'''     
def decide_district(i,j,pic,radius):
    (y,x) = pic.shape
    district = np.zeros(4)
    if (i + radius < x) and (i >= radius):
        district[0] = i - radius
        district[2] = i + radius
    if (j + radius < x) and (j >= radius):
        district[1] = j - radius
        district[3] = j + radius
    if (i + radius >= x):
        district[0] = x - 1 - (2 * radius)
        district[2] = x - 1
    if (i - radius < 0):
        district[0] = 0
        district[2] = 2 * radius
    if (j + radius >= y):
        district[1] = y - 1 - (2*radius)
        district[3] = y - 1
    if (j - radius < 0):
        district[1] = 0
        district[3] = 2 * radius
    return district

#寻找相似块
def search_sim_haar(block,pic,kernel_size,n):
    x_rounds = pic.shape[0] - kernel_size + 1
    y_rounds = pic.shape[1] - kernel_size + 1
    record_matrix = np.zeros((n,3))#三维分别记录block的左上角坐标和dist值
    block_haar = haar_transform_2d(block)
    for t in range(n):
        record_matrix[t,2] = 99999 + t
    for i in range(x_rounds):
        for j in range(y_rounds):
            block_test = haar_transform_2d(pic[i:i+kernel_size,j:j+kernel_size])
            #print(block.shape,block_test.shape)
            if dist_block(block_haar,block_test) < np.max(record_matrix[:,2]):
                maxindex = np.argmax(record_matrix[:,2])
                record_matrix[maxindex,0] = i
                record_matrix[maxindex,1] = j
                record_matrix[maxindex,2] = dist_block(block_haar,block_test)
    return record_matrix


    
def search_sim(block,pic,kernel_size,n):
    x_rounds = pic.shape[0] - kernel_size + 1
    y_rounds = pic.shape[1] - kernel_size + 1
    record_matrix = np.zeros((n,3))#三维分别记录block的左上角坐标和dist值
    for t in range(n):
        record_matrix[t,2] = 99999 + t
    for i in range(x_rounds):
        for j in range(y_rounds):
            block_test = pic[i:i+kernel_size,j:j+kernel_size]
            #print(block.shape,block_test.shape)
            if dist_block(block,block_test) < np.max(record_matrix[:,2]):
                maxindex = np.argmax(record_matrix[:,2])
                record_matrix[maxindex,0] = i
                record_matrix[maxindex,1] = j
                record_matrix[maxindex,2] = dist_block(block,block_test)
    return record_matrix

#像素融合函数，听起来很像今年iPhone的新功能“Deep Fusion”
#我们pixel_fusion加权使用的是softmax方法
def pixel_fusion(block_filter,weight):
    softmax_weight = softmax(weight)
    (x,y,z) = block_filter.shape
    result = np.zeros((x,y))
    for i in range(z):
        result += block_filter[:,:,i] * softmax_weight[i]
    return result
#三维匹配滤波  将匹配和滤波、像素融合三个过程联系起来
def filter_match(pic,kernel_size,num,method,wavelet,pic_first =0,K = 0,stride = 1):
    (x,y) = pic.shape
    record_matrix = np.zeros((x,y,2))
    result_pic = np.zeros((x,y))
    result_pic += pic
    x_ronuds = x - kernel_size + 1
    y_rounds = y - kernel_size + 1
    for i in range(0,x_ronuds,stride):
        for j in range(0,y_rounds,stride):
            print('line:',i + 1,'/',x_ronuds,'row:',j + 1,'/',y_rounds)
            block_position = decide_district(i,j,pic,kernel_size)
            pic_block = pic[i:i+kernel_size,j:j+kernel_size]
            search_block = pic[int(block_position[0]):int(block_position[2]),
                int(block_position[1]):int(block_position[3])]
            if method == 1:
                block_sim = search_sim_haar(pic_block,search_block,kernel_size,num)
                block_filter = first_filter_3d(search_block,block_sim,kernel_size,wavelet)
            else:
                block_sim = search_sim(pic_block,search_block,kernel_size,num)
                block_filter = second_filter_3d(pic,pic_first,block_sim,kernel_size,K)
            record_matrix[i:i+kernel_size,j:j+kernel_size,0] += pixel_fusion(block_filter,block_sim[:,2])
            record_matrix[i:i+kernel_size,j:j+kernel_size,1] += np.ones((kernel_size,kernel_size))
    result_pic = record_matrix[:,:,0] / record_matrix[:,:,1]
    return result_pic

#三维滤波具体操作（匹配完成后的三维滤波过程）第一阶段是三维哈尔小波变换阈值收缩
def first_filter_3d(pic,matrix_sim,kernel_size,wavelet):#pic为待去噪图片，matrix为相似块位置
    n = matrix_sim.shape[0]
    matrix_3d = np.zeros((kernel_size,kernel_size,kernel_size))
    for i in range(n):
        [x,y] = matrix_sim[i,0:2]
        x = int(x)
        y = int(y)
        matrix_3d[i,:,:] = pic[x:x+kernel_size,y:y+kernel_size]
    result = haar_filter_3d(matrix_3d,wavelet)
    return result

def second_filter_3d(pic,pic_first,matrix_sim,kernel_size,K):#pic为待去噪图片，pic_first是第一阶段预去噪后的结果，matrix——sim为相似块位置
    n = matrix_sim.shape[0]
    matrix_3d = np.zeros((kernel_size,kernel_size,kernel_size))
    matrix_3d_first = np.zeros((kernel_size,kernel_size,kernel_size))
    for i in range(n):
        [x,y] = matrix_sim[i,0:2]
        x = int(x)
        y = int(y)
        matrix_3d[i,:,:] = pic[x:x+kernel_size,y:y+kernel_size]
        matrix_3d_first[i,:,:] = pic_first[x:x+kernel_size,y:y+kernel_size]
    #result = wiener_filter_3d(matrix_3d,matrix_3d_first,K).real
    result = wiener(matrix_3d_first,matrix_3d,20,0.01)
    return result
#维纳滤波
#调用了二位快速傅立叶变换的计算包，这个也不能用的话真的顶不住了
def wiener_filter_3d(matrix,matrix_first,K):#matrix：输入函数  h：退化函数  K：参数K
    (x,y,z) = matrix.shape
    s = [x,y,z]
    matrix_fft = fft.fftn(matrix)
    #print(matrix.shape)
    #print(matrix_first.shape)
    h_fft = matrix_fft / fft.fftn(matrix_first)
    input_matrix = np.zeros((x,y,z)) + matrix
    input_matrix_fft = fft.fftn(input_matrix)
    h_abs_square = h_fft * np.conj(h_fft)
    xishu = h_abs_square / (h_abs_square + K)
    result = fft.ifftn(h_abs_square / xishu / h_fft * matrix_fft)
    return result

#用softmax来确定融合时的权值
def softmax(vector):
    size = vector.shape[0]
    vector_exp = np.exp(-vector)
    softmax_sum = np.sum(vector_exp)
    vector_pro = vector_exp / softmax_sum
    return vector_pro

def noise_pic(pic,mean,sigma):
    (x,y) = pic.shape
    noise = np.random.normal(loc = mean,scale = sigma,size = (x,y))
    pic_n = pic + noise
    return pic_n

def wiener(input1,input2,eps,K=0.01):        #维纳滤波，K=0.01
    input1_fft = fft.fftn(input1)
    input2_fft = fft.fftn(input2)
    PSF = fft.ifftn(input2_fft / input1_fft)
    PSF = np.abs(PSF) / np.abs(PSF.sum())
    PSF_fft=fft.fftn(PSF) +eps
    PSF_fft_1=np.conj(PSF_fft) /(np.abs(PSF_fft)**2 + K)
    result=fft.ifftn(input1_fft * PSF_fft_1)
    result=np.abs(fft.fftshift(result))
    result = result * 255 / result.max()
    return result


if __name__ == '__main__':
    pic = im.open('gakki1.jpeg')
    #pic = pic.resize((52,32))
    #pic.show()
    pic_L = pic.convert('L')
    pic_L = np.array(pic_L)
    ''' 
    pic_L = np.array(pic_L)
    pic_L = pic_L.reshape(512,320)
    pic_L = im.fromarray(pic_L) 
    ''' 
    #pic_L.show()
    pic_noise = noise_pic(pic_L,0,20)
    pic_noise_show = im.fromarray(pic_noise)
    #pic_noise_show.show()
    pic_noise_show.convert('RGB').save('pic_noise_show.png',quality = 95)
    
    #pic_haar_show.show()
    #for i in [5,3,1,0.5,0.3,0.1,0.05,0.03]:
    i = 0
    """ a = np.array([[2.0,2.0,2.0,4.0],[1.0,3.0,4.0,5.0],[3.0,6.0,9.0,1.0],[2.0,4.0,6.0,9.0]])
    a_haar = haar_transform_2d(a)
    a_haar_inv = haar_inv_2d(a_haar) """
    print(i)
    pic_haar = haar_filter_2d(pic_noise,8,20,2)
    pic_haar_show = im.fromarray(pic_haar)
    filename = 'pic_haar_show' + str(i) + '.png'
    pic_first_step = filter_match(pic_haar,8,8,1,i,stride = 2)
    pic = im.fromarray(pic_first_step)
    name = 'pic_final_step' + '.png'
    pic_second_step = filter_match(pic_haar,8,8,2,i,pic_first_step,0.01,2)
    print(pic_first_step)
    print(pic_second_step)
    pic_final = im.fromarray(pic_second_step)
    final_name = 'pic_final' + '.png'
    pic_final.convert('RGB').save(final_name,quality = 95)