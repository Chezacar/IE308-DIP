import PIL.Image as im
import numpy as np
import cv2
pic = im.open('gakki1.jpeg')
pic_L = pic.convert('L')
pic_L = np.array(pic_L)
(x,y) = pic_L.shape
result = np.zeros((x,y,3))
for i in range(x):
    for j in range(y):
        if pic_L[i,j]>=0 and pic_L[i,j]<64:
            result[i,j,0] = 0
            result[i,j,1] = 4 * pic_L[i,j]
            result[i,j,2] = 255
        elif pic_L[i,j]>=64 and pic_L[i,j]<128:
            result[i,j,0] = 0
            result[i,j,1] = 255
            result[i,j,2] = 511 - 4 * pic_L[i,j]
        elif pic_L[i,j]>=128 and pic_L[i,j]<192:
            result[i,j,0] = 4 * pic_L[i,j] - 511
            result[i,j,1] = 255
            result[i,j,2] = 0
        elif pic_L[i,j]>=192 and pic_L[i,j]<=256:
            result[i,j,0] = 255
            result[i,j,1] = 1023 - 4 * pic_L[i,j]
            result[i,j,2] = 0
cv2.imwrite('color.jpg',result)