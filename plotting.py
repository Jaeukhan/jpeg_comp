import numpy as np
from matplotlib import pyplot as plt
scale = [5, 10, 15, 20, 25, 30, 50, 70, 90]
subjective = [1, 20, 40, 60, 65, 65, 80, 85, 95]
psnr = [29.192916708842247, 30.205587835550073, 30.627909505339346, 30.94528916947112, 31.180281630535085, 31.371476542548926, 31.795279468229754, 32.193013322795515, 33.03134521189871]
ssim = [0.5588492874968579, 0.6448255972784948, 0.6858115179571465, 0.7120633102089353,  0.7306146248764276, 0.7443793923591271, 0.7785998702133965, 0.8061363920941724, 0.8502894964315343]

x = [8, 16, 32, 64]
psnr2 = [0.7142078542697551, 0.6453874572810242,0.5930733259723275 ,0.6253800404460719]
ssim2 = [30.897093365040647, 30.367267451206693, 29.71500619233567, 29.847895435016458]

plt.subplot(121)
plt.scatter(x, psnr2)
plt.plot(x, psnr2)
plt.xlabel('Q')
plt.ylabel('psnr')

plt.subplot(122)
plt.scatter(x, ssim2)
plt.plot(x, ssim2)
plt.xlabel('Q')
plt.ylabel('ssim')

# plt.subplot(131)
# plt.scatter(scale, psnr)
# plt.plot(scale, psnr)
# plt.xlabel('scale')
# plt.ylabel('psnr')
#
# plt.subplot(132)
# plt.scatter(scale, ssim)
# plt.plot(scale, ssim)
# plt.xlabel('scale')
# plt.ylabel('ssim')
#
# plt.subplot(133)
# plt.scatter(scale, subjective)
# plt.plot(scale, subjective)
# plt.xlabel('scale')
# plt.ylabel('subjective')

plt.show()