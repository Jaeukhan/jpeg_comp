import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy.fftpack import dct, idct
import utills


def showimg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def qualityfactor(matrix, Q=50):
    scalefactor = 1
    if Q < 50:
        scalefactor = 5000 / Q
    elif Q >= 50:
        scalefactor = 200 - 2 * Q

    x = np.floor((matrix * scalefactor + 50) / 100)
    if scalefactor == 0:
        x += 1
    # print(x)

    return x




def encode_zigzag(matrix):  # matrix shape: (4096,8,8)
    bitstream = []
    for i, block in enumerate(matrix):  # (8, 8)
        new_block = block.reshape([64])[utills.zigzag_order].tolist()
        zero_count = (new_block.count(0))-1
        bitstream.append(new_block[:-1*zero_count])
    return bitstream

def decode_zigzag(matrix):
    matrix = np.array(matrix)
    # print(matrix[0])
    zigord1 = np.zeros((matrix.shape[0], 64))
    zigord = np.zeros((matrix.shape[0], 64))
    for i, block in enumerate(matrix):
        for j in range(len(block)):
            zigord1[i][j] = block[j]
        zigord[i][utills.zigzag_order]= zigord1[i]
        # print(zigord[i])
        # print(zigord[i].reshape(8,8))
    zigord = zigord.reshape(matrix.shape[0], 8, 8)
    print("received& convert zigzag")
    print(zigord[0])
    return zigord


def quantization(blocks, type, Q):
    Q_y = qualityfactor(utills.Q_y, Q)
    Q_c = qualityfactor(utills.Q_c, Q)
    quant_block = np.zeros_like(blocks)
    if (type == 'y'):
        for i in range(len(blocks)):
            quant_block[i] = np.divide(blocks[i], Q_y).round().astype(np.float64)
    elif (type == 'c'):
        for i in range(len(blocks)):
            quant_block[i] = np.divide(blocks[i], Q_c).round().astype(np.float64)
    return quant_block


def dequantization(blocks, type, Q):
    Q_y = qualityfactor(utills.Q_y, Q)
    Q_c = qualityfactor(utills.Q_c, Q)
    quant_block = np.zeros_like(blocks)
    if (type == 'y'):
        for i in range(len(blocks)):
            quant_block[i] = np.multiply(blocks[i], Q_y).round().astype(np.float64)
    elif (type == 'c'):
        for i in range(len(blocks)):
            quant_block[i] = np.multiply(blocks[i], Q_c).round().astype(np.float64)
    return quant_block


def transform_to_block(img, blocksize=8):  # block size에 따른 Q_table interpolation과 downsampling.
    img_w, img_h = img.shape
    blocks = []
    for i in range(0, img_w, blocksize):
        for j in range(0, img_h, blocksize):
            blocks.append(img[i:i + blocksize, j:j + blocksize])
    blocks = np.array(blocks)
    print('8x8 block')
    print(blocks[0])
    print('-128: ')
    blocks = blocks - 128
    print(blocks[0])
    return blocks

def reconstruct_from_blocks(blocks):
    total_lines = []
    N_blocks    = int(len(blocks)**0.5)
    # print("N", N_blocks)
    # print(len(blocks))
    for n in range(0, len(blocks) - N_blocks + 1, N_blocks):
        res = np.concatenate(blocks[n : n + N_blocks], axis=1)
        # print(res.shape)
        total_lines.append(res)
        # print(np.array(total_lines).shape)
    blocks = np.concatenate(total_lines) + 128
    return blocks


def dct2d(imgblocks):
    dctblock = np.zeros_like(imgblocks)
    for i in range(len(imgblocks)):
        dctblock[i] = dct(dct(imgblocks[i], axis=0, norm='ortho'), axis=1, norm='ortho')
    return dctblock


def idct_2d(blocks):
    idctblock = np.zeros_like(blocks)
    for i in range(len(blocks)):
        idctblock[i] = idct(idct(blocks[i], axis=0, norm='ortho'), axis=1, norm='ortho')
    return idctblock


def chromasubsampling(chroma, h, w):
    sub = np.zeros([int(h / 2), int(w / 2)])
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            # if i % 2 == 0 and j % 2 == 0:
            sub[int(i/2), int(j/2)] = chroma[i, j]

    return sub


def chroma_interpolation(matrix):
    src = matrix.astype(np.uint8)
    matrix = cv2.resize(src, (len(matrix[0])*2, len(matrix[1])*2), interpolation=cv2.INTER_CUBIC)
    return matrix


def encoding(img, Q, N):
    # showimg(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #show original image
    h, w, c = img.shape

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = img_ycrcb[:, :, 0], img_ycrcb[:, :, 1], img_ycrcb[:, :, 2]

    y_z = np.zeros([h, w])
    for i in range(h):
        for j in range(w):
            y_z[i, j] = y[i, j]
    subcb = chromasubsampling(cb, h, w)
    subcr = chromasubsampling(cr, h, w)

    y_blocks = transform_to_block(y_z)
    cb_blocks = transform_to_block(subcb)
    cr_blocks = transform_to_block(subcr)

    y_blocks = dct2d(y_blocks)
    cb_blocks = dct2d(cb_blocks)
    cr_blocks = dct2d(cr_blocks)
    print("DCT :")
    print(y_blocks[0])

    y_qnt = quantization(y_blocks, 'y', Q)  # (4096, 8,8)
    cb_qnt = quantization(cb_blocks, 'c', Q)
    cr_qnt = quantization(cr_blocks, 'c', Q)
    print('qnt')
    print(y_qnt[0])
    y_zig = encode_zigzag(y_qnt)  # (4096, 64)
    cb_zig = encode_zigzag(cb_qnt)
    cr_zig = encode_zigzag(cr_qnt)
    print(np.array(y_zig).shape)

    return y_zig, cb_zig, cr_zig


def decoding(y_ord, cb_ord, cr_ord, Q, N):  # zigzag로 이미지 복원부분
    y_ord = decode_zigzag(y_ord)
    cb_ord = decode_zigzag(cb_ord)
    cr_ord = decode_zigzag(cr_ord)

    y_deq = dequantization(y_ord, 'y', Q)
    cb_deq = dequantization(cb_ord, 'c', Q)
    cr_deq = dequantization(cr_ord, 'c', Q)
    print(cb_deq.shape)
    print("inverse quantization")
    print(y_deq[0])

    y_i = idct_2d(y_deq)
    cb_i = idct_2d(cb_deq)
    cr_i = idct_2d(cr_deq)
    print("inverse DCT")
    print(y_i[0])

    print("+128")
    print((y_i[0]+128))

    y_i = reconstruct_from_blocks(y_i)
    cb_i = reconstruct_from_blocks(cb_i)
    cr_i = reconstruct_from_blocks(cr_i)

    cb_i = chroma_interpolation(cb_i)
    cr_i = chroma_interpolation(cr_i)


    h, w = y_i.shape
    img = np.zeros([h, w, 3])
    img[:, :, 0], img[:, :, 1], img[:, :, 2] = y_i, cr_i, cb_i
    img = img.astype(np.uint8)
    reconimg = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    return reconimg

def psnr(img1, img2):
    MSE = np.mean((img1-img2)**2)
    ps = 20*np.log10(255/np.sqrt(MSE))
    print(ps)
    with open('psnr.txt', 'a') as pn:
        pn.write(str(ps)+'\n')

    return ps
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim_write(img1, img2):
    ss = ssim(img1, img2)
    with open('ssim.txt', 'a') as pn:
        pn.write(str(ss)+'\n')
    return ss
if __name__ == '__main__':
    N = 8
    qfactor = 50  # 1~100 올라가수록 filesize up.
    direc = "hw2_img"

    lidir = os.listdir(direc)
    psnr_li = []
    ssim_li = []
    for i, file in enumerate(lidir):
        print(direc+'/'+file)
        img = cv2.imread(direc+'/'+file)
        # showimg(img)
        ybit, cbbit, crbit = encoding(img, qfactor, N)
        reconimg = decoding(ybit, cbbit, crbit, qfactor, N)
        # showimg(reconimg)
        psnr_li.append(psnr(img, reconimg))
        ssim_li.append(ssim_write(img, reconimg))
        cv2.imwrite('result/'+str(qfactor)+file[:-4]+'.jpg', reconimg)
    print(psnr_li)
    psnr_mean = np.mean(np.array(psnr_li))
    ssim_mean = np.mean(np.array(ssim_li))
    with open('psnr.txt', 'a') as pn:
        pn.write(f'mean({str(qfactor)}) :'+str(psnr_mean)+'\n')
    with open('ssim.txt', 'a') as pn:
        pn.write(f'mean({str(qfactor)}):'+str(ssim_mean)+'\n')