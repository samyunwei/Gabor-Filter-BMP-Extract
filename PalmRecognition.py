import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import os
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
import csv
import cv2
from Compare import similar

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False


class PalmRecognition(object):
    def __init__(self, dataset_path):
        self.path = dataset_path
        self.gabor_filter_size = [32, 32]
        self.gabor_filter_lambda = [
            # 2 * np.power(2, 0.5),
            4 * np.power(2, 0.5),
            # 6 * np.power(2, 0.5),
        ]  #
        self.gabor_filter_sigma = [
            1.56,
        ]  # 高斯包络的标准差.带宽设置为1时，σ 约= 0.56 λ
        self.gabor_filter_theta = [theta for theta in np.arange(0, np.pi, np.pi / 12)]  # Gabor函数平行条纹的法线方向
        self.gabor_filter_gamma = 0.5  # 空间纵横比
        self.gabor_filter_psi = 0  # 相移
        self.filters_real = []  # 滤波器实数部分
        self.filters_imaginary = []  # 滤波器虚数部分
        self.image_data = []
        self.filtering_result_real_component = []
        self.load_dataset()
        self.build_gabor_filter()

    def load_dataset(self):
        self.image_data = [os.path.join(self.path, f) for f in os.listdir(self.path)]

    def build_gabor_filter(self):
        '''构建滤波器，分为实部和虚部'''
        for r in range(len(self.gabor_filter_lambda)):  # 尺度
            for c in range(len(self.gabor_filter_theta)):  # 方向
                self.filters_real.append(self.build_a_gabor_filters_real_component(self.gabor_filter_size,
                                                                                   self.gabor_filter_sigma[-1],
                                                                                   self.gabor_filter_theta[c],
                                                                                   self.gabor_filter_lambda[r],
                                                                                   self.gabor_filter_gamma,
                                                                                   self.gabor_filter_psi))
        for r in range(len(self.gabor_filter_lambda)):
            for c in range(len(self.gabor_filter_theta)):
                self.filters_imaginary.append(self.build_a_gabor_filters_imaginary_component(self.gabor_filter_size,
                                                                                             self.gabor_filter_sigma[
                                                                                                 -1],
                                                                                             self.gabor_filter_theta[c],
                                                                                             self.gabor_filter_lambda[
                                                                                                 r],
                                                                                             self.gabor_filter_gamma,
                                                                                             self.gabor_filter_psi))

    def show_gabor_filters(self):
        ''' 显示Gabor滤波器，分为实部和虚部 '''
        # 实部
        plt.figure(1, figsize=(9, 9))
        plt.tight_layout()
        plt.axis("off")
        plt.suptitle("实部")
        for i in range(len(self.filters_real)):
            plt.subplot(5, 8, i + 1)
            plt.imshow(self.filters_real[i], cmap="gray")

        plt.show()
        # 虚部
        plt.figure(2, figsize=(9, 9))
        plt.suptitle("虚部")
        for i in range(len(self.filters_imaginary)):
            plt.subplot(5, 8, i + 1)
            plt.imshow(self.filters_imaginary[i], cmap="gray")
        plt.show()

    def show_different_gabor_filteringResult(self, img_index):
        img = cv2.imread(self.image_data[img_index], 0)
        # 实部
        plt.figure(3, figsize=(9, 9))
        plt.suptitle(self.image_data[img_index] + 'real component')
        for i in range(len(self.filters_real)):
            cov_result = signal.convolve2d(img, self.filters_real[i], mode='same', boundary='fill', fillvalue=0)
            plt.subplot(5, 8, i + 1)
            plt.imshow(cov_result, cmap="gray")
        plt.show()
        # 虚部
        plt.figure(4, figsize=(9, 9))
        plt.suptitle(self.image_data[img_index] + 'imaginary component')
        for i in range(len(self.filters_imaginary)):
            cov_result = signal.convolve2d(img, self.filters_imaginary[i], mode='same', boundary='fill', fillvalue=0)
            plt.subplot(5, 8, i + 1)
            plt.imshow(cov_result, cmap="gray")
        plt.show()

    def process(self, img_index, filter_index):
        img = cv2.imread(self.image_data[img_index], 0)
        real_result = signal.convolve2d(img, self.filters_real[filter_index], mode='same', boundary='fill', fillvalue=0)
        image_result = signal.convolve2d(img, self.filters_imaginary[filter_index], mode='same', boundary='fill',
                                         fillvalue=0)
        return real_result, image_result

    def build_a_gabor_filters_real_component(self, ksize,  # 滤波器尺寸; type:list
                                             sigma,  # 高斯包络的标准差
                                             theta,  # Gaobr函数平行余纹的法线方向
                                             lambd,  # 正弦因子的波长
                                             gamma,  # 空间纵横比
                                             psi):  # 相移
        ''' 构建一个gabor滤波器实部'''
        g_f = []
        x_max = int(0.5 * ksize[1])
        y_max = int(0.5 * ksize[0])
        sigma_x = sigma
        sigma_y = sigma / gamma
        c = np.cos(theta)
        s = np.sin(theta)
        scale = 1
        cscale = np.pi * 2 / lambd
        ex = -0.5 / (sigma_x * sigma_x)
        ey = -0.5 / (sigma_y * sigma_y)
        for y in range(-y_max, y_max, 1):
            temp_line = []
            for x in range(-x_max, x_max, 1):
                xr = x * c + y * s
                yr = -x * s + y * c
                temp = scale * np.exp(ex * xr * xr + ey * yr * yr) * np.cos(cscale * xr + psi)
                temp_line.append(temp)
            g_f.append(np.array(temp_line))
        g_f = np.array(g_f)
        return g_f

    def build_a_gabor_filters_imaginary_component(self, ksize,  # 滤波器尺寸; type:list
                                                  sigma,  # 高斯包络的标准差
                                                  theta,  # Gaobr函数平行余纹的法线方向
                                                  lambd,  # 正弦因子的波长
                                                  gamma,  # 空间纵横比
                                                  psi):
        g_f = []
        x_max = int(0.5 * ksize[1])
        y_max = int(0.5 * ksize[0])
        sigma_x = sigma
        sigma_y = sigma / gamma
        c = np.cos(theta)
        s = np.sin(theta)
        scale = 1
        cscale = np.pi * 2 / lambd
        ex = -0.5 / (sigma_x * sigma_x)
        ey = -0.5 / (sigma_y * sigma_y)
        for y in range(-y_max, y_max, 1):
            temp_line = []
            for x in range(-x_max, x_max, 1):
                xr = x * c + y * s
                yr = -x * s + y * c
                temp = scale * np.exp(ex * xr * xr + ey * yr * yr) * np.sin(cscale * xr + psi)
                temp_line.append(temp)
            g_f.append(np.array(temp_line))
        g_f = np.array(g_f)
        return g_f

    def process_images(self):
        '''
        提取每幅图像特征，并保存在csv中
        :return:
        '''
        with open("result/real_component.csv", 'w') as gf_r:
            with open("result/image_component.csv", 'w') as gf_i:
                writer_r = csv.writer(gf_r)
                writer_i = csv.writer(gf_i)
                writer_r.writerow(['image', 'real_feature'])
                writer_i.writerow(['image', 'image_feature'])
                for i in range(len(self.image_data)):
                    real_component, image_component = self.process(i, 0)
                    real_s = ' '.join(str(x) for x in real_component)
                    image_s = ' '.join(str(x) for x in image_component)
                    l_r = np.hstack([self.image_data[i], real_s])
                    l_i = np.hstack([self.image_data[i], image_s])
                    writer_r.writerow(l_r)
                    writer_i.writerow(l_i)

    def comparePalm(self, src, dest, filter_index):
        src_img = cv2.imread(src, 0)
        dest_img = cv2.imread(dest, 0)
        src_real_result = signal.convolve2d(src_img, self.filters_real[filter_index], mode='same', boundary='fill',
                                            fillvalue=0)
        dest_real_result = signal.convolve2d(dest_img, self.filters_real[filter_index], mode='same', boundary='fill',
                                             fillvalue=0)
        return similar(src_real_result, dest_real_result) * 100
