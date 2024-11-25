import os
import os.path
import random
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
import argparse
from PIL import Image

class Dataset(udata.Dataset):
    r"""Implements torch.utils.data.Dataset
    """
    def __init__(self, trainrgb=True, trainsyn=True, shuffle=False):
        super(Dataset, self).__init__()
        self.trainrgb = trainrgb
        self.trainsyn = trainsyn
        self.train_haze = 'dataset.h5'
        
        # Mở file HDF5 và lấy các key của dataset
        h5f = h5py.File(self.train_haze, 'r')
        
        self.keys = list(h5f.keys())
        if shuffle:
            random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        # Truy xuất dữ liệu từ file HDF5 với key tương ứng
        h5f = h5py.File(self.train_haze, 'r')
        
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)

def filter_high_f(fshift, img, radius_ratio):
    """
    Lọc bỏ các thông tin tần số cao ngoài khu vực trung tâm (lọc cao tần).
    """
    # Tạo bộ lọc hình tròn, giá trị trong vòng tròn là 1, ngoài là 0 để lọc
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # Tâm vòng tròn
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        cv2.circle(template, (crow, ccol), radius, 1, -1)
    # Lọc bỏ thông tin tần số cao ngoài khu vực trung tâm
    return template * fshift

def filter_low_f(fshift, img, radius_ratio):
    """
    Lọc bỏ các thông tin tần số thấp trong khu vực trung tâm (lọc thấp tần).
    """
    # Tạo bộ lọc hình tròn, giá trị trong vòng tròn là 0, ngoài là 1 để lọc
    filter_img = np.ones(fshift.shape, np.uint8)
    crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
    radius = int(radius_ratio * img.shape[0] / 2)
    if len(img.shape) == 3:
        cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv2.circle(filter_img, (crow, col), radius, 0, -1)
    # Lọc bỏ thông tin tần số thấp trong khu vực trung tâm
    return filter_img * fshift

def ifft(fshift):
    """
    Phép biến đổi Fourier ngược.
    """
    ishift = np.fft.ifftshift(fshift)  # Chuyển tần số thấp trở về góc trái trên
    iimg = np.fft.ifftn(ishift)  # Kết quả là số phức, không thể hiển thị
    iimg = np.abs(iimg)  # Lấy mô-đun của số phức
    return iimg

def get_low_high_f(img, radius_ratio):
    """
    Lấy phần tần số thấp và tần số cao của ảnh.
    """
    # Biến đổi Fourier
    f = np.fft.fftn(img)  # Biến đổi Fourier rời rạc N chiều
    fshift = np.fft.fftshift(f)  # Chuyển tần số thấp vào giữa ảnh

    # Lấy phần tần số cao và tần số thấp
    hight_parts_fshift = filter_low_f(fshift.copy(), img, radius_ratio=radius_ratio)  # Lọc tần số thấp
    low_parts_fshift = filter_high_f(fshift.copy(), img, radius_ratio=radius_ratio)

    # Biến đổi ngược để lấy lại ảnh
    low_parts_img = ifft(low_parts_fshift)
    high_parts_img = ifft(hight_parts_fshift)

    # Chuẩn hóa ảnh về giá trị [0, 255]
    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    img_new_low = np.array(img_new_low * 255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high

def fft2torch(image):
    """
    Chuyển đổi ảnh từ Fourier thành định dạng Torch tensor.
    """
    image = 255 * image
    low = np.zeros(image.shape)
    high = np.zeros(image.shape)
    for i in range(image.shape[0]):
        low[i, :, :], high[i, :, :] = get_low_high_f(image[i, :, :], radius_ratio=0.5)

    low = low / 255
    high = high / 255

    return low, high

def data_augmentation(clear, haze, mode):
    """
    Thực hiện các phép biến đổi dữ liệu trên ảnh (như xoay, lật, ...).
    """
    clear = np.transpose(clear, (1, 2, 0))
    haze = np.transpose(haze, (1, 2, 0))
    if mode == 0:
        # Không thay đổi
        clear = clear
        haze = haze
    elif mode == 1:
        # Lật theo chiều dọc
        clear = np.flipud(clear)
        haze = np.flipud(haze)
    elif mode == 2:
        # Xoay 90 độ ngược chiều kim đồng hồ
        clear = np.rot90(clear)
        haze = np.rot90(haze)
    elif mode == 3:
        # Xoay 90 độ và lật dọc
        clear = np.rot90(clear)
        clear = np.flipud(clear)
        haze = np.rot90(haze)
        haze = np.flipud(haze)
    elif mode == 4:
        # Xoay 180 độ
        clear = np.rot90(clear, k=2)
        haze = np.rot90(haze, k=2)
    elif mode == 5:
        # Xoay 180 độ và lật
        clear = np.rot90(clear, k=2)
        clear = np.flipud(clear)
        haze = np.rot90(haze, k=2)
        haze = np.flipud(haze)
    elif mode == 6:
        # Xoay 270 độ
        clear = np.rot90(clear, k=3)
        haze = np.rot90(haze, k=3)
    elif mode == 7:
        # Xoay 270 độ và lật
        clear = np.rot90(clear, k=3)
        clear = np.flipud(clear)
        haze = np.rot90(haze, k=3)
        haze = np.flipud(haze)
    else:
        raise Exception('Lựa chọn phép biến đổi không hợp lệ')
    return np.transpose(clear, (2, 0, 1)), np.transpose(haze, (2, 0, 1))

def img_to_patches(img, win, stride, Syn=True):
    """
    Chia ảnh thành các patch nhỏ với kích thước và bước di chuyển (stride) cho trước.
    """
    chl, raw, col = img.shape
    chl = int(chl)
    num_raw = np.ceil((raw - win) / stride + 1).astype(np.uint8)
    num_col = np.ceil((col - win) / stride + 1).astype(np.uint8)
    count = 0
    total_process = int(num_col) * int(num_raw)
    img_patches = np.zeros([chl, win, win, total_process])
    if Syn:
        for i in range(num_raw):
            for j in range(num_col):
                if stride * i + win <= raw and stride * j + win <= col:
                    img_patches[:, :, :, count] = img[:, stride * i: stride * i + win, stride * j: stride * j + win]
                elif stride * i + win > raw and stride * j + win <= col:
                    img_patches[:, :, :, count] = img[:, raw - win: raw, stride * j: stride * j + win]
                elif stride * i + win <= raw and stride * j + win > col:
                    img_patches[:, :, :, count] = img[:, stride * i: stride * i + win, col - win: col]
                else:
                    img_patches[:, :, :, count] = img[:, raw - win: raw, col - win: col]
                count += 1
    return img_patches

def normalize(data):
    """
    Chuẩn hóa dữ liệu ảnh về khoảng [0, 1].
    """
    return np.float32(data / 255.)

def samesize(img, size):
    """
    Điều chỉnh kích thước ảnh về kích thước mong muốn.
    """
    img = cv2.resize(img, size)
    return img

def concatenate2imgs(img, depth):
    """
    Kết hợp hai ảnh thành một ảnh.
    """
    c, w, h = img.shape
    conimg = np.zeros((c + 1, w, h))
    conimg[0:c, :, :] = img
    conimg[c, :, :] = depth
    return conimg

def Train_data():
    """
    Tạo tập dữ liệu huấn luyện từ các ảnh đầu vào và lưu trữ trong file HDF5.
    """
    train_data = 'dataset.h5'
    files1_haze = os.listdir('./input/')
    files1_clear = os.listdir('./input/')
    files2_haze = os.listdir('./input/')
    files2_clear = os.listdir('./input/')

    with h5py.File(train_data, 'w') as h5f:
        count1 = 0
        
        # Xử lý tập dữ liệu đầu tiên
        for i in range(len(files1_haze)):
            haze1 = np.array(Image.open('./input/' + files1_haze[i])) / 255
            clear1 = np.array(Image.open('./input/' + files1_clear[i])) / 255

            haze1 = haze1.transpose(2, 0, 1)
            clear1 = clear1.transpose(2, 0, 1)
            haze1 = img_to_patches(haze1, 256, 200)
            clear1 = img_to_patches(clear1, 256, 200)
            for nx in range(clear1.shape[3]):
                haze, clear = data_augmentation(haze1[:, :, :, nx].copy(), clear1[:, :, :, nx].copy(), np.random.randint(0, 7))
                clear_high, clear_low = fft2torch(clear)
                data1 = np.concatenate([haze, clear_high, clear_low, clear], 0)
                h5f.create_dataset(str(count1), data=data1)
                count1 += 1
                print(count1, clear1.shape[3], data1.shape)
        
        count2 = 0
        # Xử lý tập dữ liệu thứ hai
        for i in range(len(files2_haze)):
            haze2 = np.array(Image.open('./input/' + files2_haze[i])) / 255
            clear2 = np.array(Image.open('./input/' + files2_clear[i])) / 255

            haze2 = haze2.transpose(2, 0, 1)
            clear2 = clear2.transpose(2, 0, 1)
            haze2 = img_to_patches(haze2, 256, 256)
            clear2 = img_to_patches(clear2, 256, 256)
            for nx in range(clear2.shape[3]):
                haze, clear = data_augmentation(haze2[:, :, :, nx].copy(), clear2[:, :, :, nx].copy(), np.random.randint(0, 7))
                clear_high, clear_low = fft2torch(clear)
                data2 = np.concatenate([haze, clear_high, clear_low, clear], 0)
                h5f.create_dataset(str(count1), data=data2)
                count1 += 1
                count2 += 1
                print(count2, clear2.shape[3], data2.shape)
        print(count1 - count2, count2)

    h5f.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Building the training patch database")
    parser.add_argument("--patch_size", "--p", type=int, default=128, help="Patch size")
    parser.add_argument("--stride", "--s", type=int, default=64, help="Size of stride")
    args = parser.parse_args()
    
    Train_data()
