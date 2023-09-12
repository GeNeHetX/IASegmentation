# script_cython.pyx
import numpy as np
cimport numpy as np
import cv2
# cimport cv2
cimport cython
from libc.math cimport floor

@cython.boundscheck(False)
@cython.wraparound(False)
def process_images(str DATA_DIR, list img_name, int down_sample, int size):
    cdef int end_img_x, end_img_y
    cdef long[:, :, :] end_img
    cdef int start_x, start_y, y_tot, x_tot, y_a, x_a
    cdef int val
    cdef str folder_dir, img_path
    cdef np.ndarray image

    end_img_x, end_img_y = 0, 0

    print("Beginning")
    # Calculate x_max and y_max
    for img in img_name:
        x = int((int(img.split("=")[2][:-2])) / down_sample) + size
        if x > end_img_x:
            end_img_x = x
        y = int((int(img.split("=")[3][:-2])) / down_sample) + size
        if y > end_img_y:
            end_img_y = y

    end_img = np.zeros((2, end_img_y, end_img_x), dtype=np.int64)
    for img in img_name:
        image = cv2.imread(DATA_DIR+img, cv2.IMREAD_GRAYSCALE)
        start_x = int(int(img.split("=")[2][:-2]) / down_sample)
        start_y = int(int(img.split("=")[3][:-2]) / down_sample)
        for y_a in range(image.shape[0]):
            y_tot = start_y + y_a
            for x_a in range(image.shape[1]):
                x_tot = start_x + x_a
                val = image[y_a, x_a]
                if end_img[1, y_tot, x_tot] == 0:
                    end_img[0, y_tot, x_tot] = val
                else:
                    end_img[0, y_tot, x_tot] +=val
                end_img[1, y_tot, x_tot] += 1
    print("division operation")
    for y in range(end_img_y):
        for x in range(end_img_x):
            if end_img[1, y, x]!=0:
                end_img[0, y, x]=int(end_img[0, y, x]/end_img[1, y, x])
            else:
                end_img[0, y, x]=0

    print("end_operation division")
    return end_img[0]

def process_tile_images(str DATA_DIR, list img_name, int down_sample, int size):
    cdef int end_img_x, end_img_y
    cdef long[:, :] end_img
    cdef int start_x, start_y, y_tot, x_tot, y_a, x_a
    cdef int val
    cdef str folder_dir, img_path
    cdef np.ndarray image

    end_img_x, end_img_y = 0, 0

    print("beginning")
    # Calculate x_max and y_max
    for img in img_name:
        x = int((int(img.split("=")[2][:-2])) / down_sample) + size
        if x > end_img_x:
            end_img_x = x
        y = int((int(img.split("=")[3][:-2])) / down_sample) + size
        if y > end_img_y:
            end_img_y = y
    end_img = np.zeros((end_img_y, end_img_x), dtype=np.int64)

    for img in img_name:
        image =cv2.imread(DATA_DIR+img, cv2.IMREAD_GRAYSCALE)
        image=np.where(image==255, 0, 255)
        start_x=int(int(img.split("=")[2][:-2])/down_sample)
        start_y=int(int(img.split("=")[3][:-2])/down_sample)
        for y_a, col in enumerate(image):
            y_tot=start_y+y_a
            for x_a, val in enumerate(col):
                x_tot=start_x+x_a
                end_img[y_tot][x_tot]=val

    return end_img