import os
HOME = os.environ["HOME"]

import sys

import numpy as np

import cv2





def get_tamper_pattern(img_x, img_t):
    """returns the image difference between
       the original image img_x and the tampered image img_t"""
    return img_t.astype(int) - img_x.astype(int)

def apply_tamper_pattern(img_y, img_x, img_t):
    """apply the tamper pattern (image difference)
       to the watermarked image"""
    tamper_pattern = get_tamper_pattern(img_x, img_t)
    # tamper_map = (tamper_pattern/tamper_pattern)*255.
    tamper_map = np.zeros_like(img_x)
    tamper_map[np.where(tamper_pattern != 0)] = 255.

    img_z = img_y + tamper_pattern
    return img_z, tamper_map

def tamper_image(img, shape=(8, 8), position=(0, 0), tampertype="white"):
    """simulate a tamper by modifying an image with a block"""
    img_z = np.copy(img)
    posh, posw = position
    shapeh, shapew = shape
    h0, h1 = posh, posh + shapeh
    w0, w1 = posw, posw + shapew

    tamper_types_translation = {
        "white"  : np.zeros(shape)+255.,
        "black"  : np.zeros(shape),
        "random" : np.random.choice(range(256), shape),
    }


    tamper_map = np.zeros_like(img)
    tamper_map[h0:h1, w0:w1] = 255.

    if tamper_types_translation.has_key(tampertype):
        tamper_block = tamper_types_translation[tampertype]
        img_z[h0:h1, w0:w1] = tamper_block
        return img_z, tamper_map
    else:
        raise ValueError("this tamper type doesn't exist")


# https://stackoverflow.com/questions/40768621/python-opencv-jpeg-compression-in-memory
def jpeg_compression(image, quality_factor):
    """jpeg compression of ndarray with opencv2"""
    quality_param = [cv2.IMWRITE_JPEG_QUALITY, int(quality_factor)]
    _, compressed = cv2.imencode(".jpg", image, quality_param)
    l = len(image.shape)
    if l == 3 and image.shape[2] == 3:
        cv_load_mode = 1  # detection of color image
        compressed_array = cv2.imdecode(compressed, cv_load_mode)
        return compressed_array
    elif l == 2:
        cv_load_mode = 0  # detection of grayscale image
        compressed_array = cv2.imdecode(compressed, cv_load_mode)
        return compressed_array

def amplitudescale_change(image, amplitude_param):
    modified_image = image * amplitude_param
    return np.clip(modified_image, 0, 255)


def luminance_constantchange(image, luminance_param):
    modified_image = image + luminance_param
    return np.clip(modified_image, 0, 255)


def additive_white_gaussian_noise(image, deviation):
    mu = 1
    noisy_image = image + np.random.normal(mu, deviation, image.shape)
    return np.clip(noisy_image, 0, 255)


def low_pass_filter(img, nb_application, size=3):
    """apply a low pass filter on an image"""
    blurred = img
    for _ in range(int(nb_application)):
        blurred = cv2.blur(img, (size, size))
    return blurred

# Laplacian filter for high pass filter
kernel_lapacian = np.array(
    [
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=float)

def high_pass_filter(img, nb_application):
    """apply a high pass filter on an image"""
    kernel = kernel_lapacian
    filtered = img
    for _ in range(int(nb_application)):
        filtered = cv2.filter2D(img, -1, kernel)
    return filtered


image_processings = {
    "compression" : jpeg_compression,
    "luminance"   : luminance_constantchange,
    "amplitude"   : amplitudescale_change,
    "awgn"        : additive_white_gaussian_noise,
    "lpf"         : low_pass_filter,
    "hpf"         : high_pass_filter,
}


def main():
    pass


if __name__ == "__main__":
    sys.exit(main())