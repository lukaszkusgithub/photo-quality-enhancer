import os
import cv2 as cv
import numpy as np

_CURRENT_SYSTEM_NAME = os.name

_LOCAL_PATH = os.path.dirname(os.path.abspath('main.py'))

if _CURRENT_SYSTEM_NAME == 'nt':
    _FRAMES_PATH = os.path.join(_LOCAL_PATH, "Frames\\")
    _NEW_FRAMES_PATH = os.path.join(_LOCAL_PATH, "newFrames\\")
elif _CURRENT_SYSTEM_NAME == 'posix':
    _FRAMES_PATH = os.path.join(_LOCAL_PATH, "Frames/")
    _NEW_FRAMES_PATH = os.path.join(_LOCAL_PATH, "newFrames/")


def resize_image(file):
    img_path = os.path.join(_FRAMES_PATH, file)
    img = cv.imread(img_path)
    width = 1280
    height = int(img.shape[0] * width / img.shape[1])
    dim = (width, height)

    sr = cv.dnn_superres.DnnSuperResImpl_create()

    path = "models/EDSR_x4.pb"
    # path = "models/FSRCNN_x2.pb"

    sr.readModel(path)

    sr.setModel("edsr", 4)

    result = sr.upsample(img)

    # if width >= result.shape[1]:
    #     resized = cv.resize(result, dim, interpolation=cv.INTER_CUBIC)
    # elif width < result.shape[1]:
    #     resized = cv.resize(result, dim, interpolation=cv.INTER_AREA)

    # if width >= img.shape[1]:
    #     resized = cv.resize(img, dim, interpolation=cv.INTER_CUBIC)
    # elif width < img.shape[1]:
    #     resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    # resized = cv.detailEnhance(resized, sigma_s=10, sigma_r=0.15)

    return result


def iterate_files():
    for file in os.listdir(_FRAMES_PATH):
        new_img = resize_image(file)
        img_new_path = os.path.join(_NEW_FRAMES_PATH, file)
        cv.imwrite(img_new_path, new_img)


def main():
    iterate_files()
    return 0


if __name__ == '__main__':
    main()
