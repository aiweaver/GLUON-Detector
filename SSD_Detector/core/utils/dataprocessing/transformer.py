"""Transforms described in https://arxiv.org/abs/1512.02325."""
import os

import cv2

from core.utils.util.box_utils import *
from core.utils.util.image_utils import *


# for multiscale
class SSDTrainResize_V0(object):

    def __init__(self, height, width, mean=[0.485, 0.456, 0.406],
                 std =[0.229, 0.224, 0.225]):

        self._height = height
        self._width = width
        self._mean = mean
        self._std = std

    def __call__(self, img, bbox, name):
        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = mx.image.imresize(img, self._width, self._height, interp=interp)
        bbox = box_resize(bbox, (w, h), (self._width, self._height))

        img = mx.nd.image.to_tensor(img)  # 0 ~ 1 로 바꾸기
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        bbox[:, 0] = np.divide(bbox[:, 0], self._width)
        bbox[:, 1] = np.divide(bbox[:, 1], self._height)
        bbox[:, 2] = np.divide(bbox[:, 2], self._width)
        bbox[:, 3] = np.divide(bbox[:, 3], self._height)
        bbox = mx.nd.array(bbox)

        return img, bbox, name

# for multiscale
class SSDTrainTransform_V0(object):

    def __init__(self, height, width, mean=[0.485, 0.456, 0.406],
                 std =[0.229, 0.224, 0.225]):

        self._height = height
        self._width = width
        self._mean = mean
        self._std = std

    def __call__(self, img, bbox, name):

        # random color jittering - photo-metric distortions
        img = image_random_color_distort(img)

        # random expansion with prob 0.5
        expansion = np.random.choice([False, True], p=[0.5, 0.5])
        if expansion:
            # Random expand original image with borders, this is identical to placing the original image on a larger canvas.
            img, expand = random_expand(img, max_ratio=4, fill=[m * 255 for m in [0.485, 0.456, 0.406]],
                                        keep_ratio=True)
            bbox = box_translate(bbox, x_offset=expand[0], y_offset=expand[1], shape=img.shape[:-1])

        # random cropping
        h, w, _ = img.shape
        bbox, crop = box_random_crop_with_constraints(bbox, (w, h),
                                                      min_scale=0.1,
                                                      max_scale=1,
                                                      max_aspect_ratio=2,
                                                      constraints=None,
                                                      max_trial=50)

        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = mx.image.imresize(img, self._width, self._height, interp=interp)
        bbox = box_resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip with probability of 0.5
        h, w, _ = img.shape
        img, flips = random_flip(img, px=0.5)
        bbox = box_flip(bbox, (w, h), flip_x=flips[0])

        # random vertical flip with probability of 0.5
        img, flips = random_flip(img, py=0.5)
        bbox = box_flip(bbox, (w, h), flip_y=flips[1])

        # random translation
        translation = np.random.choice([False, True], p=[0.5, 0.5])
        if translation:
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            img = img.asnumpy()
            x_offset = np.random.randint(-20, high=20)
            y_offset = np.random.randint(-20, high=20)
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])  # +일 경우, (오른쪽, 아래)
            img = cv2.warpAffine(img, M, (w, h), borderValue=[m * 255 for m in [0.406, 0.456, 0.485]])
            bbox = box_translate(bbox, x_offset=x_offset, y_offset=y_offset, shape=(h, w))
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            img = mx.nd.array(img)

        img = mx.nd.image.to_tensor(img)  # 0 ~ 1 로 바꾸기
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        bbox[:, 0] = np.divide(bbox[:, 0], self._width)
        bbox[:, 1] = np.divide(bbox[:, 1], self._height)
        bbox[:, 2] = np.divide(bbox[:, 2], self._width)
        bbox[:, 3] = np.divide(bbox[:, 3], self._height)
        bbox = mx.nd.array(bbox)

        return img, bbox, name

class SSDTrainTransform(object):

    def __init__(self, height, width):

        self._height = height
        self._width = width

    def __call__(self, img, bbox):

        # random color jittering - photo-metric distortions
        img = image_random_color_distort(img)

        # random expansion with prob 0.5
        expansion = np.random.choice([False, True], p=[0.5, 0.5])
        if expansion:
            # Random expand original image with borders, this is identical to placing the original image on a larger canvas.
            img, expand = random_expand(img, max_ratio=4, fill=[m * 255 for m in [0.485, 0.456, 0.406]],
                                        keep_ratio=True)
            bbox = box_translate(bbox, x_offset=expand[0], y_offset=expand[1], shape=img.shape[:-1])

        # random cropping
        h, w, _ = img.shape
        bbox, crop = box_random_crop_with_constraints(bbox, (w, h),
                                                      min_scale=0.1,
                                                      max_scale=1,
                                                      max_aspect_ratio=2,
                                                      constraints=None,
                                                      max_trial=50)

        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = mx.image.imresize(img, self._width, self._height, interp=interp)
        bbox = box_resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip with probability of 0.5
        h, w, _ = img.shape
        img, flips = random_flip(img, px=0.5)
        bbox = box_flip(bbox, (w, h), flip_x=flips[0])

        # random vertical flip with probability of 0.5
        img, flips = random_flip(img, py=0.5)
        bbox = box_flip(bbox, (w, h), flip_y=flips[1])

        # random translation
        translation = np.random.choice([False, True], p=[0.5, 0.5])
        if translation:
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            img = img.asnumpy()
            x_offset = np.random.randint(-20, high=20)
            y_offset = np.random.randint(-20, high=20)
            M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])  # +일 경우, (오른쪽, 아래)
            img = cv2.warpAffine(img, M, (w, h), borderValue=[m * 255 for m in [0.406, 0.456, 0.485]])
            bbox = box_translate(bbox, x_offset=x_offset, y_offset=y_offset, shape=(h, w))
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            img = mx.nd.array(img)

        return img, bbox


# test
if __name__ == "__main__":
    import random
    from core.utils.util.utils import plot_bbox
    from core.utils.dataprocessing.dataset import DetectionDataset

    input_size = (512, 512)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), input_size=input_size,
                               transform=SSDTrainTransform(input_size[0], input_size[1]), image_normalization=False,
                               box_normalization=False)
    length = len(dataset)
    image, label, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('image shape:', image.shape)

    '''
    Data augmentation
    To make the model more robust to various input object sizes and shapes,
    each training image is randomly sampled by one of the following options

    - Use the entire original input image
    - Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3, 0.5, 0.7, or 0.9
    - Randomly sample a patch

    The size of each sampled patch is [0.1, 1] of th original image size, 
    and the aspect ratio is between 0.5 and 2. we keep the overlapped part of the ground truth box
    if the center of it is in the sampled patch. 
    After the aforementioned sampling step, each sampled patch is resized to fixed size and is horizontally flipped
    with probability of 0.5, in addition to applying some photo-metric distortions 
    '''
    plot_bbox(image, label[:, :4],
              scores=None, labels=None,
              class_names=None, colors=None, reverse_rgb=True, absolute_coordinates=True,
              image_show=True, image_save=False, image_save_path="result", image_name=file_name)
