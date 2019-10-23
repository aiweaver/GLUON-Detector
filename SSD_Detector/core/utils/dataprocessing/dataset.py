import glob
import os
from xml.etree.ElementTree import parse

# import json
import mxnet as mx
import mxnet.gluon as gluon
import numpy as np

from core.utils.util.box_utils import box_resize

# class DetectionDataset_V0(gluon.data.Dataset):
#     """
#     Parameters
#     ----------
#     path : str(jpg)
#         Path to input image directory.
#     input_size: tuple or list -> (height(int), width(int))
#     transform : object
#     mean : 이미지 정규화 한 뒤 뺄 값, Default [0.485, 0.456, 0.406]
#     std : 이미지 정규하 한 뒤 나눌 값 Default [0.229, 0.224, 0.225]
#     image_normalization = True : SSD 학습시, False : Visualization, Default True
#     box_normalization : True : SSD 학습시, False : Visualization, Default True
#     """
#     CLASSES = ['fire', 'smoke']
#
#     def __init__(self, path='Dataset/train', input_size=(512, 512), mean=[0.485, 0.456, 0.406],
#                  std=[0.229, 0.224, 0.225],
#                  transform=None, image_normalization=True, box_normalization=True):
#         super(DetectionDataset, self).__init__()
#         self._name = os.path.basename(path)
#         self._image_path_List = glob.glob(os.path.join(path, "*.jpg"))
#         self._height = input_size[0]
#         self._width = input_size[1]
#         self._transform = transform
#         self._items = []
#         self._image_normalization = image_normalization
#         self._box_normalization = box_normalization
#         self.itemname = []
#         self._mean = mean
#         self._std = std
#         self._make_item_list()
#
#     def _make_item_list(self):
#         for image_path in self._image_path_List:
#             label_path = image_path.replace(".jpg", ".json")
#             self._items.append((image_path, label_path))
#
#             # 이름 저장
#             base_image = os.path.basename(image_path)
#             name = os.path.splitext(base_image)[0]
#             self.itemname.append(name)
#
#     def __getitem__(self, idx):
#
#         image_path, label_path = self._items[idx]
#         image = mx.image.imread(image_path, flag=1, to_rgb=True)
#         label = self._parsing(label_path)  # dtype을 float 으로 해야 아래 단계에서 편하다
#         return image, label, self.itemname[idx]
#
#     def _parsing(self, path):
#         json_list = []
#         # json파일 parsing - 순서 -> topleft_x, topleft_y, bottomright_x, bottomright_y, center_x, center_y
#         with open(path, mode='r', errors='ignore') as json_file:
#             try:
#                 dict = json.load(json_file)
#                 for i in range(len(dict["landmarkAttr"])):
#                     try:
#                         xmin = int(dict["landmarkAttr"][i]["box"][0]['x'])
#                         ymin = int(dict["landmarkAttr"][i]["box"][0]['y'])
#                         xmax = int(dict["landmarkAttr"][i]["box"][1]['x'])
#                         ymax = int(dict["landmarkAttr"][i]["box"][1]['y'])
#                         classes = int(dict["landmarkAttr"][i]["attributes"][0]['selected'])
#                     except Exception:
#                         print("이상 파일 : " + path)
#                         exit(0)
#                     else:
#                         json_list.append((xmin, ymin, xmax, ymax, classes))
#                 return np.array(json_list, dtype="float32")  # 반드시 numpy여야함.
#
#             except Exception:
#                 print("이상 파일 : " + path)
#                 exit(0)
#
#     @property
#     def classes(self):
#         return self.CLASSES
#
#     @property
#     def num_class(self):
#         """Number of categories."""
#         return len(self.CLASSES)
#
#     def __str__(self):
#         return self._name + " " + "dataset"
#
#     def __len__(self):
#         return len(self._items)

# class DetectionDataset(gluon.data.Dataset):
#     """
#     Parameters
#     ----------
#     path : str(jpg)
#         Path to input image directory.
#     input_size: tuple or list -> (height(int), width(int))
#     transform : object
#     mean : 이미지 정규화 한 뒤 뺄 값, Default [0.485, 0.456, 0.406]
#     std : 이미지 정규하 한 뒤 나눌 값 Default [0.229, 0.224, 0.225]
#     image_normalization = True : SSD 학습시, False : Visualization, Default True
#     box_normalization : True : SSD 학습시, False : Visualization, Default True
#     """
#     CLASSES = ['fire', 'smoke']
#
#     def __init__(self, path='Dataset/train', input_size=(512, 512), mean=[0.485, 0.456, 0.406],
#                  std=[0.229, 0.224, 0.225],
#                  transform=None, image_normalization=True, box_normalization=True):
#         super(DetectionDataset, self).__init__()
#         self._name = os.path.basename(path)
#         self._image_path_List = glob.glob(os.path.join(path, "*.jpg"))
#         self._height = input_size[0]
#         self._width = input_size[1]
#         self._transform = transform
#         self._items = []
#         self._image_normalization = image_normalization
#         self._box_normalization = box_normalization
#         self.itemname = []
#         self._mean = mean
#         self._std = std
#         self._make_item_list()
#
#     def _make_item_list(self):
#         for image_path in self._image_path_List:
#             label_path = image_path.replace(".jpg", ".json")
#             self._items.append((image_path, label_path))
#
#             # 이름 저장
#             base_image = os.path.basename(image_path)
#             name = os.path.splitext(base_image)[0]
#             self.itemname.append(name)
#
#     def __getitem__(self, idx):
#
#         image_path, label_path = self._items[idx]
#         image = mx.image.imread(image_path, flag=1, to_rgb=True)
#         label = self._parsing(label_path)  # dtype을 float 으로 해야 아래 단계에서 편하다
#
#         h, w, _ = image.shape
#         if self._transform is not None:
#             image, label = self._transform(image, label)
#             if self._image_normalization:
#                 # to tensor
#                 # 아래의 두 함수는 mxnet 홈페이지에 설명 없음
#                 # c로 구현 한 것 파이썬으로 바인딩함 - 빠름
#                 image = mx.nd.image.to_tensor(image)  # 0 ~ 1 로 바꾸기
#                 image = mx.nd.image.normalize(image, mean=self._mean, std=self._std)
#             if self._box_normalization:  # box normalization (range 0 ~ 1)
#                 label[:, 0] = np.divide(label[:, 0], self._width)
#                 label[:, 1] = np.divide(label[:, 1], self._height)
#                 label[:, 2] = np.divide(label[:, 2], self._width)
#                 label[:, 3] = np.divide(label[:, 3], self._height)
#                 label = mx.nd.array(label)
#                 return image, label, self.itemname[idx]
#             else:
#                 return image, label, self.itemname[idx]
#         else:
#             image = mx.image.imresize(image, w=self._width, h=self._height, interp=3)
#             label = box_resize(label, (w, h), (self._width, self._height))
#             if self._image_normalization:
#                 image = mx.nd.image.to_tensor(image)  # 0 ~ 1 로 바꾸기
#                 image = mx.nd.image.normalize(image, mean=self._mean, std=self._std)
#             if self._box_normalization:  # box normalization (range 0 ~ 1)
#                 label[:, 0] = np.divide(label[:, 0], self._width)
#                 label[:, 1] = np.divide(label[:, 1], self._height)
#                 label[:, 2] = np.divide(label[:, 2], self._width)
#                 label[:, 3] = np.divide(label[:, 3], self._height)
#                 label = mx.nd.array(label)
#                 return image, label, self.itemname[idx]
#             else:
#                 label = mx.nd.array(label)
#                 return image, label, self.itemname[idx]
#
#     def _parsing(self, path):
#         json_list = []
#         # json파일 parsing - 순서 -> topleft_x, topleft_y, bottomright_x, bottomright_y, center_x, center_y
#         with open(path, mode='r', errors='ignore') as json_file:
#             try:
#                 dict = json.load(json_file)
#                 for i in range(len(dict["landmarkAttr"])):
#                     try:
#                         xmin = int(dict["landmarkAttr"][i]["box"][0]['x'])
#                         ymin = int(dict["landmarkAttr"][i]["box"][0]['y'])
#                         xmax = int(dict["landmarkAttr"][i]["box"][1]['x'])
#                         ymax = int(dict["landmarkAttr"][i]["box"][1]['y'])
#                         classes = int(dict["landmarkAttr"][i]["attributes"][0]['selected'])
#                     except Exception:
#                         print("이상 파일 : " + path)
#                         exit(0)
#                     else:
#                         json_list.append((xmin, ymin, xmax, ymax, classes))
#                 return np.array(json_list, dtype="float32")  # 반드시 numpy여야함.
#
#             except Exception:
#                 print("이상 파일 : " + path)
#                 exit(0)
#
#     @property
#     def classes(self):
#         return self.CLASSES
#
#     @property
#     def num_class(self):
#         """Number of categories."""
#         return len(self.CLASSES)
#
#     def __str__(self):
#         return self._name + " " + "dataset"
#
#     def __len__(self):
#         return len(self._items)

# https://github.com/5taku/custom_object_detection
class DetectionDataset_V0(gluon.data.Dataset):
    """
    Parameters
    ----------
    path : str(jpg)
        Path to input image directory.
    input_size : tuple or list -> (height(int), width(int))
    transform : object
    mean : 이미지 정규화 한 뒤 뺄 값, Default [0.485, 0.456, 0.406]
    std : 이미지 정규하 한 뒤 나눌 값 Default [0.229, 0.224, 0.225]
    image_normalization = True : SSD 학습시, False : Visualization, Default True
    box_normalization : True : SSD 학습시, False : Visualization, Default True
    """
    CLASSES = ['meerkat', 'otter', 'panda', 'raccoon', 'pomeranian']

    def __init__(self, path='Dataset/train'):
        super(DetectionDataset_V0, self).__init__()
        self._name = os.path.basename(path)
        self._image_path_List = glob.glob(os.path.join(path, "*.jpg"))
        self._items = []
        self.itemname = []
        self._make_item_list()

    def _make_item_list(self):
        for image_path in self._image_path_List:

            xml_path = image_path.replace(".jpg", ".xml")
            self._items.append((image_path, xml_path))

            # 이름 저장
            base_image = os.path.basename(image_path)
            name = os.path.splitext(base_image)[0]
            self.itemname.append(name)

    def __getitem__(self, idx):

        image_path, label_path = self._items[idx]
        image = mx.image.imread(image_path, flag=1, to_rgb=True)
        label = self._parsing(label_path)  # dtype을 float 으로 해야 아래 단계에서 편하다
        return image, label, self.itemname[idx]

    def _parsing(self, path):
        xml_list = []
        try:
            tree = parse(path)
            root = tree.getroot()
            object = root.findall("object")
            for ob in object:
                bndbox = ob.find("bndbox")
                xmin, ymin, xmax, ymax = [int(pos.text) for i, pos in enumerate(bndbox.iter()) if i > 0]

                # or
                # xmin = int(bndbox.findtext("xmin"))
                # ymin = int(bndbox.findtext("ymin"))
                # xmax = int(bndbox.findtext("xmax"))
                # ymax = int(bndbox.findtext("ymax"))

                select = ob.findtext("name")
                if select == "meerkat":
                    classes = 0
                elif select == "otter":
                    classes = 1
                elif select == "panda":
                    classes = 2
                elif select == "raccoon":
                    classes = 3
                elif select == "pomeranian":
                    classes = 4
                xml_list.append((xmin, ymin, xmax, ymax, classes))

        except Exception:
            print("이상 파일 : " + path)
            exit(0)
        else:
            return np.array(xml_list, dtype="float32")  # 반드시 numpy여야함.

    @property
    def classes(self):
        return self.CLASSES

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.CLASSES)

    def __str__(self):
        return self._name + " " + "dataset"

    def __len__(self):
        return len(self._items)


# https://github.com/5taku/custom_object_detection
class DetectionDataset(gluon.data.Dataset):
    """
    Parameters
    ----------
    path : str(jpg)
        Path to input image directory.
    input_size : tuple or list -> (height(int), width(int))
    transform : object
    mean : 이미지 정규화 한 뒤 뺄 값, Default [0.485, 0.456, 0.406]
    std : 이미지 정규하 한 뒤 나눌 값 Default [0.229, 0.224, 0.225]
    image_normalization = True : SSD 학습시, False : Visualization, Default True
    box_normalization : True : SSD 학습시, False : Visualization, Default True
    """
    CLASSES = ['meerkat', 'otter', 'panda', 'raccoon', 'pomeranian']

    def __init__(self, path='Dataset/train', input_size=(512, 512), mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 transform=None, image_normalization=True, box_normalization=True):
        super(DetectionDataset, self).__init__()
        self._name = os.path.basename(path)
        self._image_path_List = glob.glob(os.path.join(path, "*.jpg"))
        self._height = input_size[0]
        self._width = input_size[1]
        self._transform = transform
        self._items = []
        self._image_normalization = image_normalization
        self._box_normalization = box_normalization
        self.itemname = []
        self._mean = mean
        self._std = std
        self._make_item_list()

    def _make_item_list(self):
        for image_path in self._image_path_List:

            xml_path = image_path.replace(".jpg", ".xml")
            self._items.append((image_path, xml_path))

            # 이름 저장
            base_image = os.path.basename(image_path)
            name = os.path.splitext(base_image)[0]
            self.itemname.append(name)

    def __getitem__(self, idx):

        image_path, label_path = self._items[idx]
        image = mx.image.imread(image_path, flag=1, to_rgb=True)
        label = self._parsing(label_path)  # dtype을 float 으로 해야 아래 단계에서 편하다

        h, w, _ = image.shape
        if self._transform is not None:
            image, label = self._transform(image, label)
            if self._image_normalization:
                # to tensor
                # 아래의 두 함수는 mxnet 홈페이지에 설명 없음
                # c로 구현 한 것 파이썬으로 바인딩함 - 빠름
                image = mx.nd.image.to_tensor(image)  # 0 ~ 1 로 바꾸기
                image = mx.nd.image.normalize(image, mean=self._mean, std=self._std)
            if self._box_normalization:  # box normalization (range 0 ~ 1)
                label[:, 0] = np.divide(label[:, 0], self._width)
                label[:, 1] = np.divide(label[:, 1], self._height)
                label[:, 2] = np.divide(label[:, 2], self._width)
                label[:, 3] = np.divide(label[:, 3], self._height)
                label = mx.nd.array(label)
                return image, label, self.itemname[idx]
            else:
                return image, label, self.itemname[idx]
        else:
            image = mx.image.imresize(image, w=self._width, h=self._height, interp=3)
            label = box_resize(label, (w, h), (self._width, self._height))
            if self._image_normalization:
                image = mx.nd.image.to_tensor(image)  # 0 ~ 1 로 바꾸기
                image = mx.nd.image.normalize(image, mean=self._mean, std=self._std)
            if self._box_normalization:  # box normalization (range 0 ~ 1)
                label[:, 0] = np.divide(label[:, 0], self._width)
                label[:, 1] = np.divide(label[:, 1], self._height)
                label[:, 2] = np.divide(label[:, 2], self._width)
                label[:, 3] = np.divide(label[:, 3], self._height)
                label = mx.nd.array(label)
                return image, label, self.itemname[idx]
            else:
                label = mx.nd.array(label)
                return image, label, self.itemname[idx]

    def _parsing(self, path):
        xml_list = []
        try:
            tree = parse(path)
            root = tree.getroot()
            object = root.findall("object")
            for ob in object:
                bndbox = ob.find("bndbox")
                xmin, ymin, xmax, ymax = [int(pos.text) for i, pos in enumerate(bndbox.iter()) if i > 0]

                # or
                # xmin = int(bndbox.findtext("xmin"))
                # ymin = int(bndbox.findtext("ymin"))
                # xmax = int(bndbox.findtext("xmax"))
                # ymax = int(bndbox.findtext("ymax"))

                select = ob.findtext("name")
                if select == "meerkat":
                    classes = 0
                elif select == "otter":
                    classes = 1
                elif select == "panda":
                    classes = 2
                elif select == "raccoon":
                    classes = 3
                elif select == "pomeranian":
                    classes = 4
                xml_list.append((xmin, ymin, xmax, ymax, classes))

        except Exception:
            print("이상 파일 : " + path)
            exit(0)
        else:
            return np.array(xml_list, dtype="float32")  # 반드시 numpy여야함.

    @property
    def classes(self):
        return self.CLASSES

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.CLASSES)

    def __str__(self):
        return self._name + " " + "dataset"

    def __len__(self):
        return len(self._items)


# test
if __name__ == "__main__":
    import random
    from core.utils.util.utils import plot_bbox

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), input_size=(512, 512),
                               transform=None, image_normalization=False, box_normalization=False)
    # dataset = DetectionDataset_V0(path=os.path.join(root, 'Dataset', 'train'))
    length = len(dataset)
    image, label, file_name = dataset[random.randint(0, length - 1)]
    print('images length:', length)
    print('image shape:', image.shape)

    plot_bbox(image, label[:, :4],
              scores=None, labels=label[:, 4:5],
              class_names=dataset.classes, colors=None, reverse_rgb=True, absolute_coordinates=True,
              image_show=True, image_save=False, image_save_path="result", image_name=file_name)
