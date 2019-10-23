import os
import random

import cv2
import mxnet as mx
import numpy as np
import onnx
from onnx import checker


def check_onnx(onnx_path):
    '''
        Now we can check validity of the converted ONNX model by using ONNX checker tool.
        The tool will validate the model by checking if the content contains valid protobuf:
        If the converted protobuf format doesn’t qualify to ONNX proto specifications,
        the checker will throw errors, but in this case it successfully passes.
        This method confirms exported model protobuf is valid.
        Now, the model is ready to be imported in other frameworks for inference!
    '''

    model_proto = onnx.load(onnx_path)

    # Check if converted ONNX protobuf is valid
    checker.check_graph(model_proto.graph)


# test시 nms 통과후 적용
def plot_bbox(img, bboxes, scores=None, labels=None, thresh=0.5,
              class_names=None, colors=None, reverse_rgb=False, absolute_coordinates=True,
              image_show=False, image_save=False, image_save_path=None, image_name=None):
    """Visualize bounding boxes.
    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.(range 0 ~ 255 - uint8)
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    labels : numpy.ndarray or mxnet.nd.NDArray, optional
        Class labels of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    class_names : list of str, optional
        Description of parameter `class_names`.
    colors : dict, optional
        You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
        random colors will be substituted.
    reverse_rgb : bool, optional
        Reverse RGB<->BGR orders if `True`.
    absolute_coordinates : bool
        If `True`, absolute coordinates will be considered, otherwise coordinates
        are interpreted as in range(0, 1).
    """
    if labels is not None and not len(bboxes) == len(labels):
        raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                         .format(len(labels), len(bboxes)))
    if scores is not None and not len(bboxes) == len(scores):
        raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                         .format(len(scores), len(bboxes)))

    if image_save:
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)

    img = img.astype(np.uint8)

    if len(bboxes) < 1:
        if image_save:
            cv2.imwrite(os.path.join(image_save_path, image_name + ".jpg"), img)
        if image_show:
            cv2.imshow(image_name, img)
            cv2.waitKey(0)
    else:
        if isinstance(img, mx.nd.NDArray):
            img = img.asnumpy()
        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()
        if isinstance(labels, mx.nd.NDArray):
            labels = labels.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()

        if reverse_rgb:
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]

        copied_img = img.copy()

        if not absolute_coordinates:
            # convert to absolute coordinates using image shape
            height = img.shape[0]
            width = img.shape[1]
            bboxes[:, (0, 2)] *= width
            bboxes[:, (1, 3)] *= height

        # use random colors if None is provided
        if colors is None:
            colors = dict()
        seed_color = [30, 180, 60]

        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.ravel()[i] < thresh:  # threshold보다 작은 것 무시
                continue
            if labels is not None and labels.ravel()[i] < 0:  # 0이하 인것들 인것 무시
                continue

            cls_id = int(labels.ravel()[i]) if labels is not None else -1
            if cls_id not in colors:
                if class_names is not None and cls_id != -1:
                    colors[cls_id] = np.floor(
                        np.multiply(seed_color, np.divide(cls_id + 1, len(class_names))) + np.array([100, 40, 80]))
                    colors[cls_id] = np.clip(colors[cls_id], 0 , 255)
                    colors[cls_id] = colors[cls_id].tolist()  # 사실 여기서는 float64라서 필요 없음. 하지만 바꿔 주겠음.
                else:
                    colors[cls_id] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

            bbox[np.isinf(bbox)] = 0
            bbox[bbox < 0] = 0
            xmin, ymin, xmax, ymax = [int(np.rint(x)) for x in bbox]
            try:
                '''
                colors[cls_id] -> 기본적으로 list, tuple 자료형에 동작함

                numpy인 경우 float64만 동작함 - 나머지 동작안함
                다른 자료형 같은 경우는 tolist로 바꾼 다음 넣어줘야 동작함.
                '''
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[cls_id], thickness=3)
            except Exception as E:
                print(E)

            if class_names is not None and cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = str(cls_id) if cls_id >= 0 else ''

            score = '{:.2f}'.format(scores.ravel()[i]) if scores is not None else ''

            if class_name or score:
                cv2.putText(copied_img,
                            text='{} {}'.format(class_name, score), \
                            org=(xmin + 7, ymin + 20), \
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, \
                            fontScale=0.5, \
                            color=[255, 255, 255], \
                            thickness=1, bottomLeftOrigin=False)

        result = cv2.addWeighted(img, 0.5, copied_img, 0.5, 0)

        if image_save:
            cv2.imwrite(os.path.join(image_save_path, image_name + ".jpg"), result)
        if image_show:
            cv2.imshow(image_name, result)
            cv2.waitKey(0)

        return result
