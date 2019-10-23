import logging
import os
import platform

import cv2
import mxnet as mx
import mxnet.gluon as gluon
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
from tqdm import tqdm

from core import Prediction
from core import plot_bbox
from core import testdataloader
from core.model.backbone.VGG16 import VGG16

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def anchor_generator(input_size=None, feature_size=None, box_size=None, box_ratio=None, box_offset=(0.5, 0.5), box_clip=True):
    scaled_box_first_width = box_size[0] / input_size[1]
    scaled_box_first_height = box_size[0] / input_size[0]

    scaled_box_second_width = box_size[1] / input_size[1]
    scaled_box_second_height = box_size[1] / input_size[0]

    anchors = []
    for y in range(feature_size[0]):
        for x in range(feature_size[1]):

            cy = (y + box_offset[0]) / feature_size[0]
            cx = (x + box_offset[1]) / feature_size[1]

            anchors.append([cx, cy, scaled_box_first_width, scaled_box_first_height])
            anchors.append([cx, cy, scaled_box_second_width, scaled_box_second_height])
            for r in box_ratio[1:]:
                sr = np.sqrt(r)
                w = scaled_box_first_width * sr
                h = scaled_box_first_height / sr
                anchors.append([cx, cy, w, h])
    anchors = np.array(anchors)

    if box_clip:
        anchors = np.clip(anchors, 0, 1)
    return anchors


def run(mean= [0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
        box_sizes300=[21, 45, 101.25, 157.5, 213.75, 270, 326.25],
        box_ratios300=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0 / 3]] * 3 + [[1, 2, 0.5]] * 2,
        box_sizes512=[21, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
        box_ratios512=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0 / 3]] * 4 + [[1, 2, 0.5]] * 2,
        anchor_box_clip=True,
        use_onnx_model=False,
        video_path="",
        load_name="400_600_ADAM_PVGG16_512",
        load_period=100, GPU_COUNT=0,
        decode_number=-1,
        nms_thresh=0.45,
        nms_topk=500,
        except_class_thresh=0.01,
        plot_class_thresh=0.5,
        video_save_path="result",
        video_show=True,
        video_save=True,
        make_anchor=False):
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)

    if GPU_COUNT <= 0:
        ctx = mx.cpu(0)
    elif GPU_COUNT > 0:
        ctx = mx.gpu(0)

    # 운영체제 확인
    if platform.system() == "Linux":
        logging.info(f"{platform.system()} OS")
    elif platform.system() == "Windows":
        logging.info(f"{platform.system()} OS")
    else:
        logging.info(f"{platform.system()} OS")

    if GPU_COUNT > 0:
        free_memory, total_memory = mx.context.gpu_memory_info(0)
        free_memory = round(free_memory / (1024 * 1024 * 1024), 2)
        total_memory = round(total_memory / (1024 * 1024 * 1024), 2)
        logging.info(f'Running on {ctx} / free memory : {free_memory}GB / total memory {total_memory}GB')
    else:
        logging.info(f'Running on {ctx}')

    logging.info(f"test {load_name}")
    height = int(load_name.split("_")[0])
    width = int(load_name.split("_")[1])
    if not isinstance(height, int) and not isinstance(width, int):
        logging.info(f"height is not int")
        logging.info(f"width is not int")
        raise ValueError
    else:
        logging.info(f"input size : {(height, width)}")

    try:
        _, test_dataset = testdataloader()
    except Exception:
        logging.info("The dataset does not exist")
        exit(0)

    weight_path = f"weights/{load_name}"
    symbol_path = os.path.join(weight_path, '{}-symbol.json'.format(load_name))
    param_path = os.path.join(weight_path, '{0}-{1:04d}.params'.format(load_name, load_period))
    onnx_path = os.path.join(weight_path, "{}.onnx".format(load_name))

    if use_onnx_model:
        logging.info("ONNX model test")
        try:
            net = onnx_mxnet.import_to_gluon(onnx_path, ctx)
        except Exception:
            logging.info("loading ONNX weights 실패")
            exit(0)
        else:
            logging.info("loading ONNX weights 성공")

    else:
        logging.info("symbol model test")
        try:
            net = gluon.SymbolBlock.imports(symbol_path,
                                            ['data'],
                                            param_path, ctx=ctx)
        except Exception:
            # DEBUG, INFO, WARNING, ERROR, CRITICAL 의 5가지 등급
            logging.info("loading symbol weights 실패")
            exit(0)
        else:
            logging.info("loading symbol weights 성공")

    if make_anchor:

        feature_sizes = []
        version = int(load_name.split("_")[-1])

        if version == 300:
            box_ratios = box_ratios300
            box_sizes = box_sizes300
        elif version == 512:
            box_ratios = box_ratios512
            box_sizes = box_sizes512

        sizes = list(zip(box_sizes[:-1], box_sizes[1:]))
        vgg_features = VGG16(version=version, ctx=ctx)
        vgg_output = vgg_features(
            mx.nd.random_uniform(low=0, high=1, shape=(1, 3, height, width), ctx=ctx))
        for vgg in vgg_output:
            feature_sizes.append(vgg.shape[2:])

        anchor_list = []
        for size, ratio, feature_size in zip(sizes, box_ratios, feature_sizes):
            anchor_list.append(anchor_generator(
                feature_size=feature_size,
                input_size=(height, width),
                box_size=size,
                box_ratio=ratio,
                box_clip=anchor_box_clip))

        generated_anchors = np.reshape(np.concatenate(anchor_list, axis=0), (1, -1, 4))
        generated_anchors = mx.nd.array(generated_anchors, ctx=ctx)

    net.hybridize(active=True, static_alloc=True, static_shape=True)

    # BoxEncoder, BoxDecoder 에서 같은 값을 가져야함
    prediction = Prediction(
        softmax=False,
        num_classes=test_dataset.num_class,
        decode_number=decode_number,
        nms_thresh=nms_thresh,
        nms_topk=nms_topk,
        except_class_thresh=except_class_thresh)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = os.path.splitext(os.path.basename(video_path))[0]
    out = cv2.VideoWriter(os.path.join(video_save_path, f'{name}.avi'), fourcc, 30, (width, height))

    # while(cap.isOpened()):
    for _ in tqdm(range(0, nframe)):
        ret, image = cap.read()
        if ret:
            origin_image = image.copy()
            origin_image = cv2.resize(origin_image, (width, height), cv2.INTER_AREA)

            image[:, :, (0, 1, 2)] = image[:, :, (2, 1, 0)]  # BGR to RGB
            image = mx.image.imresize(mx.nd.array(image), w=width, h=height, interp=3)
            image = mx.nd.image.to_tensor(image)  # 0 ~ 1 로 바꾸기
            image = mx.nd.image.normalize(image, mean=mean, std=std)

            image = image.as_in_context(ctx)
            image = image.expand_dims(axis=0)

            cls_preds, box_preds, anchors = net(image)
            if make_anchor:
                ids, scores, bboxes = prediction(cls_preds, box_preds, generated_anchors)
            else:
                ids, scores, bboxes = prediction(cls_preds, box_preds, anchors)

            bbox = bboxes[0]
            bbox[:, 0] = bbox[:, 0] * width
            bbox[:, 1] = bbox[:, 1] * height
            bbox[:, 2] = bbox[:, 2] * width
            bbox[:, 3] = bbox[:, 3] * height

            result = plot_bbox(origin_image, bbox, scores=scores[0], labels=ids[0],
                               thresh=plot_class_thresh,
                               reverse_rgb=False,
                               class_names=test_dataset.classes,
                               image_name=name)
            if video_save:
                out.write(result)
            if video_show:
                cv2.imshow(name, result)
                cv2.waitKey(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(mean= [0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
        box_sizes300=[21, 45, 101.25, 157.5, 213.75, 270, 326.25],
        box_ratios300=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0 / 3]] * 3 + [[1, 2, 0.5]] * 2,
        box_sizes512=[21, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
        box_ratios512=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0 / 3]] * 4 + [[1, 2, 0.5]] * 2,
        anchor_box_clip=True,
        use_onnx_model=False,
        video_path='test_video/C050105_001.mp4',
        load_name="400_600_ADAM_PVGG16_512",
        load_period=100, GPU_COUNT=0,
        decode_number=-1,
        nms_thresh=0.45,
        nms_topk=500,
        except_class_thresh=0.01,
        plot_class_thresh=0.5,
        video_save_path="result_video",
        video_show=False,
        video_save=True,
        make_anchor=True)
