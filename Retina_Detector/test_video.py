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
from core.model.backbone.FeaturePyramidNetworks import get_fpn_resnet

logfilepath = ""  # 따로 지정하지 않으면 terminal에 뜸
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


def anchor_generator(input_size=None, feature_size=None, anchor_size=None, anchor_size_ratios=None,
                     anchor_aspect_ratios=None,
                     box_offset=(0.5, 0.5), box_clip=True):
    norm_anchor_size_width = anchor_size / input_size[1]
    norm_anchor_size_height = anchor_size / input_size[0]
    norm_anchor_size_ratios_width = np.multiply(norm_anchor_size_width, np.array(anchor_size_ratios))
    norm_anchor_size_ratios_height = np.multiply(norm_anchor_size_height, np.array(anchor_size_ratios))

    anchors = []
    for y in range(feature_size[0]):
        for x in range(feature_size[1]):
            for nasr_w, nasr_h in zip(norm_anchor_size_ratios_width, norm_anchor_size_ratios_height):
                cy = (y + box_offset[0]) / feature_size[0]
                cx = (x + box_offset[1]) / feature_size[1]
                for asr in anchor_aspect_ratios:
                    sr = np.sqrt(asr)
                    w = nasr_w * sr
                    h = nasr_h / sr
                    anchors.append([cx, cy, w, h])
    anchors = np.array(anchors)
    if box_clip:
        anchors = np.clip(anchors, 0, 1)
    return anchors


def run(mean= [0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225] ,
        anchor_sizes= [32, 64, 128, 256, 512],
        anchor_size_ratios= [1, pow(2, 1 / 3), pow(2, 2 / 3)],
        anchor_aspect_ratios= [0.5, 1, 2],
        anchor_box_clip= True,
        use_onnx_model=False,
        video_path="",
        load_name="256_512_ADAM_PRES_18",
        load_period=100, GPU_COUNT=0,
        decode_number=5000,
        nms_thresh=0.5,
        nms_topk=500,
        except_class_thresh=0.05,
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
        num_layers = int(load_name.split("_")[-1])
        fpn_resnet = get_fpn_resnet(num_layers, ctx=ctx)
        fpn_output = fpn_resnet(
            mx.nd.random_uniform(low=0, high=1, shape=(1, 3, height, width), ctx=ctx))
        for fpn in fpn_output:
            feature_sizes.append(fpn.shape[2:])

        anchor_list = []
        for feature_size, anchor_size in zip(feature_sizes, anchor_sizes):
            anchor_list.append(anchor_generator(input_size=(height, width),
                                                feature_size=feature_size,
                                                anchor_size=anchor_size,
                                                anchor_size_ratios=anchor_size_ratios,
                                                anchor_aspect_ratios=anchor_aspect_ratios,
                                                box_clip=anchor_box_clip))
        generated_anchors = np.reshape(np.concatenate(anchor_list, axis=0), (1, -1, 4))
        generated_anchors = mx.nd.array(generated_anchors, ctx=ctx)

    net.hybridize(active=True, static_alloc=True, static_shape=True)

    # BoxEncoder, BoxDecoder 에서 같은 값을 가져야함
    prediction = Prediction(
        sigmoid=False,
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
        std= [0.229, 0.224, 0.225] ,
        anchor_sizes= [32, 64, 128, 256, 512],
        anchor_size_ratios= [1, pow(2, 1 / 3), pow(2, 2 / 3)],
        anchor_aspect_ratios= [0.5, 1, 2],
        anchor_box_clip= True,
        use_onnx_model=False,
        video_path='test_video/C050105_001.mp4',
        load_name="256_512_ADAM_PRES_18",
        load_period=100, GPU_COUNT=0,
        decode_number=5000,
        nms_thresh=0.5,
        nms_topk=500,
        except_class_thresh=0.05,
        plot_class_thresh=0.5,
        video_save_path="result_video",
        video_show=True,
        video_save=True,
        make_anchor=True)
