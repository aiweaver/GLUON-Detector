import logging
import os
import platform

import mxnet as mx
import mxnet.gluon as gluon
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
from tqdm import tqdm

from core import FocalLoss, HuberLoss
from core import TargetGenerator, Prediction
from core import Voc_2007_AP
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
        load_name="256_512_ADAM_PRES_18", load_period=10, GPU_COUNT=0,
        test_dataset_path="Dataset/test",
        test_save_path="result",
        test_graph_path="test_Graph",
        num_workers=4,
        show_flag=True,
        save_flag=True,
        decode_number=5000,
        nms_thresh=0.5,
        nms_topk=500,
        except_class_thresh=0.05,
        plot_class_thresh=0.5,
        make_anchor=True):
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
        test_dataloader, test_dataset = testdataloader(path=test_dataset_path,
                                                       image_normalization=True,
                                                       box_normalization=True,
                                                       input_size=(height, width),
                                                       batch_size=1,
                                                       num_workers=num_workers)
    except Exception:
        logging.info("The dataset does not exist")
        exit(0)

    weight_path = f"weights/{load_name}"
    symbol_path = os.path.join(weight_path, '{}-symbol.json'.format(load_name))
    param_path = os.path.join(weight_path, '{0}-{1:04d}.params'.format(load_name, load_period))
    onnx_path = os.path.join(weight_path, "{}.onnx".format(load_name))

    test_update_number_per_epoch = len(test_dataloader)
    if test_update_number_per_epoch < 1:
        logging.warning(" test batch size가 데이터 수보다 큼 ")
        exit(0)

    num_classes = test_dataset.num_class  # 클래스 수
    name_classes = test_dataset.classes

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

    confidence_loss = FocalLoss(alpha=0.25,  # 논문에서 가장 좋다고 한 숫자
                                gamma=2,  # 논문에서 가장 좋다고 한 숫자
                                sparse_label=True,
                                from_logits=False,
                                batch_axis=None,
                                num_class=num_classes,
                                reduction="sum",
                                exclude=False)

    localization_loss = HuberLoss(rho=1,
                                  batch_axis=None,
                                  reduction="sum",
                                  exclude=False)

    targetgenerator = TargetGenerator(foreground_iou_thresh=plot_class_thresh, background_iou_thresh=0.4)

    prediction = Prediction(
        sigmoid=False,
        num_classes=num_classes,
        decode_number=decode_number,
        nms_thresh=nms_thresh,
        nms_topk=nms_topk,
        except_class_thresh=except_class_thresh)

    precision_recall = Voc_2007_AP(iou_thresh=0.5, class_names=name_classes)

    ground_truth_colors = {}
    for i in range(num_classes):
        ground_truth_colors[i] = (0, 0, 255)

    conf_loss_sum = 0
    loc_loss_sum = 0

    for image, label, name in tqdm(test_dataloader):
        image = image.as_in_context(ctx)
        label = label.as_in_context(ctx)
        gt_boxes = label[:, :, :4]
        gt_ids = label[:, :, 4:5]

        cls_preds, box_preds, anchors = net(image)
        if make_anchor:
            ids, scores, bboxes = prediction(cls_preds, box_preds, generated_anchors)
        else:
            ids, scores, bboxes = prediction(cls_preds, box_preds, anchors)

        precision_recall.update(pred_bboxes=bboxes,
                                pred_labels=ids,
                                pred_scores=scores,
                                gt_boxes=gt_boxes,
                                gt_labels=gt_ids)

        img = image[0].transpose(
            (1, 2, 0)) * mx.nd.array(std, ctx=ctx) + mx.nd.array(mean, ctx=ctx)
        img = (img * 255).clip(0, 255)

        gt_box = gt_boxes[0]
        bbox = bboxes[0]
        gt_box[:, 0] = gt_box[:, 0] * width
        gt_box[:, 1] = gt_box[:, 1] * height
        gt_box[:, 2] = gt_box[:, 2] * width
        gt_box[:, 3] = gt_box[:, 3] * height
        bbox[:, 0] = bbox[:, 0] * width
        bbox[:, 1] = bbox[:, 1] * height
        bbox[:, 2] = bbox[:, 2] * width
        bbox[:, 3] = bbox[:, 3] * height

        ground_truth = plot_bbox(img, gt_box, scores=None, labels=gt_ids[0], thresh=None,
                                 reverse_rgb=True,
                                 class_names=test_dataset.classes, absolute_coordinates=True,
                                 colors=ground_truth_colors)
        plot_bbox(ground_truth, bbox, scores=scores[0], labels=ids[0], thresh=plot_class_thresh,
                  reverse_rgb=False,
                  class_names=test_dataset.classes, absolute_coordinates=True,
                  image_show=show_flag, image_save=save_flag, image_save_path=test_save_path, image_name=name[0])

        cls_targets, box_targets = targetgenerator(anchors, gt_boxes, gt_ids)
        except_ignore_samples = cls_targets > -1
        positive_samples = cls_targets > 0
        positive_numbers = positive_samples.sum()

        conf_loss = confidence_loss(cls_preds, cls_targets, except_ignore_samples.expand_dims(axis=-1))
        conf_loss = mx.nd.divide(conf_loss, positive_numbers + 1)
        conf_loss_sum += conf_loss.asscalar()

        loc_loss = localization_loss(box_preds, box_targets, positive_samples.expand_dims(axis=-1))
        loc_loss_sum += loc_loss.asscalar()

    # epoch 당 평균 loss
    test_conf_loss_mean = np.divide(conf_loss_sum, test_update_number_per_epoch)
    test_loc_loss_mean = np.divide(loc_loss_sum, test_update_number_per_epoch)

    logging.info(
        f"test confidence loss : {test_conf_loss_mean} / test localization loss : {test_loc_loss_mean}")

    AP_appender = []
    round_position = 2
    class_name, precision, recall, true_positive, false_positive, threshold = precision_recall.get_PR_list()
    for j, c, p, r in zip(range(len(recall)), class_name, precision, recall):
        name, AP = precision_recall.get_AP(c, p, r)
        logging.info(f"class {j}'s {name} AP : {round(AP * 100, round_position)}%")
        AP_appender.append(AP)
    mAP_result = np.mean(AP_appender)

    logging.info(f"mAP : {round(mAP_result * 100, round_position)}%")
    precision_recall.get_PR_curve(name=class_name,
                                  precision=precision,
                                  recall=recall,
                                  threshold=threshold,
                                  AP=AP_appender, mAP=mAP_result, folder_name=test_graph_path)


if __name__ == "__main__":
    run(mean= [0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225] ,
        anchor_sizes= [32, 64, 128, 256, 512],
        anchor_size_ratios= [1, pow(2, 1 / 3), pow(2, 2 / 3)],
        anchor_aspect_ratios= [0.5, 1, 2],
        anchor_box_clip= True,
        use_onnx_model=False,
        load_name="256_512_ADAM_PRES_18", load_period=100, GPU_COUNT=0,
        test_dataset_path="Dataset/test",
        test_save_path="result",
        test_graph_path="test_Graph",
        num_workers=4,
        show_flag=True,
        save_flag=True,
        decode_number=5000,
        nms_thresh=0.5,
        nms_topk=500,
        except_class_thresh=0.05,
        plot_class_thresh=0.5,
        make_anchor=True)
