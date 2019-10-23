import glob
import logging
import os
import platform
import time

import cv2
import gluoncv
import mlflow as ml
import mxnet as mx
import mxnet.autograd as autograd
import mxnet.contrib.amp as amp
import mxnet.gluon as gluon
import numpy as np
from mxboard import SummaryWriter
from mxnet.contrib import onnx as onnx_mxnet
from tqdm import tqdm

from core import RetinaNet, FocalLoss, HuberLoss
from core import TargetGenerator, Prediction
from core import Voc_2007_AP
from core import check_onnx, plot_bbox
from core import traindataloader, validdataloader
from core.model.backbone.FeaturePyramidNetworks import get_fpn_resnet

logfilepath = ""
if os.path.isfile(logfilepath):
    os.remove(logfilepath)
logging.basicConfig(filename=logfilepath, level=logging.INFO)


# for multiscale
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


def run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        anchor_sizes=[32, 64, 128, 256, 512],
        anchor_size_ratios=[1, pow(2, 1 / 3), pow(2, 2 / 3)],
        anchor_aspect_ratios=[0.5, 1, 2],
        anchor_box_clip=True,
        graphviz=True,
        epoch=100,
        input_size=[256, 512],
        batch_size=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        multiscale=True,
        factor_scale=[8, 5],
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        GPU_COUNT=0,
        base=18,
        pretrained_base=True,
        pretrained_path="modelparam",
        AMP=True,

        eval_period=5,
        tensorboard=True,
        valid_graph_path="valid_Graph",
        using_mlflow=True,
        decode_number=5000,
        nms_thresh=0.5,
        nms_topk=500,
        except_class_thresh=0.05,
        plot_class_thresh=0.5):
    if GPU_COUNT == 0:
        ctx = mx.cpu(0)
        AMP = False
    elif GPU_COUNT == 1:
        ctx = mx.gpu(0)
    else:
        ctx = [mx.gpu(i) for i in range(GPU_COUNT)]

    # 운영체제 확인
    if platform.system() == "Linux":
        logging.info(f"{platform.system()} OS")
    elif platform.system() == "Windows":
        logging.info(f"{platform.system()} OS")
    else:
        logging.info(f"{platform.system()} OS")

    if isinstance(ctx, (list, tuple)):
        for i, c in enumerate(ctx):
            free_memory, total_memory = mx.context.gpu_memory_info(i)
            free_memory = round(free_memory / (1024 * 1024 * 1024), 2)
            total_memory = round(total_memory / (1024 * 1024 * 1024), 2)
            logging.info(f'Running on {c} / free memory : {free_memory}GB / total memory {total_memory}GB')
    else:
        if GPU_COUNT == 1:
            free_memory, total_memory = mx.context.gpu_memory_info(0)
            free_memory = round(free_memory / (1024 * 1024 * 1024), 2)
            total_memory = round(total_memory / (1024 * 1024 * 1024), 2)
            logging.info(f'Running on {ctx} / free memory : {free_memory}GB / total memory {total_memory}GB')
        else:
            logging.info(f'Running on {ctx}')

    if GPU_COUNT > 0 and batch_size < GPU_COUNT:
        logging.info(f" batch size must be greater than gpu number ")
        exit(0)

    if AMP:
        amp.init()

    if multiscale:
        logging.info("Using MultiScale ")
        fpn_resnet = get_fpn_resnet(base, ctx=mx.cpu())

    if data_augmentation:
        logging.info("Using Data Augmentation")

    logging.info(f"training Retina Detector")
    input_shape = (1, 3) + tuple(input_size)

    try:
        train_dataloader, train_dataset = traindataloader(multiscale=multiscale,
                                                          factor_scale=factor_scale,
                                                          augmentation=data_augmentation,
                                                          path=train_dataset_path,
                                                          image_normalization=True,
                                                          box_normalization=True,
                                                          input_size=input_size,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          shuffle=True, mean=mean, std=std)
        valid_dataloader, valid_dataset = validdataloader(path=valid_dataset_path,
                                                          image_normalization=True,
                                                          box_normalization=True,
                                                          input_size=input_size,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          shuffle=True, mean=mean, std=std)

    except Exception as E:
        print(E)
        exit(0)

    train_update_number_per_epoch = len(train_dataloader)
    if train_update_number_per_epoch < 1:
        logging.warning(" train batch size가 데이터 수보다 큼 ")
        exit(0)

    valid_list = glob.glob(os.path.join(valid_dataset_path, "*"))
    if valid_list:
        valid_update_number_per_epoch = len(valid_dataloader)
        if valid_update_number_per_epoch < 1:
            logging.warning(" valid batch size가 데이터 수보다 큼 ")
            exit(0)

    num_classes = train_dataset.num_class  # 클래스 수
    name_classes = train_dataset.classes

    optimizer = optimizer.upper()
    if pretrained_base:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_P" + "RES_" + str(base)
    else:
        model = str(input_size[0]) + "_" + str(input_size[1]) + "_" + optimizer + "_RES_" + str(base)

    weight_path = f"weights/{model}"
    param_path = os.path.join(weight_path, f'{model}-{load_period:04d}.params')

    if os.path.exists(param_path):
        start_epoch = load_period
        logging.info(f"loading {os.path.basename(param_path)} weights\n")
        net = gluon.SymbolBlock.imports(os.path.join(weight_path, f'{model}-symbol.json'),
                                        ['data'],
                                        param_path, ctx=ctx)
    else:
        start_epoch = 0
        net = RetinaNet(version=base,
                        input_size=input_size,
                        anchor_sizes=anchor_sizes,
                        anchor_size_ratios=anchor_size_ratios,
                        anchor_aspect_ratios=anchor_aspect_ratios,
                        num_classes=num_classes,  # foreground만
                        pretrained=pretrained_base,
                        pretrained_path=pretrained_path,
                        anchor_box_clip=anchor_box_clip,
                        ctx=ctx)

        if isinstance(ctx, (list, tuple)):
            net.summary(mx.nd.ones(shape=input_shape, ctx=ctx[0]))
        else:
            net.summary(mx.nd.ones(shape=input_shape, ctx=ctx))

        '''
        active (bool, default True) – Whether to turn hybrid on or off.
        static_alloc (bool, default False) – Statically allocate memory to improve speed. Memory usage may increase.
        static_shape (bool, default False) – Optimize for invariant input shapes between iterations. Must also set static_alloc to True. Change of input shapes is still allowed but slower.
        '''
        if multiscale:
            net.hybridize(active=True, static_alloc=True, static_shape=False)
        else:
            net.hybridize(active=True, static_alloc=True, static_shape=True)

    if start_epoch + 1 >= epoch + 1:
        logging.info("this model has already been optimized")
        exit(0)

    if tensorboard:
        summary = SummaryWriter(logdir=os.path.join("mxboard", model), max_queue=10, flush_secs=10,
                                verbose=False)
        if isinstance(ctx, (list, tuple)):
            net.forward(mx.nd.ones(shape=input_shape, ctx=ctx[0]))
        else:
            net.forward(mx.nd.ones(shape=input_shape, ctx=ctx))
        summary.add_graph(net)
    if graphviz:
        gluoncv.utils.viz.plot_network(net, shape=input_shape, save_prefix=model)

    # optimizer
    unit = 1 if (len(train_dataset) // batch_size) < 1 else len(train_dataset) // batch_size
    step = unit * decay_step
    lr_sch = mx.lr_scheduler.FactorScheduler(step=step, factor=decay_lr, stop_factor_lr=1e-12, base_lr=learning_rate)

    if AMP:
        '''
        update_on_kvstore : bool, default None
        Whether to perform parameter updates on kvstore. If None, then trainer will choose the more
        suitable option depending on the type of kvstore. If the `update_on_kvstore` argument is
        provided, environment variable `MXNET_UPDATE_ON_KVSTORE` will be ignored.
        '''
        if optimizer.upper() == "ADAM":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "beta1": 0.9,
                                                                                       "beta2": 0.999,
                                                                                       'multi_precision': False},
                                    update_on_kvstore=False)  # for Dynamic loss scaling
        elif optimizer.upper() == "RMSPROP":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "gamma1": 0.9,
                                                                                       "gamma2": 0.999,
                                                                                       'multi_precision': False},
                                    update_on_kvstore=False)  # for Dynamic loss scaling
        elif optimizer.upper() == "SGD":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "wd": 0.000001,
                                                                                       "momentum": 0.9,
                                                                                       'multi_precision': False},
                                    update_on_kvstore=False)  # for Dynamic loss scaling
        else:
            logging.error("optimizer not selected")
            exit(0)

        amp.init_trainer(trainer)

    else:
        if optimizer.upper() == "ADAM":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "beta1": 0.9,
                                                                                       "beta2": 0.999,
                                                                                       'multi_precision': False})
        elif optimizer.upper() == "RMSPROP":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "gamma1": 0.9,
                                                                                       "gamma2": 0.999,
                                                                                       'multi_precision': False})
        elif optimizer.upper() == "SGD":
            trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params={"learning_rate": learning_rate,
                                                                                       "lr_scheduler": lr_sch,
                                                                                       "wd": 0.0001,
                                                                                       "momentum": 0.9,
                                                                                       'multi_precision': False})

        else:
            logging.error("optimizer not selected")
            exit(0)

    '''
    localization loss -> Smooth L1 loss 
    confidence loss -> Focal 
    '''
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

    targetgenerator = TargetGenerator(foreground_iou_thresh=0.5, background_iou_thresh=0.4)

    prediction = Prediction(
        sigmoid=False,
        num_classes=num_classes,
        decode_number=decode_number,
        nms_thresh=nms_thresh,
        nms_topk=nms_topk,
        except_class_thresh=except_class_thresh)

    precision_recall = Voc_2007_AP(iou_thresh=0.5, class_names=name_classes)

    start_time = time.time()
    for i in tqdm(range(start_epoch + 1, epoch + 1, 1), initial=start_epoch + 1, total=epoch):
        conf_loss_sum = 0
        loc_loss_sum = 0

        for image, label, _ in train_dataloader:
            td_batch_size = image.shape[0]
            if GPU_COUNT <= 1:
                image = gluon.utils.split_and_load(image, [ctx])
                label = gluon.utils.split_and_load(label, [ctx])
            else:
                image = gluon.utils.split_and_load(image, ctx)
                label = gluon.utils.split_and_load(label, ctx)

            '''
            autograd 설명
            https://mxnet.apache.org/api/python/docs/tutorials/getting-started/crash-course/3-autograd.html
            '''
            with autograd.record(train_mode=True):

                cls_preds = []
                box_preds = []
                cls_targets = []
                box_targets = []

                cls_losses = []
                box_losses = []
                total_loss = []

                # gpu N 개를 대비한 코드 (Data Parallelism)
                for img, lb in zip(image, label):
                    gt_box = lb[:, :, :4]
                    gt_id = lb[:, :, 4:5]
                    cls_pred, box_pred, anchor = net(img)

                    if multiscale:
                        _, _, h, w = img.shape
                        feature_sizes = []
                        fpn_output = fpn_resnet(
                            mx.nd.random_uniform(low=0, high=1, shape=(1, 3, h, w), ctx=mx.cpu()))
                        for fpn in fpn_output:
                            feature_sizes.append(fpn.shape[2:])

                        anchor_list = []
                        for feature_size, anchor_size in zip(feature_sizes, anchor_sizes):
                            anchor_list.append(anchor_generator(input_size=(h, w),
                                                                feature_size=feature_size,
                                                                anchor_size=anchor_size,
                                                                anchor_size_ratios=anchor_size_ratios,
                                                                anchor_aspect_ratios=anchor_aspect_ratios,
                                                                box_clip=anchor_box_clip))
                        generated_anchors = np.reshape(np.concatenate(anchor_list, axis=0), (1, -1, 4))
                        generated_anchors = mx.nd.array(generated_anchors, ctx=img.context)  # ctx : 현재 device로 지정해주기

                        cls_target, box_target = targetgenerator(generated_anchors, gt_box, gt_id)
                    else:
                        cls_target, box_target = targetgenerator(anchor, gt_box, gt_id)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                    cls_targets.append(cls_target)
                    box_targets.append(box_target)

                for cls_p, box_p, cls_t, box_t in zip(cls_preds, box_preds, cls_targets, box_targets):
                    except_ignore_samples = cls_t > -1
                    positive_samples = cls_t > 0
                    positive_numbers = positive_samples.sum()

                    conf_loss = confidence_loss(cls_p, cls_t, except_ignore_samples.expand_dims(axis=-1))
                    conf_loss = mx.nd.divide(conf_loss, positive_numbers + 1)
                    cls_losses.append(conf_loss.asscalar())

                    loc_loss = localization_loss(box_p, box_t, positive_samples.expand_dims(axis=-1))
                    box_losses.append(loc_loss.asscalar())

                    total_loss.append(conf_loss + loc_loss)
                if AMP:
                    with amp.scale_loss(total_loss, trainer) as scaled_loss:
                        autograd.backward(scaled_loss)
                else:
                    autograd.backward(total_loss)

            trainer.step(batch_size=td_batch_size, ignore_stale_grad=False)

            conf_loss_sum += sum(cls_losses) / td_batch_size
            loc_loss_sum += sum(box_losses) / td_batch_size

        train_conf_loss_mean = np.divide(conf_loss_sum, train_update_number_per_epoch)
        train_loc_loss_mean = np.divide(loc_loss_sum, train_update_number_per_epoch)

        logging.info(
            f"train confidence loss : {train_conf_loss_mean} / train localization loss : {train_loc_loss_mean}")

        if i % eval_period == 0 and valid_list:

            conf_loss_sum = 0
            loc_loss_sum = 0

            # loss 구하기
            for image, label, _ in valid_dataloader:
                vd_batch_size = image.shape[0]
                if GPU_COUNT <= 1:
                    image = gluon.utils.split_and_load(image, [ctx], even_split=False)
                    label = gluon.utils.split_and_load(label, [ctx], even_split=False)
                else:
                    image = gluon.utils.split_and_load(image, ctx, even_split=False)
                    label = gluon.utils.split_and_load(label, ctx, even_split=False)

                # prediction, target space for Data Parallelism
                cls_preds = []
                box_preds = []

                cls_targets = []
                box_targets = []

                cls_losses = []
                box_losses = []

                # gpu N 개를 대비한 코드 (Data Parallelism)
                for img, lb in zip(image, label):
                    gt_box = lb[:, :, :4]
                    gt_id = lb[:, :, 4:5]
                    cls_pred, box_pred, anchor = net(img)
                    id, score, bbox = prediction(cls_pred, box_pred, anchor)

                    precision_recall.update(pred_bboxes=bbox,
                                            pred_labels=id,
                                            pred_scores=score,
                                            gt_boxes=gt_box,
                                            gt_labels=gt_id)

                    cls_target, box_target = targetgenerator(anchor, gt_box, gt_id)
                    cls_preds.append(cls_pred)
                    box_preds.append(box_pred)
                    cls_targets.append(cls_target)
                    box_targets.append(box_target)

                for cls_p, box_p, cls_t, box_t in zip(cls_preds, box_preds, cls_targets, box_targets):
                    except_ignore_samples = cls_t > -1
                    positive_samples = cls_t > 0
                    positive_numbers = positive_samples.sum()

                    conf_loss = confidence_loss(cls_p, cls_t, except_ignore_samples.expand_dims(axis=-1))
                    conf_loss = mx.nd.divide(conf_loss, positive_numbers + 1)
                    cls_losses.append(conf_loss.asscalar())

                    loc_loss = localization_loss(box_p, box_t, positive_samples.expand_dims(axis=-1))
                    box_losses.append(loc_loss.asscalar())

                conf_loss_sum += sum(cls_losses) / vd_batch_size
                loc_loss_sum += sum(box_losses) / vd_batch_size

            valid_conf_loss_mean = np.divide(conf_loss_sum, valid_update_number_per_epoch)
            valid_loc_loss_mean = np.divide(loc_loss_sum, valid_update_number_per_epoch)
            logging.info(
                f"valid confidence loss : {valid_conf_loss_mean} / valid localization loss : {valid_loc_loss_mean}")

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
                                          AP=AP_appender, mAP=mAP_result, folder_name=valid_graph_path, epoch=i)
            precision_recall.reset()

            if tensorboard:
                # gpu N 개를 대비한 코드 (Data Parallelism)
                dataloader_iter = iter(valid_dataloader)
                image, label, _ = next(dataloader_iter)
                if GPU_COUNT <= 1:
                    image = gluon.utils.split_and_load(image, [ctx], even_split=False)
                    label = gluon.utils.split_and_load(label, [ctx], even_split=False)
                else:
                    image = gluon.utils.split_and_load(image, ctx, even_split=False)
                    label = gluon.utils.split_and_load(label, ctx, even_split=False)

                ground_truth_colors = {}
                for k in range(num_classes):
                    ground_truth_colors[k] = (0, 0, 255)

                batch_image = []
                for img, lb in zip(image, label):
                    gt_boxes = lb[:, :, :4]
                    gt_ids = lb[:, :, 4:5]
                    cls_pred, box_pred, anchor = net(img)
                    ids, scores, bboxes = prediction(cls_pred, box_pred, anchor)

                    for ig, gt_id, gt_box, id, score, bbox in zip(img, gt_ids, gt_boxes, ids, scores, bboxes):
                        ig = np.transpose(ig.asnumpy(), axes=(1, 2, 0))
                        ig = ig * np.array(std) + np.array(mean)
                        ig = (ig * 255).clip(0, 255)

                        gt_box[:, 0] = gt_box[:, 0] * input_size[1]
                        gt_box[:, 1] = gt_box[:, 1] * input_size[0]
                        gt_box[:, 2] = gt_box[:, 2] * input_size[1]
                        gt_box[:, 3] = gt_box[:, 3] * input_size[0]

                        bbox[:, 0] = bbox[:, 0] * input_size[1]
                        bbox[:, 1] = bbox[:, 1] * input_size[0]
                        bbox[:, 2] = bbox[:, 2] * input_size[1]
                        bbox[:, 3] = bbox[:, 3] * input_size[0]

                        # ground truth box 그리기
                        ground_truth = plot_bbox(ig, gt_box, scores=None, labels=gt_id, thresh=None,
                                                 reverse_rgb=True,
                                                 class_names=valid_dataset.classes, absolute_coordinates=True,
                                                 colors=ground_truth_colors)
                        # prediction box 그리기
                        prediction_box = plot_bbox(ground_truth, bbox, scores=score, labels=id,
                                                   thresh=plot_class_thresh,
                                                   reverse_rgb=False,
                                                   class_names=valid_dataset.classes, absolute_coordinates=True)

                        # Tensorboard에 그리기 위해 BGR -> RGB / (height, width, channel) -> (channel, height, width) 를한다.
                        prediction_box = cv2.cvtColor(prediction_box, cv2.COLOR_BGR2RGB)
                        prediction_box = np.transpose(prediction_box,
                                                      axes=(2, 0, 1))
                        batch_image.append(prediction_box)  # (batch, channel, height, width)

                summary.add_image(tag="valid_result", image=np.array(batch_image), global_step=i)
                summary.add_scalar(tag="conf_loss", value={"train_conf_loss": train_conf_loss_mean,
                                                           "valid_conf_loss": valid_conf_loss_mean}, global_step=i)
                summary.add_scalar(tag="loc_loss",
                                   value={"train_loc_loss": train_loc_loss_mean, "valid_loc_loss": valid_loc_loss_mean},
                                   global_step=i)
                summary.add_scalar(tag="total_loss", value={
                    "train_total_loss": train_conf_loss_mean + train_loc_loss_mean,
                    "valid_total_loss": valid_conf_loss_mean + valid_loc_loss_mean}, global_step=i)

                params = net.collect_params().values()
                if GPU_COUNT > 1:
                    for c in ctx:
                        for p in params:
                            summary.add_histogram(tag=p.name, values=p.data(ctx=c), global_step=i, bins='default')
                else:
                    for p in params:
                        summary.add_histogram(tag=p.name, values=p.data(), global_step=i, bins='default')

        if i % save_period == 0:

            if not os.path.exists(weight_path):
                os.makedirs(weight_path)

            '''
            Hybrid models can be serialized as JSON files using the export function
            Export HybridBlock to json format that can be loaded by SymbolBlock.imports, mxnet.mod.Module or the C++ interface.
            When there are only one input, it will have name data. When there Are more than one inputs, they will be named as data0, data1, etc.
            '''
            try:
                net.export(os.path.join(weight_path, f"{model}"), epoch=i, remove_amp_cast=True)  # for onnx
                if AMP:
                    net.export(os.path.join(weight_path, f"{model}_AMP"), epoch=i,
                               remove_amp_cast=False)  # for mxnet c++(float16)
            except Exception as E:
                logging.error(f"json, param model export 예외 발생 : {E}")
            else:
                logging.error(f"json, param model export 성공")

            try:
                sym = os.path.join(weight_path, f'{model}-symbol.json')
                params = os.path.join(weight_path, f'{model}-{i:04d}.params')
                onnx_path = os.path.join(weight_path, f"{model}.onnx")

                onnx_mxnet.export_model(sym=sym, params=params, input_shape=[input_shape], input_type=np.float32,
                                        onnx_file_path=onnx_path, verbose=False)
            except Exception as E:
                logging.error(f"ONNX model export 예외 발생 : {E}")
            else:
                logging.error(f"ONNX model export 성공")

            try:
                check_onnx(onnx_path)
                logging.info(f"{os.path.basename(onnx_path)} saved completed")
            except Exception as E:
                logging.error(f"ONNX model check 예외 발생 : {E}")
            else:
                logging.info("ONNX model check completed")

    end_time = time.time()
    learning_time = end_time - start_time
    logging.info(f"learning time : 약, {learning_time / 3600:0.2f}H")
    logging.info("optimization completed")

    if using_mlflow:
        ml.log_metric("learning time", round(learning_time / 3600, 2))


if __name__ == "__main__":
    run(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        anchor_sizes=[32, 64, 128, 256, 512],
        anchor_size_ratios=[1, pow(2, 1 / 3), pow(2, 2 / 3)],
        anchor_aspect_ratios=[0.5, 1, 2],
        anchor_box_clip=True,
        graphviz=False,
        epoch=10,
        input_size=[256, 512],
        batch_size=4,
        train_dataset_path="Dataset/train",
        valid_dataset_path="Dataset/valid",
        data_augmentation=True,
        num_workers=4,
        optimizer="ADAM",
        save_period=5,
        load_period=10,
        learning_rate=0.001, decay_lr=0.999, decay_step=10,
        GPU_COUNT=0,
        base=18,
        pretrained_base=True,
        pretrained_path="modelparam",
        AMP=True,
        eval_period=5,
        tensorboard=True,
        valid_graph_path="valid_Graph",
        using_mlflow=True,
        decode_number=5000,
        nms_thresh=0.5,
        nms_topk=500,
        except_class_thresh=0.05,
        plot_class_thresh=0.5)
