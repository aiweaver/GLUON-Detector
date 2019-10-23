# gluoncv에 있는 코드 참고

"""Encoder functions.
Encoders are used during training, which assign training targets.
"""
import mxnet as mx
import mxnet.gluon as gluon


class BBoxCornerToCenter(gluon.Block):
    def __init__(self, axis=-1):
        super(BBoxCornerToCenter, self).__init__()
        self._axis = axis

    def forward(self, x):
        F = mx.nd
        xmin, ymin, xmax, ymax = F.split(x, axis=self._axis, num_outputs=4)
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + width / 2
        y_center = ymin + height / 2
        return x_center, y_center, width, height


class BoxEncoder(gluon.Block):
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(BoxEncoder, self).__init__()
        self._stds = stds
        self._means = means
        self._corner_to_center = BBoxCornerToCenter(axis=-1)

    def forward(self, matches, samples, anchors, gt_boxes):
        F = mx.nd
        gt_boxes = F.repeat(gt_boxes.reshape((0, 1, -1, 4)), repeats=matches.shape[1], axis=1)
        gt_boxes = F.split(gt_boxes, axis=-1, num_outputs=4, squeeze_axis=True)
        gt_boxed_element = [F.pick(gt_boxes[i], matches, axis=-1) for i in range(4)]
        gt_boxes = F.stack(*gt_boxed_element, axis=-1)
        gt_box_x, gt_box_y, gt_box_w, gt_box_h = self._corner_to_center(gt_boxes)
        anchor_x, anchor_y, anchor_w, anchor_h = self._corner_to_center(anchors)  #
        norm_x = (F.divide(F.subtract(gt_box_x, anchor_x), anchor_w) - self._means[0]) / self._stds[0]
        norm_y = (F.divide(F.subtract(gt_box_y, anchor_y), anchor_h) - self._means[1]) / self._stds[1]
        norm_w = (F.log(F.divide(gt_box_w, anchor_w)) - self._means[2]) / self._stds[2]
        norm_h = (F.log(F.divide(gt_box_h, anchor_h)) - self._means[3]) / self._stds[3]
        box_ids = F.concat(norm_x, norm_y, norm_w, norm_h, dim=-1)
        samples_repeat = F.repeat(samples.reshape((0, -1, 1)), repeats=4, axis=-1) > 0
        targets = F.where(samples_repeat, box_ids, F.zeros_like(box_ids))
        return targets


class ClassEncoder(gluon.Block):
    def __init__(self):
        super(ClassEncoder, self).__init__()

    def forward(self, matches, samples, gt_ids):
        F = mx.nd
        gt_ids = F.repeat(gt_ids.reshape((0, 1, -1)), axis=1, repeats=matches.shape[1])
        target_ids = F.pick(gt_ids, matches, axis=-1) + 1  # background 고려
        targets = F.where(samples > 0, target_ids, F.ones_like(target_ids) * -1)
        targets = F.where(samples < 0, F.zeros_like(targets), targets)
        return targets  # foreground class + background class 가 되어서 출력


# test
if __name__ == "__main__":
    from core import SSD_VGG16, DetectionDataset
    from core import MatchSampler, BBoxCenterToCorner
    import os

    input_size = (512, 512)
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), input_size=input_size,
                               image_normalization=True,
                               box_normalization=True)
    num_classes = dataset.num_class
    image, label, _ = dataset[0]

    net = SSD_VGG16(version=512, input_size=input_size,
                    # box_sizes=[21, 45, 101.25, 157.5, 213.75, 270, 326.25],
                    box_sizes=[21, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
                    # box_ratios=[[1, 2, 0.5]] +  # conv4_3
                    #           [[1, 2, 0.5, 3, 1.0 / 3]] * 3 +  # conv7, conv8_2, conv9_2
                    #           [[1, 2, 0.5]] * 2,  # conv10_2, conv11_2
                    box_ratios=[[1, 2, 0.5]] +  # conv4_3
                               [[1, 2, 0.5, 3, 1.0 / 3]] * 4 +  # conv7, conv8_2, conv9_2, conv10_2
                               [[1, 2, 0.5]] * 2,  # conv11_2, conv12_2
                    num_classes=num_classes,
                    pretrained=True,
                    pretrained_path=os.path.join(root, "modelparam"),
                    anchor_box_offset=(0.5, 0.5),
                    anchor_box_clip=True,
                    ctx=mx.cpu())

    net.hybridize(active=True, static_alloc=True, static_shape=True)
    # MatchSampler안에 BBoxCenterToCorner 를 집어넣으면 backward가 안됨. -> 매우 이상한 오류인듯
    centertocorner = BBoxCenterToCorner(axis=-1)
    matchsampler = MatchSampler(foreground_iou_thresh=0.5)
    classEncoder = ClassEncoder()
    boxEncoder = BoxEncoder(stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.))

    data = image.as_in_context(mx.cpu(0))
    label = label.as_in_context(mx.cpu(0))

    # batch 형태로 만들기
    data = data.expand_dims(axis=0)
    label = label.expand_dims(axis=0)

    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    _, _, anchors = net(data)

    anchors = centertocorner(anchors)
    matches, samples = matchsampler(anchors, gt_boxes)
    cls_targets = classEncoder(matches, samples, gt_ids)
    box_targets = boxEncoder(matches, samples, anchors, gt_boxes)

    print(f"cls_targets shape : {cls_targets.shape}")
    print(f"box_targets shape : {box_targets.shape}")
    '''
    cls_targets shape : (1, 24564)
    box_targets shape : (1, 24564, 4)
    '''
