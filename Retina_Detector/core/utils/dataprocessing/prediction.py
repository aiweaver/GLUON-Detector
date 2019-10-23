import mxnet as mx
from mxnet.gluon import Block

from core.utils.dataprocessing.predictFunction.decoder import ClassDecoder, BoxDecoder, BoxDecodeLimit


class Prediction(Block):

    def __init__(self,
                 sigmoid=False,
                 means=(0., 0., 0., 0.),
                 stds=(0.1, 0.1, 0.2, 0.2),
                 num_classes=3,
                 decode_number=5000,
                 nms_thresh=0.5,
                 nms_topk=500,
                 except_class_thresh=0.05):  # 논문
        super(Prediction, self).__init__()

        self._classdecoder = ClassDecoder(num_classes=num_classes, thresh=except_class_thresh, sigmoid=sigmoid)
        self._boxdecodelimit = BoxDecodeLimit(decode_number=decode_number)
        self._boxdecoder = BoxDecoder(stds=stds, means=means)
        self._decode_number = decode_number
        self._num_classes = num_classes
        self._nms_thresh = nms_thresh
        self._nms_topk = nms_topk

    def forward(self, cls_preds, box_preds, anchors):
        F = mx.nd
        class_ids, class_scores = self._classdecoder(cls_preds)
        class_ids, class_scores, box_preds, anchors = self._boxdecodelimit(class_ids, class_scores, box_preds, anchors)
        box_predictions = self._boxdecoder(box_preds, anchors)

        # 클래스 각각에 대한 결과
        results = []
        for i in range(self._num_classes):
            class_id = class_ids.slice_axis(axis=-1, begin=i, end=i + 1)
            class_score = class_scores.slice_axis(axis=-1, begin=i, end=i + 1)
            per_result = F.concat(*[class_id, class_score, box_predictions], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)

        if self._nms_thresh > 0 and self._nms_thresh < 1:
            '''
            Apply non-maximum suppression to input.
            The output will be sorted in descending order according to score. 
            Boxes with overlaps larger than overlap_thresh, smaller scores and background boxes will be removed and filled with -1, 
            '''
            result = F.contrib.box_nms(
                result,
                overlap_thresh=self._nms_thresh,
                topk=self._nms_topk,
                id_index=0, score_index=1, coord_start=2,
                force_suppress=False, in_format="corner", out_format="corner")

        ids = F.slice_axis(result, axis=-1, begin=0, end=1)
        scores = F.slice_axis(result, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        return ids, scores, bboxes


# test
if __name__ == "__main__":
    from core import RetinaNet, DetectionDataset
    import os

    input_size = (512, 512)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset = DetectionDataset(path=os.path.join(root, 'Dataset', 'train'), input_size=input_size,
                               image_normalization=True,
                               box_normalization=True)
    num_classes = dataset.num_class
    image, label, _ = dataset[0]

    net = RetinaNet(version=18,
                    input_size=input_size,
                    anchor_sizes=[32, 64, 128, 256, 512],
                    anchor_size_ratios=[1, pow(2, 1 / 3), pow(2, 2 / 3)],
                    anchor_aspect_ratios=[0.5, 1, 2],
                    num_classes=num_classes,  # foreground만
                    pretrained=True,
                    pretrained_path=os.path.join(root, "modelparam"),
                    anchor_box_offset=(0.5, 0.5),
                    anchor_box_clip=True,
                    ctx=mx.cpu())

    net.hybridize(active=True, static_alloc=True, static_shape=True)

    pred = Prediction(
        sigmoid=False,
        means=(0., 0., 0., 0.),
        stds=(0.1, 0.1, 0.2, 0.2),
        num_classes=num_classes,
        decode_number=1000,
        nms_thresh=0.5,
        nms_topk=100,
        except_class_thresh=0.05)

    data = image.as_in_context(mx.cpu(0))
    label = label.as_in_context(mx.cpu(0))

    # batch 형태로 만들기
    data = data.expand_dims(axis=0)
    label = label.expand_dims(axis=0)
    gt_boxes = label[:, :, :4]
    gt_ids = label[:, :, 4:5]
    cls_preds, box_preds, anchors = net(data)
    ids, scores, bboxes = pred(cls_preds, box_preds, anchors)

    print(f"nms class id shape : {ids.shape}")
    print(f"nms class scores shape : {scores.shape}")
    print(f"nms box predictions shape : {bboxes.shape}")
    '''
    nms class id shape : (1, 5000, 1)
    nms class scores shape : (1, 5000, 1)
    nms box predictions shape : (1, 5000, 4)
    '''
