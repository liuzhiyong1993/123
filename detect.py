# ----------------------------------------------------------------------------------------------------------------------
# 检测接口
# ----------------------------------------------------------------------------------------------------------------------

from utils.datasets import *
import argparse
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords

# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='./runs/train/exp/weights/best.pt', help='path to weights file')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
opt = parser.parse_args()
print(opt)


# ----------------------------------------------------------------------------------------------------------------------

def bbox_rel(image_width, image_height,  *xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


class Yolo():
    def __init__(self):
        self.writer = None
        self.prepare()

    def prepare(self):
        global model, device, classes, colors, names
        device = select_device(device='0')

        model = torch.load(opt.weights, map_location=device)['model'].float()

        model.to(device).eval()

        names = model.names if hasattr(model, 'names') else model.modules.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    def detect(self, frame):
        im0 = frame
        img = letterbox(frame, new_shape=416)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        boxes = []
        bbox_xywh = []
        classes = []
        confs = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, score, cls in det:
                    label = '%s ' % (names[int(cls)])

                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]

                    if names[int(cls)] in ['car', 'truck', 'bus']:
                        bbox_xywh.append(obj)

                        if names[int(cls)] == 'car':
                            classes.append(0)
                        elif names[int(cls)] == 'truck':
                            classes.append(1)
                        elif names[int(cls)] == 'bus':
                            classes.append(2)

                        confs.append([score.item()])
                        boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(score), int(cls), label])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                classess = torch.Tensor(classes)

        return boxes, xywhs, confss, classess


# ----------------------------------------------------------------------------------------------------------------------

yolo = Yolo()


def recognition(frame):
    boxes = yolo.detect(frame)
    return boxes

# ----------------------------------------------------------------------------------------------------------------------

