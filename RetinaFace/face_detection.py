import cv2
import sys
import numpy as np
import os
from retinaface import RetinaFace
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_align


target_size = 1024
max_size = 1600

parser = argparse.ArgumentParser(description='Align Images using RetinaFace')
# general
parser.add_argument('--data-dir', default='', help='')
parser.add_argument('--image-size', type=int, default=224, help='')
parser.add_argument('--gpu', type=int, default=0, help='')
parser.add_argument('--det-prefix', type=str, default='../models/RetinaFace/R50', help='')
parser.add_argument('--output', default='./', help='path to save.')
parser.add_argument('--align-mode', default='normal', help='align mode.')
args = parser.parse_args()

detector = RetinaFace(args.det_prefix, 0, args.gpu, 'net3')


def adjust_bbox(bbox, im_shape, offset):
    if int(bbox[3]) - int(bbox[1]) > int(bbox[2]) - int(bbox[0]):
        diff = ((int(bbox[3]) - int(bbox[1])) - (int(bbox[2]) - int(bbox[0]))) // 2

        top = max(int(bbox[1]), 0)
        bottom = min(int(bbox[3]), im_shape[0])
        left = max(int(bbox[0]) - diff, 0)
        right = min(int(bbox[2]) + diff, im_shape[1])
    else:
        diff = ((int(bbox[2]) - int(bbox[0])) - (int(bbox[3]) - int(bbox[1]))) // 2

        top = max(int(bbox[1]) - diff, 0)
        bottom = min(int(bbox[3]) + diff, im_shape[0])
        left = max(int(bbox[0]), 0)
        right = min(int(bbox[2]), im_shape[1])

    top -= (int((bottom - top) * offset))
    bottom += (int((bottom - top) * offset))
    left -= (int((right - left) * offset))
    right += (int((right - left) * offset))

    return top, bottom, left, right


def get_norm_crop(image_path):
    im = cv2.imread(image_path)
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    bbox, landmark = detector.detect(im, threshold=0.5, scales=[im_scale])

    if bbox.shape[0] == 0:
        bbox, landmark = detector.detect(im, threshold=0.05, scales=[im_scale * 0.75, im_scale, im_scale * 2.0])
        # print('refine', im.shape, bbox.shape, landmark.shape)
    nrof_faces = bbox.shape[0]
    if nrof_faces > 0:
        det = bbox[:, 0:4]
        img_size = np.asarray(im.shape)[0:2]
        bindex = 0
        if nrof_faces > 1:
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering

        _bbox = det[bindex, 0:4]

        top, bottom, left, right = adjust_bbox(_bbox, im_shape, 0.1)

        cropped = im[top:bottom, left:right]
        cropped = cv2.resize(cropped, (args.image_size, args.image_size))

        _landmark = landmark[bindex]
        warped = face_align.norm_crop(im, landmark=_landmark, image_size=args.image_size, mode=args.align_mode)
        return warped, cropped
    else:
        return None


aligned, cropped = get_norm_crop('X000001.jpg')
cv2.imwrite('aligned.jpg', aligned)
cv2.imwrite('cropped.jpg', cropped)
