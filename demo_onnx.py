from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np

import sys
sys.path.append("./lib")
import time

# import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform
import torch.onnx
import onnx
import onnxruntime

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

box_session = onnxruntime.InferenceSession("box_model.onnx")
hrnet_session = onnxruntime.InferenceSession("hrnet.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_person_detection_boxes(box_session, img, threshold=0.5):
    # global box_session
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    ort_inputs = {box_session.get_inputs()[0].name: to_numpy(transformed_img.to(CTX).unsqueeze_(0))}
    ort_outs = box_session.run(None, ort_inputs)
    onnx_preds = [{'boxes': torch.tensor(ort_outs[0]).to(CTX), 'labels': torch.tensor(ort_outs[1]).to(CTX), 'scores': torch.tensor(ort_outs[2]).to(CTX)}]
    pred = onnx_preds
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

    person_boxes = []
    # Select box has score larger than threshold and is person
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)

    return person_boxes


def get_pose_estimation_prediction(hrnet_session, image, centers, scales, transform):
    # global hrnet_session
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, (288, 384))
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(288), int(384)),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    
    model_inputs = torch.stack(model_inputs)
    
    hrnet_input = {'input': to_numpy(model_inputs.to(CTX))}
    ort_outs = hrnet_session.run(None, hrnet_input)
    output = torch.tensor(ort_outs[0])
    coords, _ = get_final_preds(
        True,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='./output/')
    parser.add_argument('--inferenceFps', type=int, default=1)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    global hrnet_session, box_session
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # args = parse_args()
    # update_config(cfg, args)
    # pose_dir = prepare_output_dirs(args.outputDir)
    # csv_output_rows = []
    

    # Loading an video
    # vidcap = cv2.VideoCapture(args.videoFile)
    # fps = vidcap.get(cv2.CAP_PROP_FPS)
    # if fps < args.inferenceFps:
    #     print('desired inference fps is '+str(args.inferenceFps)+' but video fps is '+str(fps))
    #     exit()
    # skip_frame_cnt = round(fps / args.inferenceFps)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]),
    #                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(skip_frame_cnt), (frame_width, frame_height))

    count = 0
    while vidcap.isOpened():
        
        total_now = time.time()
        # ret, image_bgr = vidcap.read()
        # count += 1

        # if not ret:
        #     continue

        # if count % skip_frame_cnt != 0:
        #     continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Clone 2 image for person detection and pose estimation
        # if cfg.DATASET.COLOR_RGB:
        #     image_per = image_rgb.copy()
        #     image_pose = image_rgb.copy()
        # else:
        #     image_per = image_bgr.copy()
        #     image_pose = image_bgr.copy()

        # Clone 1 image for debugging purpose
        image_per = image_rgb.copy()
        image_pose = image_rgb.copy()
        image_debug = image_rgb.copy()

        # object detection box
        now = time.time()
        pred_boxes = get_person_detection_boxes(box_session, image_per, threshold=0.9)
        # print(pred_boxes)
        then = time.time()
        print("Find person bbox in: {} sec".format(then - now))

        # Can not find people. Move to next frame
        if not pred_boxes:
            count += 1
            continue

        # if args.writeBoxFrames:
        #     for box in pred_boxes:
        #         print(box[0])
        #         # cv2.rectangle(frame, (int(box[0][0]), int(box[0][1])), (int(x + w), int(y + h)), (0, 255, 0), 2)
        #         cv2.rectangle(image_debug, (int(box[0][0]), int(box[0][1])), (int(box[0][0] + box[1][0]), int(box[1][0] + box[1][1])), (0, 255, 0), 2)
        #         # cv2.rectangle(image_debug, int(box[0]), int(box[1]), color=(0, 255, 0),
                #               thickness=3)  # Draw Rectangle with the coordinates

        # pose estimation : for multiple people
        centers = []
        scales = []
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, 288, 384)
            centers.append(center)
            scales.append(scale)

        now = time.time()
        pose_preds = get_pose_estimation_prediction(hrnet_session, image_pose, centers, scales, transform=pose_transform)
        then = time.time()
        print("Find person pose in: {} sec".format(then - now))

        new_csv_row = []
        for coords in pose_preds:
            # Draw each point on image
            for coord in coords:
                x_coord, y_coord = int(coord[0]), int(coord[1])
                cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
                new_csv_row.extend([x_coord, y_coord])

        total_then = time.time()

        text = "{:03.2f} sec".format(total_then - total_now)
        cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("pos", image_debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # csv_output_rows.append(new_csv_row)
        # img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
        # cv2.imwrite(img_file, image_debug)
        # outcap.write(image_debug)


    # write csv
    csv_headers = ['frame']
    for keypoint in COCO_KEYPOINT_INDEXES.values():
        csv_headers.extend([keypoint+'_x', keypoint+'_y'])

    csv_output_filename = os.path.join(args.outputDir, 'pose-data.csv')
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)

    vidcap.release()
    outcap.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
