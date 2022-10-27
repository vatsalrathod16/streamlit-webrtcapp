import asyncio
import logging
import queue
from this import d
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional
import subprocess

import os
import os.path
import shutil
import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
import time
from utils.transforms import get_affine_transform
from core.inference import get_final_preds
#from config import cfg
#from config import update_config

from aiortc.contrib.media import MediaPlayer


import onnxruntime
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch.nn as nn
import torch.nn.init as init
from PIL import Image
import torchvision.transforms as transforms
from skimage.transform import rescale, resize, downscale_local_mean



import argparse
import os

import cv2
import numpy as np
from loguru import logger

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="bytetrack_s.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--video_path",
        type=str,
        default='../../videos/palace.mp4',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.1,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-n",
        "--nms_thr",
        type=float,
        default=0.7,
        help="NMS threshould.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="608,1088",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser



def main():
    st.header("WebRTC demo")

    pages = {
        "Real time human detection (sendrecv)": app_object_detection,
        "Real time object detection (sendrecv)": app_object_detection,
        "Real time video transform with simple OpenCV filters (sendrecv)": app_video_filters,  # noqa: E501
        "Real time audio filter (sendrecv)": app_audio_filter,
        "Delayed echo (sendrecv)": app_delayed_echo,
        "Consuming media files on server-side and streaming it to browser (recvonly)": app_streaming,  # noqa: E501
        "Consuming media files on server-side and streaming it to browser ByteTrack (recvonly)": app_streaming_bytetrack,  # noqa: E501
        "Consuming media files on server-side and streaming it to browser ByteTrack class (recvonly)": app_streaming_classBytetrack,
        "Poseestimation(recvonly)": app_streaming_poseestimation,
        "VITPOSE (recvonly)": app_streaming_vitpose,
        "VITPOSE2 (recvonly)": app_streaming_vitpose2,
        "MMPOSE (recvonly)": app_streaming_mmpose,
        "ONNX Testing" : onnx_test,
        "WebRTC is sendonly and images are shown via st.image() (sendonly)": app_sendonly_video,  # noqa: E501
        "WebRTC is sendonly and audio frames are visualized with matplotlib (sendonly)": app_sendonly_audio,  # noqa: E501
        "Simple video and audio loopback (sendrecv)": app_loopback,
        "Configure media constraints and HTML element styles with loopback (sendrecv)": app_media_constraints,  # noqa: E501
        "Control the playing state programatically": app_programatically_play,
        "Customize UI texts": app_customize_ui_texts,
    }
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()

    st.sidebar.markdown(
        """
---
<a href="https://www.buymeacoffee.com/whitphx" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="180" height="50" ></a>
    """,  # noqa: E501
        unsafe_allow_html=True,
    )

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_loopback():
    """Simple video loopback"""
    webrtc_streamer(key="loopback")






class PostProcessing():
    def session(onnxpath):
        ort_session = onnxruntime.InferenceSession(onnxpath)

    def vis_pose(self, img,points):
        for i,point in enumerate(points):
            x,y = point
            x= int(x)
            y= int(y)
            cv2.circle(img,(x,y),4,(0,0,255),thickness=-1,lineType=cv2.FILLED)
            # cv2.putText(img,'{}'.format(i),)
        return img


    def transform_preds(self, coords, center, scale, output_size, use_udp=False):
        assert coords.shape[1] in (2, 4, 5)
        assert len(list(center)) == 2
        assert len(list(scale)) == 2
        assert len(output_size) == 2

        # Recover the scale which is normalized by a factor of 200.
        scale = scale * 200.0

        if use_udp:
            scale_x = scale[0] / (output_size[0] - 1.0)
            scale_y = scale[1] / (output_size[1] - 1.0)
        else:
            scale_x = scale[0] / output_size[0]
            scale_y = scale[1] / output_size[1]

        target_coords = np.ones_like(coords)
        target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

        return target_coords

    def _get_max_preds(self, heatmaps):

        assert isinstance(heatmaps,
                        np.ndarray), ('heatmaps should be numpy.ndarray')
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds, maxvals
    

    def keypoints_from_heatmaps(self,
                                heatmaps,
                                center,
                                scale,
                                unbiased=False,
                                post_process='default',
                                kernel=11,
                                valid_radius_factor=0.0546875,
                                use_udp=False,
                                target_type='GaussianHeatmap'):
        """Get final keypoint predictions from heatmaps and transform them back to
        the image.

        Note:
            - batch size: N
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
            center (np.ndarray[N, 2]): Center of the bounding box (x, y).
            scale (np.ndarray[N, 2]): Scale of the bounding box
                wrt height/width.
            post_process (str/None): Choice of methods to post-process
                heatmaps. Currently supported: None, 'default', 'unbiased',
                'megvii'.
            unbiased (bool): Option to use unbiased decoding. Mutually
                exclusive with megvii.
                Note: this arg is deprecated and unbiased=True can be replaced
                by post_process='unbiased'
                Paper ref: Zhang et al. Distribution-Aware Coordinate
                Representation for Human Pose Estimation (CVPR 2020).
            kernel (int): Gaussian kernel size (K) for modulation, which should
                match the heatmap gaussian sigma when training.
                K=17 for sigma=3 and k=11 for sigma=2.
            valid_radius_factor (float): The radius factor of the positive area
                in classification heatmap for UDP.
            use_udp (bool): Use unbiased data processing.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Classification target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).
                Paper ref: Huang et al. The Devil is in the Details: Delving into
                Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

        Returns:
            tuple: A tuple containing keypoint predictions and scores.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        # Avoid being affected
        heatmaps = heatmaps.copy()

        # detect conflicts
        if unbiased:
            assert post_process not in [False, None, 'megvii']
        if post_process in ['megvii', 'unbiased']:
            assert kernel > 0
        if use_udp:
            assert not post_process == 'megvii'
        
        # start processing

        N, K, H, W = heatmaps.shape

        preds, maxvals = self._get_max_preds(heatmaps)
        # if post_process == 'unbiased':  # alleviate biased coordinate
        #     # apply Gaussian distribution modulation.
        #     heatmaps = np.log(
        #         np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
        #     for n in range(N):
        #         for k in range(K):
        #             preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        if post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5
        # Transform back to the image
        for i in range(N):
            preds[i] = self.transform_preds(
                preds[i], center, scale, [W, H], use_udp=use_udp)

        if post_process == 'megvii':
            maxvals = maxvals / 255.0 + 0.5

        return preds, maxvals
    
    def mapping(self, img, heatmaps, old_img):
        imgWidth, imgHeight, channel = old_img.shape

        imageHegiht,imageWidth = img.shape[2:]
        aspect_ratio = imageWidth / imageHegiht

        boxX, boxY, boxW, boxH = 0, 0, imgHeight, imgWidth
        center = np.array([boxX + boxW/2 , boxY + boxH/2])

        PIXEL_STD = 200
        PADDING = 1
        scale = np.array([boxW*1/PIXEL_STD, boxH*1/PIXEL_STD])
        res = self.keypoints_from_heatmaps(heatmaps,center, scale)[0]
        img = self.vis_pose(old_img ,res[0])
        return img
    
    def inference(self, img, ort_session):

        img1 = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (192,256))
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img = img.unsqueeze(0)
        img = img.numpy()

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: img}
        ort_outs = ort_session.run(None, ort_inputs)
        return self.mapping(img = img, heatmaps = ort_outs[0], old_img = img1)



class Bytetrackprocessing():
    def bytetrackModel(frame):
        args = make_parser().parse_args()
        predictor = Predictor(args)
        tracker = BYTETracker(args, frame_rate=30)
        timer = Timer()
        frame_id = 0
        results = []
        outputs, img_info = predictor.inference(frame, timer)
        online_targets = tracker.update(outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        timer.toc()
        results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                    fps=1. / timer.average_time)
        return online_im

class SuperResolutionNet(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

class Predictor(object):
    def __init__(self, args):
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.args = args
        self.session = onnxruntime.InferenceSession(args.model)
        self.input_shape = tuple(map(int, args.input_shape.split(',')))

    def inference(self, ori_img, timer):
        img_info = {"id": 0}
        height, width = ori_img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = ori_img

        img, ratio = preprocess(ori_img, self.input_shape, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        timer.tic()
        output = self.session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], self.input_shape, p6=self.args.with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.args.nms_thr, score_thr=self.args.score_thr)
        print(dets[:, :-1])
        return dets[:, :-1], img_info

def onnx_test():
    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
    def create_player():
        return MediaPlayer(str("/home/vatsal/Desktop/testonx/palace.mp4"))

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def anotateimg(img, ort_session):
        center_coordinates = (120, 50)
        radius = 20
        color = (255, 0, 0)
        thickness = 2
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.circle(img, center_coordinates, radius, color, thickness)
        #img= cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print(img.shape)
        # print(type(img),img.shape)
        # to_tensor = transforms.ToTensor()
        # img = to_tensor(img)
        # window_name = 'Image'
        '''resize = transforms.Resize([224, 224])
        final_img = resize(img)'''
        img = np.array(resize(img, (224, 224), anti_aliasing=True),dtype=np.float32)
        # print(final_img.shape, type(final_img))
        # img_ycbcr = img.convert('YCbCr')
        # img_y, img_cb, img_cr = img_ycbcr.split()
        # to_tensor = transforms.ToTensor()
        # img_y = to_tensor(img_y)

        img = torch.tensor(np.expand_dims(np.expand_dims(img, axis=0) , axis=0))
        print(img.shape)
        # img = torch.Tensor(img).unsqueeze(0)
        # img = torch.rand((1, 1, 224, 224))
        print(img.shape)
        

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
        ort_outs = ort_session.run(None, ort_inputs)
        final_img = ort_outs[0]
        final_img = np.transpose(final_img[0], (1, 2, 0))
        final_img= cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
        #print(final_img.shape)

        # fina = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

        # # get the output image follow post-processing step from PyTorch implementation
        # final_img = Image.merge(
        #     "YCbCr", [
        #         img_out_y,
        #         img_cb.resize(img_out_y.size, Image.BICUBIC),
        #         img_cr.resize(img_out_y.size, Image.BICUBIC),
        #     ]).convert("RGB")

        # print(np.uint8((final_img * 255.0).clip(0, 255)[0]))
        #print((final_img *255).clip(0, 255).astype(np.uint8))
        #print(final_img *255).clip(0, 255).astype(np.uint8)
        return (final_img *255).clip(0, 255).astype(np.uint8)

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        print('calling superresolution')
        img2 = anotateimg(img, ort_session)
        print(' done with calling superresolution')
        return av.VideoFrame.from_ndarray(img2, format="bgr24")

    webrtc_streamer(
        key='onnxtest',
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
    )

def onnx_test1():
    
    torch_model = SuperResolutionNet(upscale_factor=3)
    batch_size=1
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    torch_out = torch_model(x)
    print(f'torch output {torch_out.shape}')

    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    print(ort_inputs['input'].shape)
    ort_outs = ort_session.run(None, ort_inputs)
    print(f'Ort output {ort_outs[0].shape}')

    # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    vidcap = cv2.VideoCapture('/home/vatsal/Desktop/testonx/palace.mp4')
    success,image = vidcap.read()
    count = 0

    result_queue = (
        queue.Queue()
    )  # TODO: A general-purpose shared state object may be more useful.

    while success:
        # cv2.imwrite("./frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,img = vidcap.read()
        resize = transforms.Resize([224, 224])
        img = Image.fromarray(img)
        img = resize(img)
        img_ycbcr = img.convert('YCbCr')
        img_y, img_cb, img_cr = img_ycbcr.split()

        to_tensor = transforms.ToTensor()
        img_y = to_tensor(img_y)
        img_y.unsqueeze_(0)

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
        ort_outs = ort_session.run(None, ort_inputs)
        img_out_y = ort_outs[0]

        img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

        # get the output image follow post-processing step from PyTorch implementation
        final_img = Image.merge(
            "YCbCr", [
                img_out_y,
                img_cb.resize(img_out_y.size, Image.BICUBIC),
                img_cr.resize(img_out_y.size, Image.BICUBIC),
            ]).convert("RGB")
        final_img.save(f"/home/vatsal/Downloads/catimage{count}.jpg")
        count+=1


        def callback(frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")

            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            # result_queue.put(result)  # TODO:

            return av.VideoFrame.from_ndarray(image, format="bgr24")
    
    

    webrtc_ctx = webrtc_streamer(
        key="ONNX",
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
        



    # img = Image.open("/home/vatsal/Downloads/catimage.jpg")

    # resize = transforms.Resize([224, 224])
    # img = resize(img)

    # img_ycbcr = img.convert('YCbCr')
    # img_y, img_cb, img_cr = img_ycbcr.split()

    # to_tensor = transforms.ToTensor()
    # img_y = to_tensor(img_y)
    # img_y.unsqueeze_(0)

    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    # ort_outs = ort_session.run(None, ort_inputs)
    # img_out_y = ort_outs[0]

    # img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    # # get the output image follow post-processing step from PyTorch implementation
    # final_img = Image.merge(
    #     "YCbCr", [
    #         img_out_y,
    #         img_cb.resize(img_out_y.size, Image.BICUBIC),
    #         img_cr.resize(img_out_y.size, Image.BICUBIC),
    #     ]).convert("RGB")

    # Save the image, we will compare this with the output image from mobile device
    #final_img.save("/home/vatsal/Downloads/catimage26.jpg")

def app_video_filters():
    """Video transforms with OpenCV"""

    _type = st.radio("Select transform type", ("noop", "cartoon", "edges", "rotate"))

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if _type == "noop":
            pass
        elif _type == "cartoon":
            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)
        elif _type == "edges":
            # perform edge detection
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
        elif _type == "rotate":
            # rotate image
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown(
        "This demo is based on "
        "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
        "Many thanks to the project."
    )


def app_audio_filter():
    gain = st.slider("Gain", -10.0, +20.0, 1.0, 0.05)

    def process_audio(frame: av.AudioFrame) -> av.AudioFrame:
        raw_samples = frame.to_ndarray()
        sound = pydub.AudioSegment(
            data=raw_samples.tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels),
        )

        sound = sound.apply_gain(gain)

        # Ref: https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples  # noqa
        channel_sounds = sound.split_to_mono()
        channel_samples = [s.get_array_of_samples() for s in channel_sounds]
        new_samples: np.ndarray = np.array(channel_samples).T
        new_samples = new_samples.reshape(raw_samples.shape)

        new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
        new_frame.sample_rate = frame.sample_rate
        return new_frame

    webrtc_streamer(
        key="audio-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        audio_frame_callback=process_audio,
        async_processing=True,
    )


def app_delayed_echo():
    delay = st.slider("Delay", 0.0, 5.0, 1.0, 0.05)

    async def queued_video_frames_callback(
        frames: List[av.VideoFrame],
    ) -> List[av.VideoFrame]:
        logger.debug("Delay: %f", delay)
        # A standalone `await ...` is interpreted as an expression and
        # the Streamlit magic's target, which leads implicit calls of `st.write`.
        # To prevent it, fix it as `_ = await ...`, a statement.
        # See https://discuss.streamlit.io/t/issue-with-asyncio-run-in-streamlit/7745/15
        _ = await asyncio.sleep(delay)
        return frames

    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> List[av.AudioFrame]:
        _ = await asyncio.sleep(delay)
        return frames

    webrtc_streamer(
        key="delay",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        queued_video_frames_callback=queued_video_frames_callback,
        queued_audio_frames_callback=queued_audio_frames_callback,
        async_processing=True,
    )



def app_human_detection():
    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        frame = frame.to_ndarray(format="bgr24")
        online_im = Bytetrackprocessing.bytetrackModel(frame)
        return av.VideoFrame.from_ndarray(online_im, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown(
        "This demo uses a model and code from "
        "https://github.com/robmarkcole/object-detection-app. "
        "Many thanks to the project."
    )


def app_object_detection():
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """
    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    @st.experimental_singleton
    def generate_label_colors():
        return np.random.uniform(0, 255, size=(len(CLASSES), 3))

    COLORS = generate_label_colors()

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    # Session-specific caching
    cache_key = "object_detection_dnn"
    if cache_key in st.session_state:
        net = st.session_state[cache_key]
    else:
        net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
        st.session_state[cache_key] = net

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )

    def _annotate_image(image, detections):
        # loop over the detections
        (h, w) = image.shape[:2]
        result: List[Detection] = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                name = CLASSES[idx]
                result.append(Detection(name=name, prob=float(confidence)))

                # display the prediction
                label = f"{name}: {round(confidence * 100, 2)}%"
                cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    image,
                    label,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    COLORS[idx],
                    2,
                )
        return image, result

    result_queue = (
        queue.Queue()
    )  # TODO: A general-purpose shared state object may be more useful.

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
        )
        net.setInput(blob)
        detections = net.forward()
        annotated_image, result = _annotate_image(image, detections)

        # NOTE: This `recv` method is called in another thread,
        # so it must be thread-safe.
        result_queue.put(result)  # TODO:

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                try:
                    result = result_queue.get(timeout=1.0)
                except queue.Empty:
                    result = None
                labels_placeholder.table(result)

    st.markdown(
        "This demo uses a model and code from "
        "https://github.com/robmarkcole/object-detection-app. "
        "Many thanks to the project."
    )


def app_streaming():
    """Media streamings"""
    MEDIAFILES = {
        "big_buck_bunny_720p_2mb.mp4 (local)": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_2mb.mp4",  # noqa: E501
            "local_file_path": HERE / "data/big_buck_bunny_720p_2mb.mp4",
            "type": "video",
        },
        "big_buck_bunny_720p_10mb.mp4 (local)": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_10mb.mp4",  # noqa: E501
            "local_file_path": HERE / "data/big_buck_bunny_720p_10mb.mp4",
            "type": "video",
        },
        "file_example_MP3_700KB.mp3 (local)": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3",  # noqa: E501
            "local_file_path": HERE / "data/file_example_MP3_700KB.mp3",
            "type": "audio",
        },
        "file_example_MP3_5MG.mp3 (local)": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_5MG.mp3",  # noqa: E501
            "local_file_path": HERE / "data/file_example_MP3_5MG.mp3",
            "type": "audio",
        },
        "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov": {
            "url": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
            "type": "video",
        },
    }
    media_file_label = st.radio(
        "Select a media source to stream", tuple(MEDIAFILES.keys())
    )
    media_file_info = MEDIAFILES[media_file_label]
    if "local_file_path" in media_file_info:
        download_file(media_file_info["url"], media_file_info["local_file_path"])

    def create_player():
        if "local_file_path" in media_file_info:
            return MediaPlayer(str(media_file_info["local_file_path"]))
        else:
            return MediaPlayer(media_file_info["url"])

        # NOTE: To stream the video from webcam, use the code below.
        # return MediaPlayer(
        #     "1:none",
        #     format="avfoundation",
        #     options={"framerate": "30", "video_size": "1280x720"},
        # )

    key = f"media-streaming-{media_file_label}"
    ctx: Optional[WebRtcStreamerContext] = st.session_state.get(key)
    if media_file_info["type"] == "video" and ctx and ctx.state.playing:
        _type = st.radio(
            "Select transform type", ("noop", "cartoon", "edges", "rotate")
        )
    else:
        _type = "noop"

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if _type == "noop":
            pass
        elif _type == "cartoon":
            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)
        elif _type == "edges":
            # perform edge detection
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
        elif _type == "rotate":
            # rotate image
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key=key,
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": media_file_info["type"] == "video",
            "audio": media_file_info["type"] == "audio",
        },
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
    )

    st.markdown(
        "The video filter in this demo is based on "
        "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
        "Many thanks to the project."
    )


def app_streaming_bytetrack1():
    #subprocess.call(" python3 /home/vatsal/Desktop/ByteTrack/tools/demo_track.py video -f /home/vatsal/Desktop/ByteTrack/exps/example/mot/yolox_x_mix_det.py -c /home/vatsal/Desktop/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result", shell=True)
    d = "/home/vatsal/Desktop/ByteTrack/YOLOX_outputs/yolox_x_mix_det/track_vis/2022_08_23_16_12_37/palace.mp4"
    MEDIAFILES = {
        "big_buck_bunny_720p_2mb.mp4 (local)": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_2mb.mp4",  # noqa: E501
            "local_file_path": d,
            "type": "video",
        },
        "big_buck_bunny_720p_10mb.mp4 (local)": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_10mb.mp4",  # noqa: E501
            "local_file_path": HERE / "data/big_buck_bunny_720p_10mb.mp4",
            "type": "video",
        },
        "file_example_MP3_700KB.mp3 (local)": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3",  # noqa: E501
            "local_file_path": HERE / "data/file_example_MP3_700KB.mp3",
            "type": "audio",
        },
        "file_example_MP3_5MG.mp3 (local)": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_5MG.mp3",  # noqa: E501
            "local_file_path": HERE / "data/file_example_MP3_5MG.mp3",
            "type": "audio",
        },
        "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov": {
            "url": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
            "type": "video",
        },
    }
    media_file_label = "big_buck_bunny_720p_2mb.mp4 (local)"
    media_file_info = MEDIAFILES[media_file_label]

    def create_player():
        if "local_file_path" in media_file_info:
            return MediaPlayer(str(media_file_info["local_file_path"]))
        else:
            return MediaPlayer(media_file_info["url"])

        # NOTE: To stream the video from webcam, use the code below.
        # return MediaPlayer(
        #     "1:none",
        #     format="avfoundation",
        #     options={"framerate": "30", "video_size": "1280x720"},
        # )

    key = f"media-streaming-{media_file_label}"
    #ctx: Optional[WebRtcStreamerContext] = st.session_state.get(key)
    _type = "noop"

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key=key,
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": media_file_info["type"] == "video",
            "audio": media_file_info["type"] == "audio",
        },
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
    )

def app_streaming_bytetrack():
    #subprocess.call(" python3 /home/vatsal/Desktop/ByteTrack/tools/demo_track.py video -f /home/vatsal/Desktop/ByteTrack/exps/example/mot/yolox_x_mix_det.py -c /home/vatsal/Desktop/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result", shell=True)
    def create_player():
        return MediaPlayer(str("/home/vatsal/Desktop/testonx/palace.mp4"))

    key = "media streaming bytetrack modeling"
    def video_frame_callback(frame):
        frame = frame.to_ndarray(format="bgr24")
        args = make_parser().parse_args()
        predictor = Predictor(args)
        tracker = BYTETracker(args, frame_rate=30)
        timer = Timer()
        frame_id = 0
        results = []
        outputs, img_info = predictor.inference(frame, timer)
        online_targets = tracker.update(outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        timer.toc()
        results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
                                    fps=1. / timer.average_time)
        return av.VideoFrame.from_ndarray(online_im, format="bgr24")

    webrtc_streamer(
        key=key,
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
    )

def app_streaming_classBytetrack():
    #subprocess.call(" python3 /home/vatsal/Desktop/ByteTrack/tools/demo_track.py video -f /home/vatsal/Desktop/ByteTrack/exps/example/mot/yolox_x_mix_det.py -c /home/vatsal/Desktop/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result", shell=True)
    def create_player():
        return MediaPlayer(str("/home/vatsal/Desktop/testonx/palace.mp4"))

    key = "media streaming bytetrack modeling"
    def video_frame_callback(frame):
        frame = frame.to_ndarray(format="bgr24")
        online_im = Bytetrackprocessing.bytetrackModel(frame)
        return av.VideoFrame.from_ndarray(online_im, format="bgr24")

     
    webrtc_streamer(
        key=key,
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
    )




def app_streaming_vitpose():
    ort_session = onnxruntime.InferenceSession("/home/vatsal/Desktop/vitpose/ViTPose/tmp.onnx")
    #subprocess.call(" python3 /home/vatsal/Desktop/ByteTrack/tools/demo_track.py video -f /home/vatsal/Desktop/ByteTrack/exps/example/mot/yolox_x_mix_det.py -c /home/vatsal/Desktop/ByteTrack/pretrained/bytetrack_x_mot17.pth.tar --fp16 --fuse --save_result", shell=True)
    def create_player():
        return MediaPlayer(str("/home/vatsal/Desktop/MMPOSE/mmpose/testdata/single12.mp4"))

    key = "media streaming bytetrack modeling"
    def transform_preds(coords, center, scale, output_size, use_udp=False):

        print(center)
        assert coords.shape[1] in (2, 4, 5)
        assert len(list(center)) == 2
        assert len(list(scale)) == 2
        assert len(output_size) == 2

        # Recover the scale which is normalized by a factor of 200.
        scale = scale * 200.0

        if use_udp:
            scale_x = scale[0] / (output_size[0] - 1.0)
            scale_y = scale[1] / (output_size[1] - 1.0)
        else:
            scale_x = scale[0] / output_size[0]
            scale_y = scale[1] / output_size[1]

        target_coords = np.ones_like(coords)
        target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
        target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

        return target_coords

    def _get_max_preds(heatmaps):

        assert isinstance(heatmaps,
                        np.ndarray), ('heatmaps should be numpy.ndarray')
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        N, K, _, W = heatmaps.shape
        heatmaps_reshaped = heatmaps.reshape((N, K, -1))
        idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
        maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        preds[:, :, 0] = preds[:, :, 0] % W
        preds[:, :, 1] = preds[:, :, 1] // W

        preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
        return preds, maxvals
    

    def keypoints_from_heatmaps(heatmaps,
                                center,
                                scale,
                                unbiased=False,
                                post_process='default',
                                kernel=11,
                                valid_radius_factor=0.0546875,
                                use_udp=False,
                                target_type='GaussianHeatmap'):
        """Get final keypoint predictions from heatmaps and transform them back to
        the image.

        Note:
            - batch size: N
            - num keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
            center (np.ndarray[N, 2]): Center of the bounding box (x, y).
            scale (np.ndarray[N, 2]): Scale of the bounding box
                wrt height/width.
            post_process (str/None): Choice of methods to post-process
                heatmaps. Currently supported: None, 'default', 'unbiased',
                'megvii'.
            unbiased (bool): Option to use unbiased decoding. Mutually
                exclusive with megvii.
                Note: this arg is deprecated and unbiased=True can be replaced
                by post_process='unbiased'
                Paper ref: Zhang et al. Distribution-Aware Coordinate
                Representation for Human Pose Estimation (CVPR 2020).
            kernel (int): Gaussian kernel size (K) for modulation, which should
                match the heatmap gaussian sigma when training.
                K=17 for sigma=3 and k=11 for sigma=2.
            valid_radius_factor (float): The radius factor of the positive area
                in classification heatmap for UDP.
            use_udp (bool): Use unbiased data processing.
            target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
                GaussianHeatmap: Classification target with gaussian distribution.
                CombinedTarget: The combination of classification target
                (response map) and regression target (offset map).
                Paper ref: Huang et al. The Devil is in the Details: Delving into
                Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

        Returns:
            tuple: A tuple containing keypoint predictions and scores.

            - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
            - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
        """
        # Avoid being affected
        heatmaps = heatmaps.copy()

        # detect conflicts
        if unbiased:
            assert post_process not in [False, None, 'megvii']
        if post_process in ['megvii', 'unbiased']:
            assert kernel > 0
        if use_udp:
            assert not post_process == 'megvii'
        
        # start processing

        N, K, H, W = heatmaps.shape

        preds, maxvals = _get_max_preds(heatmaps)
        # if post_process == 'unbiased':  # alleviate biased coordinate
        #     # apply Gaussian distribution modulation.
        #     heatmaps = np.log(
        #         np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
        #     for n in range(N):
        #         for k in range(K):
        #             preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        if post_process is not None:
            # add +/-0.25 shift to the predicted locations for higher acc.
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array([
                            heatmap[py][px + 1] - heatmap[py][px - 1],
                            heatmap[py + 1][px] - heatmap[py - 1][px]
                        ])
                        preds[n][k] += np.sign(diff) * .25
                        if post_process == 'megvii':
                            preds[n][k] += 0.5
        # Transform back to the image
        for i in range(N):
            preds[i] = transform_preds(
                preds[i], center, scale, [W, H], use_udp=use_udp)

        if post_process == 'megvii':
            maxvals = maxvals / 255.0 + 0.5

        return preds, maxvals

    def vitpose(img, ort_session):
        img = cv2.flip(img, -1)
        img1 = img
        imgWidth, imgHeight, channel = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (192,256))
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        img = img.unsqueeze(0)
        img = img.numpy()

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        def vis_pose(img,points):
            for i,point in enumerate(points):
                x,y = point
                x= int(x)
                y= int(y)
                cv2.circle(img,(x,y),4,(0,0,255),thickness=-1,lineType=cv2.FILLED)
                # cv2.putText(img,'{}'.format(i),)
            return img
        ort_inputs = {ort_session.get_inputs()[0].name: img}
        ort_outs = ort_session.run(None, ort_inputs)
        heatmaps = ort_outs[0]
        print(heatmaps.shape)

        imageHegiht,imageWidth = img.shape[2:]
        aspect_ratio = imageWidth / imageHegiht

        boxX, boxY, boxW, boxH = 0, 0, imgHeight, imgWidth
        center = np.array([boxX + boxW/2 , boxY + boxH/2])
        # if boxW > aspect_ratio * boxH:
        #     boxH = boxW * 1.0 / aspect_ratio
        # elif boxW < aspect_ratio * boxH:
        #     boxW = boxH * aspect_ratio

        PIXEL_STD = 200
        PADDING = 1
        scale = np.array([boxW*1/PIXEL_STD, boxH*1/PIXEL_STD]) 
        res = keypoints_from_heatmaps(heatmaps,center, scale)[0]
        img = vis_pose(img1,res[0])
        return img
    def video_frame_callback(frame):
        frame = frame.to_ndarray(format="bgr24")
        online_im = vitpose(frame,ort_session)
        return av.VideoFrame.from_ndarray(online_im, format="bgr24")

     
    webrtc_streamer(
        key=key,
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
    )


def app_streaming_mmpose():
    ort_session = onnxruntime.InferenceSession("/home/vatsal/Desktop/MMPOSE/mmpose/end2end.onnx")
    def create_player():
        return MediaPlayer(str("/home/vatsal/Desktop/MMPOSE/mmpose/testdata/single.mp4"))

    key = "media streaming bytetrack modeling"


          
    def video_frame_callback(frame):
        frame = frame.to_ndarray(format="bgr24")
        model = PostProcessing()
        online_im = model.inference(frame,ort_session)
        return av.VideoFrame.from_ndarray(online_im, format="bgr24")
 
    webrtc_streamer(
        key=key,
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
    )


def app_streaming_vitpose2():
    ort_session = onnxruntime.InferenceSession("/home/vatsal/Desktop/vitpose/ViTPose/tmp.onnx")
    def create_player():
        return MediaPlayer(str("/home/vatsal/Desktop/MMPOSE/mmpose/testdata/single.mp4"))

    key = "media streaming bytetrack modeling"


          
    def video_frame_callback(frame):
        frame = frame.to_ndarray(format="bgr24")
        model = PostProcessing()
        online_im = model.inference(frame,ort_session)
        return av.VideoFrame.from_ndarray(online_im, format="bgr24")
 
    webrtc_streamer(
        key=key,
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
    )


def app_streaming_poseestimation():
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

    CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    box_session = onnxruntime.InferenceSession("box_model.onnx")
    hrnet_session = onnxruntime.InferenceSession("hrnet.onnx")
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    # cudnn.benchmark = cfg.CUDNN.BENCHMARK
    # torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    # torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # args = parse_args()
    # update_config(cfg, args)
    # csv_output_rows = []
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
            trans = get_affine_transform(center, scale, rotation, (288,384))
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
        print(output.shape)
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

    def create_player():
        return MediaPlayer(str("/home/vatsal/Desktop/testonx/palace.mp4"))

    key = "media streaming bytetrack modeling"

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    def video_frame_callback(image_bgr):
        total_now = time.time()
        print(image_bgr)
        image_bgr = image_bgr.to_ndarray(format="bgr24")
        #cv2.imwrite(f"/home/vatsal/Desktop/stapp/streamlit-webrtc-example/test/test{total_now}.jpg",image_bgr)


        #image_bgr = cv2.imread(image_bgr)
        image_rgb = image_bgr #cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"/home/vatsal/Desktop/stapp/streamlit-webrtc-example/test/test1{total_now}.jpg",image_rgb)
        print(image_rgb)
        image_per = image_rgb.copy()
        image_pose = image_rgb.copy()
        image_debug = image_rgb.copy()
        # object detection box
        now = time.time()
        pred_boxes = get_person_detection_boxes(box_session, image_per, threshold=0.9)
        print(pred_boxes)
        then = time.time()
        print("Find person bbox in: {} sec".format(then - now))


        # if args.writeBoxFrames:
        #     for box in pred_boxes:
        #         print(box[0])
        #         # cv2.rectangle(frame, (int(box[0][0]), int(box[0][1])), (int(x + w), int(y + h)), (0, 255, 0), 2)
        #         cv2.rectangle(image_debug, (int(box[0][0]), int(box[0][1])), (int(box[0][0] + box[1][0]), int(box[1][0] + box[1][1])), (0, 255, 0), 2)
        #         # cv2.rectangle(image_debug, int(box[0]), int(box[1]), color=(0, 255, 0),
        #         #               thickness=3)  # Draw Rectangle with the coordinates

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
        #cv2.imwrite(f"/home/vatsal/Desktop/stapp/streamlit-webrtc-example/test/test1{image_debug}.jpg",image_bgr)
        print(image_debug)
        #cv2.imshow("pos", av.VideoFrame.from_ndarray(image_debug, format="bgr24"))
        #csv_output_rows.append(new_csv_row)
        #img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
        #cv2.imwrite(img_file, image_debug)
        return av.VideoFrame.from_ndarray(image_debug, format="bgr24")
 
    webrtc_streamer(
        key=key,
        mode=WebRtcMode.RECVONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        player_factory=create_player,
        video_frame_callback=video_frame_callback,
    )


def app_sendonly_video():
    """A sample to use WebRTC in sendonly mode to transfer frames
    from the browser to the server and to render frames via `st.image`."""
    webrtc_ctx = webrtc_streamer(
        key="video-sendonly",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True},
    )
    image_place = st.empty()
    while True:
        if webrtc_ctx.video_receiver:
            try:
                video_frame = webrtc_ctx.video_receiver.get_frame(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            img_rgb = video_frame.to_ndarray(format="rgb24")
            image_place.image(img_rgb)
        else:
            logger.warning("AudioReciver is not set. Abort.")
            break


def app_sendonly_audio():
    """A sample to use WebRTC in sendonly mode to transfer audio frames
    from the browser to the server and visualize them with matplotlib
    and `st.pyplot`."""
    webrtc_ctx = webrtc_streamer(
        key="sendonly-audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"audio": True},
    )

    fig_place = st.empty()

    fig, [ax_time, ax_freq] = plt.subplots(
        2, 1, gridspec_kw={"top": 1.5, "bottom": 0.2}
    )

    sound_window_len = 5000  # 5s
    sound_window_buffer = None
    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                logger.warning("Queue is empty. Abort.")
                break

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                if sound_window_buffer is None:
                    sound_window_buffer = pydub.AudioSegment.silent(
                        duration=sound_window_len
                    )

                sound_window_buffer += sound_chunk
                if len(sound_window_buffer) > sound_window_len:
                    sound_window_buffer = sound_window_buffer[-sound_window_len:]

            if sound_window_buffer:
                # Ref: https://own-search-and-study.xyz/2017/10/27/python%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E9%9F%B3%E5%A3%B0%E3%83%87%E3%83%BC%E3%82%BF%E3%81%8B%E3%82%89%E3%82%B9%E3%83%9A%E3%82%AF%E3%83%88%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%A0%E3%82%92%E4%BD%9C/  # noqa
                sound_window_buffer = sound_window_buffer.set_channels(
                    1
                )  # Stereo to mono
                sample = np.array(sound_window_buffer.get_array_of_samples())

                ax_time.cla()
                times = (np.arange(-len(sample), 0)) / sound_window_buffer.frame_rate
                ax_time.plot(times, sample)
                ax_time.set_xlabel("Time")
                ax_time.set_ylabel("Magnitude")

                spec = np.fft.fft(sample)
                freq = np.fft.fftfreq(sample.shape[0], 1.0 / sound_chunk.frame_rate)
                freq = freq[: int(freq.shape[0] / 2)]
                spec = spec[: int(spec.shape[0] / 2)]
                spec[0] = spec[0] / 2

                ax_freq.cla()
                ax_freq.plot(freq, np.abs(spec))
                ax_freq.set_xlabel("Frequency")
                ax_freq.set_yscale("log")
                ax_freq.set_ylabel("Magnitude")

                fig_place.pyplot(fig)
        else:
            logger.warning("AudioReciver is not set. Abort.")
            break


def app_media_constraints():
    """A sample to configure MediaStreamConstraints object"""
    frame_rate = 5
    webrtc_streamer(
        key="media-constraints",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"frameRate": {"ideal": frame_rate}},
        },
        video_html_attrs={
            "style": {"width": "50%", "margin": "0 auto", "border": "5px yellow solid"},
            "controls": False,
            "autoPlay": True,
        },
    )
    st.write(f"The frame rate is set as {frame_rate}. Video style is changed.")


def app_programatically_play():
    """A sample of controlling the playing state from Python."""
    playing = st.checkbox("Playing", value=True)

    webrtc_streamer(
        key="programatic_control",
        desired_playing_state=playing,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
    )


def app_customize_ui_texts():
    webrtc_streamer(
        key="custom_ui_texts",
        rtc_configuration=RTC_CONFIGURATION,
        translations={
            "start": "開始",
            "stop": "停止",
            "select_device": "デバイス選択",
            "media_api_not_available": "Media APIが利用できない環境です",
            "device_ask_permission": "メディアデバイスへのアクセスを許可してください",
            "device_not_available": "メディアデバイスを利用できません",
            "device_access_denied": "メディアデバイスへのアクセスが拒否されました",
        },
    )


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
