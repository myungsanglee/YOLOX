#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2
import numpy as np
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
LPR_CLASSES = (
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'rk',
    'sk',
    'ek',
    'fk',
    'ak',
    'rj',
    'sj',
    'ej',
    'fj',
    'aj',
    'rh',
    'sh',
    'eh',
    'fh',
    'ah',
    'rn',
    'sn',
    'en',
    'fn',
    'an',
    'qj',
    'tj',
    'dj',
    'wj',
    'qh',
    'th',
    'dh',
    'wh',
    'qn',
    'tn',
    'dn',
    'wn',
    'gj',
    'gk',
    'gh',
    'dk',
    'qk',
    'tk',
    'wk',
    'tjdnf',
    'qntks',
    'eorn',
    'dlscjs',
    'rhkdwn',
    'eowjs',
    'dnftks',
    'rudrl',
    'rkddnjs',
    'cndqnr',
    'cndska',
    'wjsqnr',
    'wjsska',
    'rudqnr',
    'rudska',
    'wpwn',
    'dhlry',
    'dudtk',
    'wnsdhl',
    'wnsdud',
    'rnrrl',
    'guqwjd',
    'eovy',
    'qo'
)

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img_path = img
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        ############################################################################################
        img_h, img_w = img.shape[:2]

        txt_path = img_path.rsplit('.', 1)[0] + '.txt'
        with open(txt_path, 'r') as f:
            labels = f.read().splitlines()
        # labels = [[float(y) for y in x.split(' ')[1:]] for x in labels if x.split(' ')[0] == '8']
        labels = [[float(y) for y in x.split(' ')[1:]] for x in labels if x.split(' ')[0] == '0']
        label = labels[0]
        
        cx = label[0] * img_w
        cy = label[1] * img_h
        w = label[2] * img_w
        h = label[3] * img_h
        
        xmin = int(cx - (w / 2))
        ymin = int(cy - (h / 2))
        xmax = int(cx + (w / 2))
        ymax = int(cy + (h / 2))
        
        global crop_img_height
        crop_img_height = ymax - ymin
        
        img = img[ymin:ymax, xmin:xmax].copy()
        ############################################################################################
        
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, np.zeros((0, 6))
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        
        ############################################################################################
        detections = torch.cat([cls.unsqueeze(-1), cls.unsqueeze(-1), bboxes], dim=1)
        return img, detections
        ############################################################################################
        
        # vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        # return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    
    global crop_img_height
    with open('/home/fssv2/myungsang/datasets/lpr/lpr_kr.names') as f:
        name_list = f.read().splitlines()
    true_num = 0
    total_num = len(files)
    
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        
        img, detections = predictor.visual(outputs[0], img_info, predictor.confthre)
        plate_num, detections = get_plate_number(detections, crop_img_height, name_list)
        true_label = os.path.basename(image_name).rsplit('.', 1)[0]
        if len(true_label.split('-')) > 1:
            true_label = true_label.split('-')[0]
        if plate_num == true_label:
            true_num += 1
        print(f'True: {true_label}, Pred: {plate_num}')
        
        # show image
        for detection in detections:
            _, _, x1, y1, x2, y2 = detection
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0))
        cv2.imshow('Demo', img)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        
        # if save_result:
        #     save_folder = os.path.join(
        #         vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        #     )
        #     os.makedirs(save_folder, exist_ok=True)
        #     save_file_name = os.path.join(save_folder, os.path.basename(image_name))
        #     logger.info("Saving detection result in {}".format(save_file_name))
        #     cv2.imwrite(save_file_name, result_image)
        
    print(f'Accuracy: {true_num} / {total_num} = {(true_num / total_num)*100:.2f}%')


def get_plate_number(detections, network_height, cls_name_list):
    detect_num = len(detections)
    if detect_num < 4:
        return 'None', detections
    elif detect_num > 8:
        detections = np.delete(detections, np.argsort(detections[..., 1])[:detect_num-8], axis=0)
    detections = detections[np.argsort(detections[..., 3])]
    
    thresh = int(network_height / 5)
    y1 = detections[1][3] - detections[0][3]
    y2 = detections[3][3] - detections[2][3]
    
    # 외교 번호판
    if y1 > thresh:
        detections[1:] = detections[1:][np.argsort(detections[1:, 2])]
    
    # 운수/건설 번호판
    elif y2 > thresh:
        detections[:3] = detections[:3][np.argsort(detections[:3, 2])]
        detections[3:] = detections[3:][np.argsort(detections[3:, 2])]
    
    # 일반 가로형 번호판
    else:
        detections = detections[np.argsort(detections[..., 2])]

    # 번호판 포맷에 맞는지 체크
    detections = check_plate(detections)

    plate_num = ''
    for cls_idx in detections[..., 0]:
        plate_num += cls_name_list[int(cls_idx)]
    
    return plate_num, detections


def check_plate(detections):
    # 가, 나, 다, ... 번호는 하나만 존재하고 그 뒤의 번호는 4자리만 올 수 있다.
    str_idx_list = np.where((10<=detections[..., 0]) & (detections[..., 0]<=48))[0]
    if len(str_idx_list):
        if len(str_idx_list) > 1:
            arg_idx = np.argsort(-detections[str_idx_list][..., 1])
            delete_idx = str_idx_list[arg_idx[1:]]
            detections = np.delete(detections, delete_idx, axis=0)
            str_idx = str_idx_list[arg_idx[0]]
        else:
            str_idx = str_idx_list[0]
            
        if len(detections[str_idx+1:]) > 4:
            delete_idx = np.argsort(-detections[str_idx+1:, 1])[4:] + (str_idx + 1)
            detections = np.delete(detections, delete_idx, axis=0)

    # 서울, 경기, ... 지역 번호는 하나만 존재
    area_idx_list = np.where((49<=detections[..., 0]) & (detections[..., 0]<=64))[0]
    if len(area_idx_list) > 1:
        arg_idx = np.argsort(-detections[area_idx_list][..., 1])
        delete_idx = area_idx_list[arg_idx[1:]]
        detections = np.delete(detections, delete_idx, axis=0)
    
    # 외교, 영사, ... 번호는 하나만 존재
    diplomacy_idx_list = np.where(64 < detections[..., 0])[0]
    if len(diplomacy_idx_list) > 1:
        arg_idx = np.argsort(-detections[diplomacy_idx_list][..., 1])
        delete_idx = diplomacy_idx_list[arg_idx[1:]]
        detections = np.delete(detections, delete_idx, axis=0)
    
    return detections

def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, LPR_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
