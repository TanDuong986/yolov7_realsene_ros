#!/home/thesisrb/Workspace/miniforge3/envs/py36tor/bin/python3
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import pyrealsense2 as rs
import numpy as np
import os

path_detect = os.path.dirname(os.path.realpath(__file__))

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    ## bắt đầu khác source ,  từ đoạn này sẽ config và xử lý tuần tự dữ liệu nhận được từ đọc frame (realsense, webcam, video)
    config = rs.config() # tạo instance camera và config #################
    config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30) # chọn 
    config.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to) # align để đảm bảo đồng bộ về depth và rgb đây là cấu hình

    #######################################################
    while(True):
        #t0 = time.time()
        frames = pipeline.wait_for_frames() # có frame của  rgb và depth

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame() # đây là instance chưa phải ảnh
        depth_frame = aligned_frames.get_depth_frame() 
        if not depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data()) # chuyển ảnh sang numpy để  tính toán
        depth_image = np.asanyarray(depth_frame.get_data()) # ảnh này là khoảng cách, chưa có màu
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
        img = cv2.resize(img,(640,640))
        # Letterbox
        im0 = img.copy()
        img = img[np.newaxis, :, :, :] #thêm một chiều mới làm batch

        # Stack
        img = np.stack(img, 0) # chất các batch lên nhau

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img) # sắp xếp các array img vào bộ nhớ liền mạch để tăng tốc

        # sau đoạn này thì bắt đầu giống source, đã có ảnh chất đống, đoạn sau là infer và in kết quả vào rgb và depth
        ###################################################################################

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: # đảm bảo biểu diễn theo dạng BCHW
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # không có classify giống source


        # Process detections
        for i, det in enumerate(pred):  # detections per image

            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # đoạn này không dùng
            # có thể khai báo mấy folder save ở đây
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    # plot_one_box(xyxy, depth_colormap, label=label, color=colors[int(cls)], line_thickness=2)             

            # Print time (inference + NMS)
            print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            # if thì stream nhé
            cv2.imshow("Recognition result", im0)
            # cv2.imshow("Recognition result depth",depth_colormap)
            
            # if save_img: # nếu muốn save
            #     if vid_path != save_path:  # check tên hoặc tạo mới
            #         vid_path = save_path
            #         if isinstance(vid_writer, cv2.VideoWriter):
            #             vid_writer.release()  # release previous video writer
            #         if vid_cap:  # lấy kích thước vào FPS mong muốn từ video record
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #         else:  # stream
            #             fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path += '.mp4'
            #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) # tạo instance ghi
            #     vid_writer.write(im0) # ghi liên tục
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__': #source, weights, view_img, save_txt, imgsz, trace
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weight/origin_v4data_v7nm.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['last.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
