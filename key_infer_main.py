#!/usr/bin/env python
import argparse
import time
from pathlib import Path
import pandas as pd
import cv2
from numpy import random

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Header

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression,\
      strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, TracedModel

from rsgt.__init_rs__ import *
import numpy as np
from cv_bridge import CvBridge

path_detect = os.path.dirname(os.path.realpath(__file__))


def determine_dist(box,object_dist,frame_size=(640,480)): 
    ''' box is tuple , object_dist is object realsense
    xmin,ymin|xmax,ymax  / range_find distance from centroid/ number of sample
    find range to compute distance
    firstly use random choice point | can experiment on fixed point to robust realiability '''

    HFOV,VFOV = 69.4,42.5 # horizon and verticle field of view of realsense | see detail in :https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html
    # 86,57 for depth
    centroid = [int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
    cheat = object_dist.get_distance(*centroid)
    w2 = frame_size[1]/2 # mid of W
    h2 = frame_size[0]/2 # mid of H
    h_angle = round(((centroid[0] - w2) / w2) * (HFOV/2),4)
    v_angle = round(((centroid[1] - h2) / h2) * (VFOV/2),4)
    angle = (h_angle,v_angle)
    return cheat,angle

def detect():
    weights, imgsz, trace = opt.weights, opt.img_size, not opt.no_trace

    #init rosnode######################### 
    # Initialize the ROS node
    rospy.init_node('yolo_node', anonymous=False)

    # Create a publisher with topic name 'string_topic' and message type String
    pub_txt = rospy.Publisher('bbox_topic', String, queue_size=100)
    pub_img = rospy.Publisher('rgb_topic', Image, queue_size=100)

    ###########################################################
    bridge = CvBridge()

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if trace:
    #     model = TracedModel(model, device, opt.img_size)

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

    # bat dau khac source, tu doan nay se config va xu ly tuan tu du lieu nhan duoc tu frame realsense
    #config camera
    cfg = pipeline.start(config)
    dev = cfg.get_device()
    advnc_mode = rs.rs400_advanced_mode(dev)
    advnc_mode.load_json(json_string)
    align = rs.align(rs.stream.color)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    
    for x in range(5):
        pipeline.wait_for_frames()
    #######################################################
    id_frame = 1000
    header = ["bbox","confident","distance","angle","resolution","frame_id"]
    while not rospy.is_shutdown():

        # wait for full of two data: depth and color
        frames = pipeline.wait_for_frames()
        #Create alignment primitive with color as its target stream  | make RGBD
        frames = align.process(frames)

        color_frame = frames.get_color_frame()         
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame or not color_frame:
            continue            

        # depth_frame = decimation.process(depth_frame)
        # depth_frame = depth_to_disparity.process(depth_frame)
        # depth_frame = spatial.process(depth_frame)
        # depth_frame = temporal.process(depth_frame)
        depth_frame = hole.process(depth_frame)
        # depth_frame = disparity_to_depth.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame() # convert to depth frame object origin
        
        colorizer = rs.colorizer()
        depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        # depth_image = np.asanyarray(depth_frame.get_data()).astype(float)
        color_image = np.asanyarray(color_frame.get_data()) #720x1280 |16:9
        ################### DONE LOAD IMAGE FRAME################
        '''
        Now, RGB with 1280x720 will be resized to 640x640 and infer with YOLO.
        Depth will wait for box of yolo results and calculate distance and angle
        Then, resize the box return of YOLO and display RGB, distance, angle
        '''
        
        img = np.asanyarray(color_frame.get_data()) # chuyển ảnh sang numpy để  tính toán
        # depth_image = np.asanyarray(depth_frame.get_data()) # ảnh này là khoảng cách, chưa có màu
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
        img = cv2.resize(img,(640,640))
        # Letterbox
        img = img[np.newaxis, :, :, :] #add new axis to use as batch

        # Stack
        img = np.stack(img, 0) # stack all batch

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img) # arrange array to accelarate

        # same the source
        ###################################################################################

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: # make sure arrange follow BCHW
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        # t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        # t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # t3 = time_synchronized()

        # not have classify like source

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if len(det):
                '''
                det is matrix n*m (n is number of object detected, m is 6 )
                m is representated for bbox(4 xmin,ymin,xmax,ymax in scale 640) and confident, value of class (0 is head)'''

                # Write results
                dist_arr =[]
                angle_arr =[]
                box_arr =[]
                relu = []
                id_arr =[]

                # try to update this by use index of det, not use for loop
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    box= plot_one_box(xyxy, color_image
                                      , label=label
                                      , color=colors[int(cls)]
                                      , line_thickness=2
                                      , draw=False)
                    dist,angle = determine_dist(box=box,object_dist=depth_frame,
                                                frame_size= color_image.shape[0:2])
                    ## draw circle on image
                    # cv2.circle(depth_image,(int((box[2]+box[0])/2),int((box[3]+box[1])/2)),10,(0,255,0),thickness=-1)
                    # print(f'dist is {dist}')
                    # print(f'angle is {angle}')
                    # cv2.rectangle(color_image,(box[0],box[1]),(box[2],box[3]),color=(0,0,255),thickness=2)   
                    # column : bbox (xyxy) | dist (m) | angle  

                    dist_arr.append(dist)
                    angle_arr.append(angle)
                    box_arr.append(box)
                    relu.append(color_image.shape[0:2])
                    id_arr.append(str(id_frame))
                
                #prepare result txt to publish
                dict_rs = {header[0]:box_arr,header[1]:det[:,4].to('cpu'),header[2]:dist_arr,header[3]:angle_arr,header[4]:relu,header[-1]:id_arr}
                data = pd.DataFrame(dict_rs,columns=header)
                frame_trans= data.to_json(orient='records')

                pub_txt.publish(frame_trans)
                rospy.loginfo("head detected")
                # print(frame_trans)
                # print()
                # rate.sleep()
                
            else :
                rospy.loginfo('____ No head ! _____')
        img_msg = bridge.cv2_to_imgmsg(color_image,encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_msg.header.frame_id = str(id_frame)
        pub_img.publish(img_msg)
        # imgg = np.hstack((color_image,depth_image))
        # Print time (inference + NMS)
        # print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
        if id_frame == 1999:
            id_frame =1000
        else:
            id_frame +=1

        # cv2.imshow("Recognition result",color_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     pipeline.stop() 
        #     break
    pipeline.stop()  
        


if __name__ == '__main__': #source, weights, view_img, save_txt, imgsz, trace
    parser = argparse.ArgumentParser()
    wp = os.path.join(path_detect,'weight/origin_v4data_v7nm.pt')
    parser.add_argument('--weights', nargs='+', type=str, default=wp, help='model.pt path(s)')
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
    #check_requirements(exclude=('pycocotools', 'thop'))
    #print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['last.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
