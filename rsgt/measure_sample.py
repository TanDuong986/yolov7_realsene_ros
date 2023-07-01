import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from __init_rs__ import *

def determine_dist(box,object_dist,frame_size=(640,480),range_find = 0.2): 
    ''' box is tuple , object_dist is object realsense
    xmin,ymin|xmax,ymax  / range_find distance from centroid/ number of sample
    find range to compute distance
    firstly use random choice point | can experiment on fixed point to robust realiability '''

    HFOV,VFOV = 69.4,42.5 # horizon and verticle field of view of realsense | see detail in :https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html
    # 86,57 for depth
    centroid = [int((box[0]+box[2])/2),int((box[1]+box[3])/2)]
    cheat = object_dist.get_distance(*centroid)
    # w,h = box[2]-box[0],box[3]-box[1]
    # range_csd = (int(w*range_find/2), int(h*range_find/2))
    # num_point =5
    # # list_point= np.array([[np.random.randint(-range_csd[0],range_csd[0]) for _ in range(sample)],
    # #                       [np.random .randint(-range_csd[1],range_csd[1]) for _ in range(sample)]])
    # # list_point[0] += centroid[0]
    # # list_point[1] += centroid[1]
    # list_point = np.array([[centroid[0]-range_csd[0],centroid[0],centroid[0]+range_csd[0],centroid[0]-range_csd[0],centroid[0]+range_csd[0]],
    #               [centroid[1]-range_csd[1],centroid[1],centroid[1]-range_csd[1],centroid[1]+range_csd[1],centroid[1]+range_csd[1]]])
    # '''P1-------P3
    #     ----P2---- (centroid)
    #    P4-------P5'''
    # # turn on camera and get distance of all point created
    # dist = np.array([object_dist.get_distance(list_point[0,i],list_point[1,i]) for i in range(num_point)])
    # valid_point = np.nonzero(dist)
    w2 = frame_size[1]/2 # mid of W
    h2 = frame_size[0]/2 # mid of H
    h_angle = round(((centroid[0] - w2) / w2) * (HFOV/2),4)
    v_angle = round(((centroid[1] - h2) / h2) * (VFOV/2),4)
    angle = (h_angle,v_angle)
    # result = (np.sum(dist[valid_point]))/num_point
    return cheat,angle

if __name__ == "__main__":

    cfg = pipeline.start(config)

    dev = cfg.get_device()
    advnc_mode = rs.rs400_advanced_mode(dev)
    advnc_mode.load_json(json_string)
    depth_to_disparity = rs.disparity_transform(True) # convert to desperity form
    disparity_to_depth = rs.disparity_transform(False)
    align = rs.align(rs.stream.color)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    
    for x in range(5):
        pipeline.wait_for_frames()

    try:
        while True:
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

            box = [150,300,500,700]
            
            # depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            depth_image = np.asanyarray(depth_frame.get_data()).astype(float)
            color_image = np.asanyarray(color_frame.get_data()) #720x1280 |16:9

            dist,angle = determine_dist(box =box,frame_size= color_image.shape[0:2],object_dist= depth_frame,range_find=0.5)

            colorizer = rs.colorizer()
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            # depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.rectangle(color_image,(box[0],box[1],box[2]-box[0],box[3]-box[1]),(0,255,0),thickness=5)
            cv2.circle(color_image,(int((box[2]+box[0])/2),int((box[3]+box[1])/2)),5,(0,255,0),thickness=-1)
            cv2.putText(color_image,f'{round(dist*100,4)} cm',(int((box[2]+box[0])/2),int((box[3]+box[1])/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2) #f'{round(dist*100,4)}
            cv2.putText(color_image,f'agl = {angle} degree',(int((box[2]+box[0])/2 + 20),int((box[3]+box[1])/2 +20)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=2) #f'{round(dist*100,4)}
            # Stack both images horizontally
            images = np.hstack((color_image,depth_image))

            cv2.imshow("depth",images)

            key = cv2.waitKey(1)
            # Esc or 'q'
            if key & 0xFF == ord('q') or key ==27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()