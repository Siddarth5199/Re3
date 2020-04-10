import cv2
import glob
import numpy as np
import sys
import os.path
import pandas as pd
import copy
from Re3.tracker import re3_tracker

def get_detections():
    df=pd.read_csv('gt.txt',header=None)
    data=df.groupby(0,as_index=False)
    detection_boundingbox=[]
    class_names=[]
    for frame,remaining in data:
        remaining.reset_index(inplace=True,drop=True)
        det_temp=[]
        class_temp=[]
        for j in range(remaining.shape[0]):
            det_temp.append(list(remaining.iloc[j,[2,3,4,5]]))
            class_temp.append(remaining.iloc[j,1])
        detection_boundingbox.append(copy.copy(det_temp))
        class_names.append(copy.copy(class_temp))
    return detection_boundingbox,class_names

detection_boundingbox,predicted_class=get_detections()

PATH = "input_image/"
img_file=sorted(os.listdir(PATH))
trackers = re3_tracker.Re3Tracker()

store_id=[]
counter=0
save_class=0
for i in range(len(img_file)):
    img_path=PATH+img_file[i]
    image=cv2.imread(img_path)
    imageRGB=image[:,:,::-1]
    det_bbox=detection_boundingbox[i]
    pred_class=predicted_class[i]
    if(i==0):
        for j in range(len(det_bbox)):
            store_id.append(pred_class[j])
            save_class=pred_class[j]
            det_bbox[j][2]=det_bbox[j][0]+det_bbox[j][2]
            det_bbox[j][3]=det_bbox[j][1]+det_bbox[j][3]
            trackers.add_tracker(pred_class[j],imageRGB,det_bbox[j])
        #trackers.get_unique_id()
        print("bbox being fed",trackers.get_bbox())
    pred_bbox=trackers.multi_track(store_id,imageRGB)
    print(pred_bbox)
    for i in range(len(pred_bbox)):
        bbox=pred_bbox[i]
        x1=int(bbox[0])
        y1=int(bbox[1])
        x2=int(bbox[2])
        y2=int(bbox[3])
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imwrite("/workspace/midhilesh/yolov3+re3/output_img/"+str(counter)+'.jpg',image)
    counter+=1
    print("predicted value",pred_bbox)
    trackers.get_bbox()

