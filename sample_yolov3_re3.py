import cv2
import glob
import numpy as np
import sys
import os.path
import warnings
import time
import copy
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.utils.linear_assignment_ import linear_assignment
from Re3.tracker.re3_tracker import Re3Tracker

basedir = os.path.dirname(__file__)

sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

dict_class={"person":1,"car":2,"motorbike":3,"bicycle":4,"truck":5,"bus":6,"backpack":7,"aeroplane":8,"stop sign":9,"horse":10,"boat":11,"traffic light":12,"train":13}
count_class=[0]*(len(dict_class)+1)

# ***********************
# path of the folder containing frames of videos as images
# PATH = "../data/siemens2/siemens2imgs/"
# PATH = "samplevideo/"
PATH = "/content/drive/My Drive/chineseimgs/"
# PATH = "occlusion_video1/"

# tracking ids intiated till now
objects=[]

def iou_intersection(a, b):
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
  
    return float(s_intsec)/(s_a + s_b - s_intsec)

# tracker - predicted
def hungarian_method(tracker,detector,iou_threshold=0.3):

    print("tracker length:",len(tracker),"detector length:",len(detector))
    iou_mat=np.zeros((len(tracker),len(detector)))
    for i in range(len(tracker)):
        for j in range(len(detector)):
            iou_mat[i,j]=iou_intersection(tracker[i],detector[j])

    matched_id=linear_assignment(-iou_mat)

    unmatched_tracker=[]
    unmatched_detector=[]

    for i in range(len(tracker)):
        if(i not in matched_id[:,0]):
            unmatched_tracker.append(i)

    for i in range(len(detector)):
        if i not in matched_id[:,1]:
            unmatched_detector.append(i)

    id_remove=[]

    for m in range(len(matched_id)):
        if(iou_mat[matched_id[m][0],matched_id[m][1]] < iou_threshold):
            unmatched_tracker.append(matched_id[m][0])
            unmatched_detector.append(matched_id[m][1])
            id_remove.append(m)

    matched_id=np.delete(matched_id,id_remove,axis=0)

    return [np.array(matched_id),np.array(unmatched_tracker),np.array(unmatched_detector)]

def get_detections():
    # *******************************
    # df=pd.read_csv('../data/siemens2/siemens2det.txt',header=None)
    df=pd.read_csv('/content/drive/My Drive/chinese.txt',header=None)
    data=df.groupby(0,as_index=False)
    detection_boundingbox=[]
    class_names=[]
    for frame,remaining in data:
        remaining.reset_index(inplace=True,drop=True)
        det_temp=[]
        class_temp=[]
        for j in range(remaining.shape[0]):
            x=remaining.iloc[j,2]
            y=remaining.iloc[j,3]
            x1=remaining.iloc[j,4]
            y1=remaining.iloc[j,5]
            det_temp.append(list([x,y,x1,y1]))
            class_temp.append(remaining.iloc[j,6])
        detection_boundingbox.append(copy.copy(det_temp))
        class_names.append(copy.copy(class_temp))
    return detection_boundingbox,class_names

if __name__ == "__main__":

    # img_file = sorted(os.listdir(PATH),key=lambda x:int(x[:-4]))
    img_file = sorted(os.listdir(PATH),key=lambda x:int(x[3:-4]))
    print(img_file)
    trackers = Re3Tracker()
    detection_boundingbox,predicted_class=get_detections()
    counter=1
    img_path=PATH+img_file[0]
    image=cv2.imread(img_path)
    # cap = cv2.VideoCapture('/content/drive/My\ Drive/chinese.avi')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # h,w,c = image.shape
    # print(cap.get(cv2.CAP_PROP_FPS),"\n\n\n")
    # out = cv2.VideoWriter('output.avi',fourcc,cap.get(cv2.CAP_PROP_FPS),(w,h))
    for i in range(len(img_file)):
        print("image number:",counter)
        img_path=PATH+img_file[i]
        image=cv2.imread(img_path)
        imageRGB=image[:,:,::-1]
        draw_bbox=[]
        pred_labels=[]

        t0=time.time()
        print("DET_BBBOX...............", i, len(detection_boundingbox))
        det_bbox=detection_boundingbox[i]
        pred_class=predicted_class[i]

        # print(det_bbox,pred_class)

        pred_bbox=trackers.multi_track(objects, imageRGB)
        #print(pred_bbox)
        if(pred_bbox is None):
            pred_bbox=[]
            print("----No prediction--------")
            trackers.get_bbox()
        matched_id,unmatched_tracker,unmatched_detector=hungarian_method(pred_bbox,det_bbox)

        #print("matched id",matched_id)

        # case3
        if(matched_id.size>0):
            draw_bbox,pred_labels=trackers.update_tracker(objects,matched_id,det_bbox,draw_bbox,pred_labels)

        # case1
        if(unmatched_detector.size>0):
            for trk in unmatched_detector:
                z=det_bbox[trk]
                unique_id=str(pred_class[trk])
                new_id = unique_id+'-'+str(count_class[dict_class[unique_id]])
                print(new_id," tracker has been created")
                objects.append(new_id)
                # updating new object with new trackin id
                trackers.add_tracker(new_id,imageRGB,z)
                count_class[dict_class[unique_id]]+=1
                #count_class[dict_class[pred_class[trk]]]+=1
        
        death_threshold = 5
        # case2
        if(unmatched_tracker.size>0):
            object_dead=[]
            print("------------unmatched tracker--------------")
            for trk in unmatched_tracker: 
                # Object might be killed
                trackers.update_miss_data(objects[trk])
                print("tracker being updated with miss data", objects[trk], "current miss value",trackers.get_miss_data(objects[trk]))

                #case 1: occluded by some object for which tracker has been initialized
                if(trackers.get_miss_data(objects[trk])<death_threshold):
                    occluded=False
                    for j in range(len(pred_bbox)):
                        if(iou_intersection(pred_bbox[j],pred_bbox[trk])>0.7 and j!=trk):
                            print("********************************occluded",(objects[j],objects[trk]),"********************")
                            occluded=True
                    if(occluded):
                        #trackers.update_occluded_data(objects[trk])
                        print("might occluded")
                    else:
                        #case 2: not occluded and not detected so use predicted bounding boxes
                        draw_bbox.append(pred_bbox[trk])
                        pred_labels.append(objects[trk])
                else:
                    print("--------This object has reached miss count greater than assigned",objects[trk])
                    #case 3: object went out of frame : EXIT
                    object_dead.append(trk)
            #print("***********************************lenght of objects to be dead******************************",len(object_dead))
            for r in sorted(object_dead, reverse=True):
                print("OBJECTDEAD........", object_dead, r)
                del_id=objects[r]
                objects.remove(del_id)
                trackers.remove_tracker(del_id)
        
        print("fps:",1/(time.time()-t0))
        print('\n')
        # cap = cv2.VideoCapture(video_path)
        

        for i in range(len(draw_bbox)):
            bbox=draw_bbox[i]
            label=pred_labels[i]
            x1=int(bbox[0])
            y1=int(bbox[1])
            x2=int(bbox[2])
            y2=int(bbox[3])
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(image,label,(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        # out.write(image)
        # cv2.imwrite("/content/output/"+str(counter)+'.jpg',image)
        counter+=1