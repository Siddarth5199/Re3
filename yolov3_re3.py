import cv2
import glob
import numpy as np
import sys
import os.path
import warnings
import time
warnings.filterwarnings("ignore")

from sklearn.utils.linear_assignment_ import linear_assignment
from darknet.python.darknet import localize
from Re3.tracker import re3_tracker

basedir = os.path.dirname(__file__)
print("basedir",basedir)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

dict_class={"person":1,"car":2,"motorbike":3,"bicycle":4,"truck":5,"bus":6,"traffic light":7,"train":8}
count_class=[0]*(len(dict_class)+1)

# path of the folder containing frames of videos as images
PATH = "input_image/"

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

	for m in matched_id:
		if(iou_mat[m[0],m[1]] < iou_threshold):
			unmatched_tracker.append(m[0])
			unmatched_detector.append(m[1])

	return [np.array(matched_id),np.array(unmatched_tracker),np.array(unmatched_detector)]

if __name__ == "__main__":

    img_file=sorted(os.listdir(PATH))
    #net = load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
    #meta = load_meta(b"cfg/coco.data")
    trackers = re3_tracker.Re3Tracker()

    for img in img_file:
        img_path=PATH+img
        image=cv2.imread(img_path)
        imageRGB=image[:,:,::-1]
        draw_bbox=[]

        t0=time.time()
        # will return detections
        output=localize(img_path)
        det_bbox=output[0][2]
        pred_class=output[0][0]
        #print("detected boxes",det_bbox)

        pred_bbox=trackers.multi_track(objects, imageRGB)
        if(pred_bbox is None):
            pred_bbox=[]
        matched_id,unmatched_tracker,unmatched_detector=hungarian_method(pred_bbox,det_bbox)

        #print("matched id",matched_id)

        # case3
        if(matched_id.size>0):
            draw_bbox=trackers.update_tracker(objects,matched_id,det_bbox,draw_bbox)

        # case1
        if(unmatched_detector.size>0):
            for trk in unmatched_detector:
                z=det_bbox[trk]
                unique_id=pred_class[trk]+'-'+str(count_class[dict_class[pred_class[trk]]])
                objects.append(unique_id)
                # updating new object with new trackin id
                trackers.add_tracker(unique_id,imageRGB,z)
                count_class[dict_class[pred_class[trk]]]+=1
        
        death_threshold = 5
        # case2
        if(unmatched_tracker.size>0):
            for trk in unmatched_tracker: 
                # Object might be killed
                trackers.update_miss_data(objects[trk])

                #case 1: occluded by some object for which tracker has been initialized
                if(trackers.get_miss_data(objects[trk])<death_threshold):
                    occluded=False
                    for j in range(len(pred_bbox)):
                        if(iou_intersection(pred_bbox[j],pred_bbox[trk])>0.7 and j!=trk):
                            occluded=True
                    if(occluded):
                        #trackers.update_occluded_data(objects[trk])
                        print("might occluded")
                    else:
                        #case 2: not occluded and not detected so use predicted bounding boxes
                        draw_bbox.append(pred_bbox[trk])
                else:
                    #case 3: object went out of frame : EXIT
                    del_id=objects[trk]
                    objects.remove(del_id)
                    trackers.remove_tracker(del_id)
        print("fps:",1/(time.time()-t0))
            




    

