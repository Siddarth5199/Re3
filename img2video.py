import cv2
import os
import re
print("start")
image=cv2.imread('/home/hdd1/midhilesh/TANISHQ/yolov3+re3/output_img/occlusion_death10/0.jpg')
print(image.shape)
img_file=sorted(os.listdir('/home/hdd1/midhilesh/TANISHQ/yolov3+re3/output_img/occlusion_death10'),key=lambda x:int(re.findall('[0-9]+',x)[0]))
print(img_file)
output_file="/home/hdd1/midhilesh/TANISHQ/yolov3+re3/output_img/occlusion_death10.avi"
# vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1280,720))
# vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M','J','P','G'), 15, (960, 540))
vid_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1920, 1080))

for img in img_file:
    image=cv2.imread('/home/hdd1/midhilesh/TANISHQ/yolov3+re3/output_img/occlusion_death10/'+img)
    print(img, image.shape)
    vid_writer.write(image)
