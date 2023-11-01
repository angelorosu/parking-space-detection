import cv2
from util import *
import numpy as np


def calc_diff(im1, im2):
    # Calculate mean pixel values for each image
    mean_im1 = np.mean(im1)
    mean_im2 = np.mean(im2)

    # Compute absolute differences in mean pixel values
    abs_diff_im1 = np.abs(mean_im1)
    abs_diff_im2 = np.abs(mean_im2)

    # Return difference in absolute differences
    return abs_diff_im1 - abs_diff_im2



mask = '/Users/angelor/Projects/Parking_detector/ParkingCounter/data/mask_1920_1080.png'


video_path = '/Users/angelor/Projects/Parking_detector/ParkingCounter/data/parking_1920_1080_loop.mp4'


mask = cv2.imread(mask,0)

cap = cv2.VideoCapture(video_path)


connected_components = cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)

spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None

frame_nmr = 0
ret = True
step = 30
while ret:
    ret,frame = cap.read()
    
    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_index,spot in enumerate(spots):
            x1,y1,w,h = spot
            
            spots_crop = frame[y1:y1+h,x1:x1+w, :]
            diffs[spot_index] = calc_diff(spots_crop,previous_frame[y1:y1+h,x1:x1+w, :])
    
    
    if frame_nmr % step == 0: 
        for spot_index,spot in enumerate(spots):
            x1,y1,w,h = spot
            
            spots_crop = frame[y1:y1+h,x1:x1+w, :]
            spot_status = empty_or_not(spots_crop)
            spots_status[spot_index]= spot_status
            

    if frame_nmr % step == 0:
        previous_frame = frame.copy()
        
    for spot_index,spot in enumerate(spots):
        spot_status= spots_status[spot_index]
        x1,y1,w,h = spots[spot_index]
        
        
        if spot_status:
            frame = cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)
        else:
            frame = cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2)
    
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
    frame_nmr += 1
    
    
    
cap.release()  
cv2.destroyAllWindows()
