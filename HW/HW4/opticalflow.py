
# Question 3- Optical Flow #


import cv2 as cv
import os

########################### FRAME EXTRACTION PART ################################

path = "/write/the/own/video/path"
def frames_from_video(video_path):
   
   cam =  # read video file using the path


   if not  os.path.exists("/give/path/to/saving/frames"): #check the path exist
      
      #Creating a folder here to saving frames
    currframe = 0
    extraction_rate =  # set FPS values  ** save 1 frame per seconds
    while(True): 
       
       ret, frame = #read frame

       if not ret:
          break
       if extraction_rate :  # you have to  save some frames according to setted FPS
          
            #save the frame like /frame_i.png whish i is current frame

            continue
       currframe += 1
    
    cam.release()
    # destroy all windows 

frames_from_video(path)  # extracted and saved all frame at a certain FPS
       
############################### END ###############################################



############################### OPTICAL FLOW PART ################################

import numpy as np
from PIL import Image  # NECESSARY LIB 
import matplotlib.pyplot as plt
import glob

def optical_flow(frames_saved_path):

  frames_path =  # Find the all extracted frame path and sort them from first saved to last.  
  """***Hint: Hope saved frames name was including the currframe parameter. If its is, you can easly sort them."""

  frames_images = [Image.open(frames) for frames in frames_path] ## read and append all image
  
  feature_params = # Define ShiTomasi corner detection parameters here as dict form.


  lk_params = # Define the Lucas Kanade parameters as dict form.

  # Create some random colors to marking features 
  color = np.random.randint(0, 255, (100, 3)) 
  # Take first frame and find corners in it

  firs_frame = frames_images[0] # it is video's first frames.
  firs_frame = # Reshape it as 224*244 as array  data type

  flow = list()
  first_frame_gray = #convert frame from RGB to GRAY.

  p0 = # write good feature tracker and use first_frame_gray with  **feature_params** as features  
  """**Hint: look cv2 library"""

  # Create a mask image for drawing purposes
  mask = # create a zeros array mask shape of first_frame.

  for count,frame2 in  enumerate(     ): # Starting loop with second frames of the video

      frame = # Reshape it as 224*244 as array  data type.
      frame_gray = #convert frame from RGB to GRAY
      # calculate optical flow
      p1, st, err = # Calculate optical flow using Lucak Kanade technique and  ***lk_params***.

      # Select good points
      if p1 is not None: # if features exist
          good_new = #write feature
          good_old = 

      # draw the tracks
      for i, (new, old) in enumerate(zip(good_new, good_old)):
          a, b = new.ravel()
          c, d = old.ravel()
          mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
          frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

      img = cv.add(frame, mask)

      #show the image here and SAVE ımage
      first_frame_gray = frame_gray.copy()
      p0 = good_new.reshape(-1, 1, 2)
      shape = p0.shape
      if count==0:
        flow.append(np.zeros(shape))
        flow.append(p0)
      else:
        flow.append(p0)

def create_video_from_frame(path):
   """ create video from created in
   Optical_flow function"""

##################################### END ######################################