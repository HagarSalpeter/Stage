# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:05:16 2022

@author: hagar
"""

# # 2. Capture Landmarks & Export to CSV
# <!--<img src="https://i.imgur.com/8bForKY.png">-->
# <!--<img src="https://i.imgur.com/AzKNp7A.png">-->

# In[7]:


import csv
import os
import numpy as np


# In[84]:


num_coords = len(results.pose_landmarks.landmark)
num_coords


# In[85]:


landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]


# In[86]:


landmarks


# In[87]:


with open('coords_position2.csv',mode='w', newline='') as f: 
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)


# In[97]:


class_name = "04"


# In[98]:


# Load Video
fn = 'position_04'
cap = cv2.VideoCapture(f"videos/position/{fn}.mp4")
cap.set(3,640)
cap.set(4,480)


# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        
        #4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        

        # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose]).flatten())
            
            
            row = pose_row
            
            # Append class name 
            row.insert(0, class_name)
            
            # Export to CSV
            with open('coords_position2.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
            
        except:
            pass
                        
        cv2.imshow('cued_estimated', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
