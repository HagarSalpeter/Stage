#!/usr/bin/env python
# coding: utf-8

# ### 0. Install and Import Dependencies
# 

# In[1]:


get_ipython().run_line_magic('config', "IPCompleter.greedy=True #I'm not sure what is this")


# In[5]:


get_ipython().system('pip install mediapipe opencv-python pandas scikit-learn')


# In[2]:


import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import statistics


# In[3]:


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
mp_pose = mp.solutions.pose



# # 1. Initializing Detections

#  # Landmarks map #
# <img src=https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png />
# 
# relevant_landmarks = [0,16,18,20,22]

# In[4]:


cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


# This is the syntax to get the value of specific coord
# 
# point 16: (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]) 
# 
# point 0:
# (results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE])
# 
# point 20:
# (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]) 
# 

# # 2. Capture Landmarks & Export to CSV

# In[5]:


import csv
import os
import numpy as np


# In[8]:


#create a file with all the relevant coordinates
#if results.right_hand_landmarks.landmark:
    #num_right_hand_landmark = len(results.right_hand_landmarks.landmark)
    
num_coords = len(results.pose_landmarks.landmark) #+len(results.face_landmarks.landmark)+num_right_hand_landmark
num_coords


# In[9]:


landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


# In[10]:


data_file = 'all_the_coords.csv' # need to save the file in the data folder

with open(data_file,mode='w', newline='') as f: 
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)


# In[11]:


classes = ['position_00','position_01','position_02','position_03','position_04'] #put the names of the files here


# In[13]:


for label in classes:
    # Load Video
    fn = label
    class_name = label[-2:] # only the number of the video
    #change this path to relative path:
    cap = cv2.VideoCapture(os.path.join('..', 'data', 'training_videos', f'{fn}.mp4'))
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
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                row = pose_row

                # Append class name 
                row.insert(0, class_name)

                # Export to CSV
                with open(data_file, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

            except:
                pass

            cv2.imshow('cued_estimated', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    print(f'{label} was learned')
    cap.release()
    cv2.destroyAllWindows()


# Edit the csv to the relevant one for training

# In[34]:


import pandas as pd


# In[32]:


df = pd.read_csv(r"C:\Users\hagar\cuedspeech_perception\automatic detection\decoders\position\code\notebook_files\train_data.csv")


# In[36]:


df['d_x'] = df['x1'] - df['x17']
df['d_y'] = df['y1'] - df['y17']
df['d_z'] = df['z1'] - df['z17']
df['alpha'] = df['d_y']/df['d_x']
df['distance_1_17'] = np.sqrt((df['d_x'])**2 + (df['d_y'])**2 + (df['d_z'])**2)
df_distance = df.filter(['class','d_x','d_y','d_z','alpha','distance_1_17'], axis=1)


# In[37]:


df_distance

