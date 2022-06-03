# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:07:14 2022

@author: hagar
"""
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle # file to save the trained model

# 4. Make Detections with Model

with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)


# Creating the output file for the predictions    

landmarks = ['predicted_class','probablitiy_of_pred']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]


with open(r'C:\Users\hagar\OneDrive - mail.tau.ac.il\Desktop\Stage\LPC_2022\Hand decoder\Position\results\position_estimation_by_frames.csv',mode='w', newline='') as f: 
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)



# Load Video
fn = 'word_h0_01'
cap = cv2.VideoCapture(f"data/videos/{fn}.mp4")
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
        
       
        # Pose Detections
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
            
            # Name colls
            num_coords = len(results.pose_landmarks.landmark)
            col_name = []
            for val in range(1, num_coords+1):
                col_name += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]

            # Make Detections
            X = pd.DataFrame([row], columns = col_name)
            predicted_position = model.predict(X)[0]
            position_prob = model.predict_proba(X)[0]
            
            # Append prediction class and probability 
            row.insert(0, predicted_position)
            row.insert(1, position_prob)
            
            # Export to CSV
            with open('position_estimation_by_frames.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            
        except:
            pass
                        
        cv2.imshow('cued_estimated', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()