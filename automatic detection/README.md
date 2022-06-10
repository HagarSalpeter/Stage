# README #

In this file you can find two different tools to detect the onsets of the hand automatically.
Both tools were based on media pipe solution, to get the values of each coordinate in in the videos:

## Landmark pose map: ##
![alt text](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

## Landmark hand map: ##
![alt text](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

-	In the decoders you can find the decoder for the shape and the decoder for the position. Each decoder has 3 files inside:
o	code – were you can find the scripts for each step
o	data – the test and training videos, the csv file with the coordinates and classification to train the decoder.
o	Results: the marked video and a csv with classification prediction in each frame.

-	In the minimal velocity you can find the script for this solution, all the visualization for the results are inside the script.
