# Stage repository #

In this repository you can find all the scripts I generated during my internship.
The repository is divided to four main files:

## [1. Automatic detection:](https://github.com/HagarSalpeter/Stage/tree/main/automatic%20detection)
In this file you can find two different tools to detect the onsets of the hand automatically.
Both tools were based on media pipe solution, to get the values of each coordinate in in the videos:

### Landmark pose map: ###
![alt text](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)

### Landmark hand map: ###
![alt text](https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)

-	In the decoders you can find the decoder for the shape and the decoder for the position. Each decoder has 3 files inside:
o	code – were you can find the scripts for each step
o	data – the test and training videos, the csv file with the coordinates and classification to train the decoder.
o	Results: the marked video and a csv with classification prediction in each frame.

-	In the minimal velocity you can find the script for this solution, all the visualization for the results are inside the script.



## [2. Basic useful functions](https://github.com/HagarSalpeter/Stage/tree/main/basic%20useful%20functions)
In this file you can find functions to:
 - Get the phonological LPC form of a word or a whole sentence
 - Get the number of LPC gestures of a word or a whole sentence
 - Get the LPC code for a word per gesture

you can initialize this script at the begging of new ones to call this functions. all the details are in the script.

## [3. Material preparation](https://github.com/HagarSalpeter/Stage/tree/main/material%20preparation)
- In this part you can find the all the scripts to generate the materials of the sentences and the words.
- In the "Final_Materials.xlsx" you can find the final materials that we used for filming the materials.

## [4. Automatic video editing](https://github.com/HagarSalpeter/Stage/tree/main/automatic%20video%20editing)
- In this file you can find the script that was apllied on all the videos to adjust the colors and add fade in and fade out effect in standartazing way.
- In the script there is a way to apply the pipline to all the files in a certain folder in a batch
