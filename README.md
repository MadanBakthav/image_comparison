# image_comparison
An algorithm to remove similar images in a dataset

1080 Images from 4 different camera-views are stored in a folder. To train a model, the images looking similar need to be removed. If all the images are checked for similarity, it requires 582660 combinations. All the camera's views are different. Therefore, it is not required to check the image from different cameras. Initially the images are separated according to camera. By doing this, many unwanted combinations are avoided. In addition, The algorithm is built to avoid the skip the comparison of images which are found to be similar in previous checks. This also helps the program to run faster. 

All these scripts were run in a hardware with RAM 8GB and intel i5 processor (4 cores).
If all the combinations are checked for similarity, it would have taken more than two hours. Without using parallel computation, the algorithm takes only 185 seconds. 
With the use of parallel computation, it takes 106 seconds


# Command for execution
python name_of_script.py -i "path_to_folder"

# Important parameters

1. Score - The minimum difference in the images should be more than 10% of the image area (H * W). If the score is more than 10%, it is considered to be different image. It can be changed.
2. Minimum area of contour considered for checking the contours for scoring is 5% of the image area.

# Dependencies
1. imutils
2. openCV
3. numpy
4. tqdm
5. glob
6. itertools
