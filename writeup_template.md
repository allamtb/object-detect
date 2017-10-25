##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output/car.jpg
[image2]: ./output/notcar.jpg
[image3]: ./output/notcar-hog.jpg
[image4]: ./output/carhog.jpg
[image5]: ./output/findwindow.jpg
[image6]: ./output/cardetect.png



## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the  code cell named `use hog function and show ` of my IPython notebook ```object-detect.ipynb```.  

I started by reading in all the `cars` and `non-cars` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image4]
![alt text][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters ,then i found combine  all of the three features (the `color_hist` / `spatial_feat` / `hog_feat`) can get a better result.
here is my final parameters:

```python
output = open('svc.pkl', 'rb')
output1 = open('scale.pkl','rb')
svc = pickle.load(output)
X_scaler = pickle.load(output1)
color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True

```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the `color_hist` / `spatial_feat` / `hog_feat` three features.

first I extracted all the cars and non-cars features, and then apply
 `StandardScaler` to avoid individual features or sets of features dominating the response  of my classifier.

 then I Random Shuffling of the data using `train_test_split` with a  ratio of 0.2, which means 20 percent are test samples.

 finally I use the svm funtion to fit and predic my data:

 ```python
  svc = LinearSVC()
  svc.fit(X_train, y_train)
  svc.predict(X_test[0:n_predict])
 ```
  and the test Accuracy of SVC is  0.9896.
  it's good.



###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I choose the ystart = 400 , and yend = 600 for windows search, because of the vedio size is 720 height.
and ystart < 400 it's sky .


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my videoresult](./project_video_proc.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


1. use the heatmap to record hot_windows.
   `  heat = add_heat(heat_map,hot_windows) `
2. apply_threshold to filter false positives.
   `heat = apply_threshold(heat,threadhold)`
3. Find final boxes from heatmap using label function [wiki Connected-component labeling ](https://en.wikipedia.org/wiki/Connected-component_labeling)
    `labels = label(heatmap)`


![alt text][image6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


 the train samples are all png format and the project need to be predic is jpg , the format between these is different, one is rgb and another is rgba , it's puzzling .
 later I should learn more base theory about color format .
 If the picture size change ,my pipeline may fail because I hardcode the search range to optimize the cpu speed

 I think I can use some smart algorithm to comput the percentage ,not the hardcode value.
