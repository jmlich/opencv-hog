
opencv based classifier of pictures. 

- extracts feature vector from image (see feature_vector function)
-- "simple" hog
-- hog_cv
-- color histogram
- svm model is trained and evaluated
-- size of the training is 80% of data set
- list of true/false positive/negative of testing set is printed in html to stdout

The input is defined as set of pictures with same resolution (e.g. 48x48) and class name

annotation file looks like this:
```
/var/www/html/fire/sun-dataset/SUNOUT/a/art_school/sun_aprywzqxjcezkbch.jpg,non-fire
/var/www/html/fire/labelme_samples-80/000093_65.jpg,fire
```
![Sample Output](https://git.fit.vutbr.cz/imlich/opencv-hog/raw/master/sample_output.jpg)

