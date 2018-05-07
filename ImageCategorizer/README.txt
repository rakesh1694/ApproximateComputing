Dependencies:
Python 3
OpenCV

This python module can be used to calculate the mean edge value of an image as well as the minimum image shape for which the image edge map similarity value is greater than a given threshold.

Here is a sample python code describing how to use this module:

#import python module
from Image_Categorizer import Image_Categorizer
 
image_categorizer = Image_Categorizer()

# read image from the image_path and rise and crop to get 224x224 image
image = image_categorizer.read_and_crop(image_path) 
# get normalized mean edge value of original image and minimum shape for which the edge map similarity 
# value is greater than a given threshold
threshold = 0.95
norm_mean_edge, category_shape = image_categorizer.get_image_category_shape(image,threshold)
print(norm_mean_edge, category_shape)