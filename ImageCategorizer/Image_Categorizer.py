import numpy as np
import cv2
from numpy import linalg as LA

class Image_Categorizer:
    def __init__(self,min_mean_edge=0.0,max_mean_edge=1450.0):
        self.max_mean_edge = max_mean_edge
        self.min_mean_edge = min_mean_edge
    def get_edge_map(self,image):
        if(len(image.shape)==3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=-1)
        sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=-1)
        edge_map = (sobelx**2+sobely**2)**0.5
        return edge_map
    def get_mean_edge(self,image):
        edge = self.get_edge_map(image)
        size = image.shape[0]*image.shape[1]
        mean_edge = sum(sum(edge))/size
        return mean_edge
    def normalize_value(self,mean_edge):
        normalized_mean_edge = (mean_edge-self.min_mean_edge)/(self.max_mean_edge-self.min_mean_edge)
        if(normalized_mean_edge<0.0):
            normalized_mean_edge = 0.0
        if(normalized_mean_edge>1.0):
            normalized_mean_edge = 1.0
        return normalized_mean_edge
    def get_normalized_mean_edge(self,image):
        mean_edge = self.get_mean_edge(image)
        normalized_mean_edge = self.normalize_value(mean_edge)
        return normalized_mean_edge
    def get_normalized_mean_edge_from_edge_map(self,edge_map):
        size = edge_map.shape[0]*edge_map.shape[1]
        mean_edge = sum(sum(edge_map))/size
        normalized_mean_edge = (mean_edge-self.min_mean_edge)/(self.max_mean_edge-self.min_mean_edge)
        if(normalized_mean_edge<0.0):
            normalized_mean_edge = 0.0
        if(normalized_mean_edge>1.0):
            normalized_mean_edge = 1.0
        return normalized_mean_edge
    def get_image_similarity_from_edge_map(self,edge_map1,edge_map2):
        edge_similarity = sum(sum(edge_map1*edge_map2))/(LA.norm(edge_map1)*LA.norm(edge_map2))
        return edge_similarity
    def get_image_similarity(self,image1,image2):
        edge_map1 = self.get_edge_map(image1)
        edge_map2 = self.get_edge_map(image2)
        edge_similarity = self.get_image_similarity_from_edge_map(edge_map1,edge_map2)
        return edge_similarity
    def read_and_crop(self,image_path):
        img = cv2.imread(image_path)
        (img_H, img_W, img_C) = img.shape
        #print(img_H, img_W, img_C)
        ResizeShape = (int(img_W*224/img_H),224) if img_H < img_W else (224,int(img_H*224/img_W))
        #print(ResizeShape)
        resize_img = cv2.resize(img,ResizeShape,interpolation = cv2.INTER_LINEAR )
        (resizeimg_H, resizeimg_W, resizeimg_C) = resize_img.shape
        CutWidth = int((resizeimg_H-224)/2) if resizeimg_H>224 else int((resizeimg_W-224)/2)
        #print(CutWidth,resizeimg_H, resizeimg_W)
        resize_img = resize_img[CutWidth:CutWidth + 224, 0:224, ::] if resizeimg_H > resizeimg_W else             resize_img[0:224, CutWidth:CutWidth + 224, ::]
        return resize_img
    def get_modified_image(self,ori_img,downsampled_shape):
        ori_shape = (ori_img.shape[0],ori_img.shape[0])
        category_shape = (downsampled_shape,downsampled_shape)
        downsampled_img = cv2.resize(ori_img,category_shape,interpolation = cv2.INTER_LINEAR )
        rec_img = cv2.resize(downsampled_img,ori_shape,interpolation = cv2.INTER_LINEAR )
        return rec_img
    def get_image_category_shape(self,image,threshold):
        ori_edge_map = self.get_edge_map(image)
        norm_mean_edge = self.get_normalized_mean_edge_from_edge_map(ori_edge_map)
        category_sizes = [224, 192, 160, 128, 112, 96, 80, 64, 56]
        curr_idx = 1
        while(curr_idx<len(category_sizes)):
            rec_img = self.get_modified_image(image,category_sizes[curr_idx])
            rec_edge_map = self.get_edge_map(rec_img)
            edge_similarity = self.get_image_similarity_from_edge_map(ori_edge_map,rec_edge_map)
            if(edge_similarity<threshold):
                break
            curr_idx += 1
        category_shape = category_sizes[curr_idx-1]
        return norm_mean_edge,category_shape