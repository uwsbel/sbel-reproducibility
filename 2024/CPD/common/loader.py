import torch
import torchvision
# import torch.nn as nn
# import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import time
import os
import glob
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re


def cityscapes_colors2labels(colors):
    h,w,_ = colors.shape

    labels = np.zeros((h, w), dtype=np.long)
    
    #label colors pulled from: https://pytorch.org/vision/main/_modules/torchvision/datasets/cityscapes.html#Cityscapes
    labels[np.all((colors[:,:,0:3] == [0, 0, 0]),axis=2)] = 0
    labels[np.all((colors[:,:,0:3] == [111, 74, 0]),axis=2)] = 5
    labels[np.all((colors[:,:,0:3] == [81, 0, 81]),axis=2)] = 6
    labels[np.all((colors[:,:,0:3] == [128, 64, 128]),axis=2)] = 7
    labels[np.all((colors[:,:,0:3] == [244, 35, 232]),axis=2)] = 8
    labels[np.all((colors[:,:,0:3] == [250, 170, 160]),axis=2)] = 9
    labels[np.all((colors[:,:,0:3] == [230, 150, 140]),axis=2)] = 10
    labels[np.all((colors[:,:,0:3] == [70, 70, 70]),axis=2)] = 11
    labels[np.all((colors[:,:,0:3] == [102, 102, 156]),axis=2)] = 12
    labels[np.all((colors[:,:,0:3] == [190, 153, 153]),axis=2)] = 13
    labels[np.all((colors[:,:,0:3] == [180, 165, 180]),axis=2)] = 14
    labels[np.all((colors[:,:,0:3] == [150, 100, 100]),axis=2)] = 15
    labels[np.all((colors[:,:,0:3] == [150, 120, 90]),axis=2)] = 16
    labels[np.all((colors[:,:,0:3] == [153, 153, 153]),axis=2)] = 17
    labels[np.all((colors[:,:,0:3] == [250, 170, 30]),axis=2)] = 19
    labels[np.all((colors[:,:,0:3] == [220, 220, 0]),axis=2)] = 20
    labels[np.all((colors[:,:,0:3] == [107, 142, 35]),axis=2)] = 21
    labels[np.all((colors[:,:,0:3] == [152, 251, 152]),axis=2)] = 22
    labels[np.all((colors[:,:,0:3] == [70, 130, 180]),axis=2)] = 23
    labels[np.all((colors[:,:,0:3] == [220, 20, 60]),axis=2)] = 24
    labels[np.all((colors[:,:,0:3] == [255, 0, 0]),axis=2)] = 25
    labels[np.all((colors[:,:,0:3] == [0, 0, 142]),axis=2)] = 26
    labels[np.all((colors[:,:,0:3] == [0, 0, 70]),axis=2)] = 27
    labels[np.all((colors[:,:,0:3] == [0, 60, 100]),axis=2)] = 28
    labels[np.all((colors[:,:,0:3] == [0, 0, 90]),axis=2)] = 29
    labels[np.all((colors[:,:,0:3] == [0, 0, 110]),axis=2)] = 30
    labels[np.all((colors[:,:,0:3] == [0, 80, 100]),axis=2)] = 31
    labels[np.all((colors[:,:,0:3] == [0, 0, 230]),axis=2)] = 32
    labels[np.all((colors[:,:,0:3] == [119, 11, 32]),axis=2)] = 33
    labels[np.all((colors[:,:,0:3] == [0, 0, 142]),axis=2)] = 26
    return labels

def cityscapes_labels2colors(labels):
    h,w = labels.shape

    colors = np.zeros((h, w, 3), dtype=np.long)
    
    #label colors pulled from: https://pytorch.org/vision/main/_modules/torchvision/datasets/cityscapes.html#Cityscapes
    colors[labels[:,:]==0] = [0, 0, 0]
    colors[labels[:,:]==5] = [111, 74, 0]
    colors[labels[:,:]==6] = [81, 0, 81]
    colors[labels[:,:]==7] = [128, 64, 128]
    colors[labels[:,:]==8] = [244, 35, 232]
    colors[labels[:,:]==9] = [250, 170, 160]
    colors[labels[:,:]==10] = [230, 150, 140]
    colors[labels[:,:]==11] = [70, 70, 70]
    colors[labels[:,:]==12] = [102, 102, 156]
    colors[labels[:,:]==13] = [190, 153, 153]
    colors[labels[:,:]==14] = [180, 165, 180]
    colors[labels[:,:]==15] = [150, 100, 100]
    colors[labels[:,:]==16] = [150, 120, 90]
    colors[labels[:,:]==17] = [153, 153, 153]
    colors[labels[:,:]==19] = [250, 170, 30]
    colors[labels[:,:]==20] = [220, 220, 0]
    colors[labels[:,:]==21] = [107, 142, 35]
    colors[labels[:,:]==22] = [152, 251, 152]
    colors[labels[:,:]==23] = [70, 130, 180]
    colors[labels[:,:]==24] = [220, 20, 60]
    colors[labels[:,:]==25] = [255, 0, 0]
    colors[labels[:,:]==26] = [0, 0, 142]
    colors[labels[:,:]==27] = [0, 0, 70]
    colors[labels[:,:]==28] = [0, 60, 100]
    colors[labels[:,:]==29] = [0, 0, 90]
    colors[labels[:,:]==30] = [0, 0, 110]
    colors[labels[:,:]==31] = [0, 80, 100]
    colors[labels[:,:]==32] = [0, 0, 230]
    colors[labels[:,:]==33] = [119, 11, 32]
    colors[labels[:,:]==26] = [0, 0, 142]
    return colors



class SemanticSegImgLoader():
    def __init__(self, data_root, max_samples=-1, crop_size=(512,512),resize_width=720,apply_transforms=False,seg_format=".png"):
        self.imgs = []
        self.seg_imgs = []
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, "imgs")
        self.seg_dir = os.path.join(data_root, "semantic_maps")
        self.max_samples = max_samples
        self.crop_size = crop_size
        self.resize_width = resize_width

        #transform parameters
        self.use_transforms = apply_transforms
        np.random.seed(1)
        self.flip_prob = 0.5
        self.max_translation = (0.2,0.2)
        self.brightness = (0.75,1.5)
        self.sharpness = (0.75,1.5)
        self.saturation = (0.75,1.5)
        self.contrast = (0.75,1.5)
        self.max_zoom = 3.0

        if(not os.path.exists(self.img_dir)):
            print("Error, img directory not found: {}".format(self.img_dir))
            exit(1)

        if(not os.path.exists(self.seg_dir)):
            print("Error, segmented img directory not found: {}".format(self.seg_dir))
            exit(1)

        self.imgs = glob.glob(self.img_dir + "/*.png")
        self.imgs.extend(glob.glob(self.img_dir + "/*.jpg"))
        if(len(self.imgs) == 0):
            print("Error: no image files (.png or .jpg) found in {}".format(
                os.path.join(self.data_root, "imgs/")))
            exit(1)

        for f in self.imgs:
            basename = os.path.splitext(os.path.basename(f))[0]
            self.seg_imgs.append(os.path.join(
                self.seg_dir, basename+seg_format))

        if(self.max_samples > 0 and self.max_samples < len(self.imgs)):
            self.imgs = self.imgs[0:self.max_samples]
            self.seg_imgs = self.seg_imgs[0:self.max_samples]

        print("Data loaded. Imgs={}, Seg Imgs={}".format(
            len(self.imgs), len(self.seg_imgs)))

        np.random.seed(1)

    def __len__(self):
        return len(self.imgs)

    def ApplyTransforms(self,img,seg_img):
        #get height and width parameters
        height,width,_ = np.asarray(img).shape

        #=== random horizontal flip ===
        if(np.random.rand() > self.flip_prob):
            #flip image horizontally
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg_img = seg_img.transpose(Image.FLIP_LEFT_RIGHT)

        #=== random zoom and crop ===
        zoom = np.random.uniform(1.0,self.max_zoom)
        img = img.resize((int(zoom*width),int(zoom*height)),resample=Image.BILINEAR)
        seg_img = seg_img.resize((int(zoom*width),int(zoom*height)),resample=Image.NEAREST)

        img,seg_img = self.crop(img,seg_img)

        # crop_pt = np.random.uniform(0.0,0.9,(2))
        # # crop_pt = [0,0]
        # crop_pt = (crop_pt * (zoom*width - width, zoom*height - height)).astype(np.int32)

        # crop_pt[0] = np.clip(crop_pt[0],0,img.size[0] - width)
        # crop_pt[1] = np.clip(crop_pt[1],0,img.size[1] - height)

        # img = img[crop_pt[1],crop_pt[1]+height,crop_pt[0],crop_pt[0]+width,:]
        # img = img.crop((crop_pt[0],crop_pt[1],crop_pt[0]+width,crop_pt[1]+height))
        # seg_img = seg_img.crop((crop_pt[0],crop_pt[1],crop_pt[0]+width,crop_pt[1]+height))

        #=== random translation ===
        #get translation parameters
        # t_x,t_y = np.random.uniform(-1,1,size=2) * self.max_translation * (height,width)
        # t_x = int(t_x)
        # t_y = int(t_y)
        # #apply translation to img
        # img = img.transform(img.size,Image.AFFINE,(1,0,t_x,0,1,t_y))
        # seg_img = seg_img.transform(img.size,Image.AFFINE,(1,0,t_x,0,1,t_y))

        #=== random brightness, hue, saturation changes ===
        brighten = ImageEnhance.Brightness(img)
        img = brighten.enhance(np.random.uniform(self.brightness[0],self.brightness[1],size=1))

        sharpen = ImageEnhance.Sharpness(img)
        img = sharpen.enhance(np.random.uniform(self.sharpness[0],self.sharpness[1],size=1))

        saturate = ImageEnhance.Color(img)
        img = saturate.enhance(np.random.uniform(self.saturation[0],self.saturation[1],size=1))

        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(np.random.uniform(self.contrast[0],self.contrast[1],size=1))

        return img,seg_img

    def resize(self,img,label):
        height = int(float(img.size[1]) / img.size[0] * self.resize_width)
        resize = transforms.Resize(size=(height,self.resize_width),interpolation=transforms.InterpolationMode.BILINEAR)
        img = resize(img)

        height = int(float(img.size[1]) / img.size[0] * self.resize_width)
        resize = transforms.Resize(size=(height,self.resize_width),interpolation=transforms.InterpolationMode.NEAREST)
        label = resize(label)

        return img,label

    def crop(self,img,label):
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.crop_size)

        img = F.crop(img, i, j, h, w)
        label = F.crop(label, i, j, h, w)

        return img,label

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load files
        img = Image.open(self.imgs[idx])
        label = Image.open(self.seg_imgs[idx])

        if(self.resize_width > 0):
            img,label = self.resize(img,label)

        if(len(self.crop_size) == 2 and self.crop_size[0] > 0 and self.crop_size[1] > 0):
            img,label = self.crop(img,label)

        if(self.use_transforms):
            img,label = self.ApplyTransforms(img,label)


        if(label.mode == 'P'):
            label = label.convert("RGB")
        label = np.asarray(label).astype(np.long)

        #convert color maps to labels if not already done
        if(len(label.shape) != 2):
            label = cityscapes_colors2labels(label)

        img = np.asarray(img).astype(np.float32)/255.0
        img = img.transpose(2,0,1)

        return img,label


class SegImgLoader():
    def __init__(self, data_root, max_samples=-1, img_format=".png", seg_format=".png"):
        self.imgs = []
        self.seg_imgs = []
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, "imgs")
        self.seg_dir = os.path.join(data_root, "seg_imgs")
        self.max_samples = max_samples

        if(not os.path.exists(self.img_dir)):
            print("Error, img directory not found: {}".format(self.img_dir))
            exit(1)

        if(not os.path.exists(self.seg_dir)):
            print("Error, segmented img directory not found: {}".format(self.seg_dir))
            exit(1)

        self.imgs = glob.glob(self.img_dir + "/*"+img_format)
        if(len(self.imgs) == 0):
            print("Error: no images found in {}".format(
                os.path.join(self.data_root, "imgs/*"+img_format)))
            exit(1)

        for f in self.imgs:
            basename = os.path.splitext(os.path.basename(f))[0]
            self.seg_imgs.append(os.path.join(
                self.seg_dir, basename+seg_format))

        if(self.max_samples > 0 and self.max_samples < len(self.imgs)):
            self.imgs = self.imgs[0:self.max_samples]
            self.seg_imgs = self.seg_imgs[0:self.max_samples]

        print("Data loaded. Imgs={}, Seg Imgs={}".format(
            len(self.imgs), len(self.seg_imgs)))

    def ConvertSegToBoxes(self, semantic_maps):

        boxes = []
        labels = []

        box_count = 0

        for c in range(1, np.max(semantic_maps[:, :, 0])+1):
            for i in range(1, np.max(semantic_maps[:, :, 1])+1):
                indices = np.where(
                    np.logical_and(semantic_maps[:, :, 1] == i, semantic_maps[:, :, 0] == c))
                if(indices[0].shape[0] > 1):
                    y0 = np.min(indices[0])
                    y1 = np.max(indices[0])
                    x0 = np.min(indices[1])
                    x1 = np.max(indices[1])

                    if(x1 > x0 and y1 > y0 ):

                        #change x0,y0,x1,y1 to normalized center (x,y) and width, height
                        x_center = .5 * (x0 + x1+1) / float(semantic_maps.shape[1])
                        y_center = .5 * (y0 + y1+1) / float(semantic_maps.shape[0])
                        x_size = (x1-x0+1) /  float(semantic_maps.shape[1])
                        y_size = (y1-y0+1) /  float(semantic_maps.shape[0])

                        # boxes.append(np.array([x0, y0, x1, y1]))
                        boxes.append(np.array([x_center, y_center, x_size, y_size]))
                        labels.append(c)
                        box_count += 1

        boxes = np.asarray(boxes)#.astype(np.int32)
        labels = np.asarray(labels).astype(np.int32)
        return boxes, labels

    def GenerateAAVBBFromSeg(self,label_format=".txt"):
        label_dir = os.path.join(self.data_root, "labels")
        if(not os.path.exists(label_dir)):
            os.mkdir(label_dir)

        for i in range(len(self.imgs)):
            #load segmentation img
            seg_img = np.array(Image.open(self.seg_imgs[i])).view(np.uint16)[:, :, :]
            # seg_img = np.array(Image.open(self.seg_imgs[i])) #.view(np.uint16)[:, :, :]

            #generate boxes and labels from segmentation img
            boxes,classes = self.ConvertSegToBoxes(seg_img)

            if(len(classes)>0):
                classes -= np.ones(classes.shape).astype(np.int32)
                classes = np.reshape(classes, (len(classes),1))
                output = np.append(classes.astype(np.str),boxes.astype(np.str),axis=1)

                basename = os.path.splitext(os.path.basename(self.imgs[i]))[0]
                file_name = os.path.join(label_dir, basename+label_format)
                np.savetxt(file_name,output,fmt='%s')

                print("Generated AABB file {}/{}".format(i+1,len(self.imgs)))

                


class ObjectDetectionImgLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, max_boxes, apply_transforms=False, max_samples=-1, box_format=".txt"):
        self.name = "Object Detection Image Loader"
        self.data_root = data_root
        self.max_boxes = max_boxes
        self.max_samples = max_samples

        self.img_dir = os.path.join(self.data_root, "imgs")
        self.label_dir = os.path.join(self.data_root, "labels")

        self.imgs = []
        self.labels = []

        #transform parameters
        self.use_transforms = apply_transforms
        np.random.seed(1)
        self.flip_prob = 0.5
        self.max_translation = (0.2,0.2)
        self.brightness = (0.75,1.5)
        self.sharpness = (0.25,4.0)
        self.saturation = (0.75,1.33)
        self.contrast = (0.75,1.33)
        self.max_zoom = 2.0

        if(not os.path.exists(self.data_root)):
            print("Error: directory not found. Data root = {}".format(self.data_root))
            exit(1)

        if(not os.path.exists(self.img_dir)):
            print("Error: directory not found. Image directory = {}".format(self.img_dir))
            exit(1)

        if(not os.path.exists(self.label_dir)):
            print("Error: directory not found. Label directory = {}".format(self.label_dir))
            exit(1)

        self.imgs = glob.glob(os.path.join(self.data_root, "imgs/*"))

        #sort the images for consistency
        
        def check_num(s):
            try:
                f = float(s)
                return f
            except:
                return s
        def frame_counter(frame):
            return [check_num(s) for s in re.split('_|\.',frame)]
        self.imgs.sort(key=frame_counter)


        if(len(self.imgs) == 0):
            print("Error: no images found in {}".format(
                os.path.join(self.data_root, "imgs/*")))
            exit(1)

        for f in self.imgs:
            basename = os.path.splitext(os.path.basename(f))[0]
            self.labels.append(os.path.join(self.label_dir, basename+box_format))

        if(self.max_samples > 0 and self.max_samples < len(self.imgs)):
            self.imgs = self.imgs[0:self.max_samples]
            self.labels = self.labels[0:self.max_samples]

        print("Data loaded. Imgs={}, Labels={}".format(
            len(self.imgs), len(self.labels)))

    def __len__(self):
        return len(self.imgs)


    def ApplyTransforms(self,img,boxes,classes):
        #get height and width parameters
        height = np.asarray(img).shape[0]
        width = np.asarray(img).shape[1]

        #=== random horizontal flip ===
        if(np.random.rand() > self.flip_prob):
            #flip image horizontally
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

            #flip boxes horizontally
            boxes_x_0 = width - 1 - boxes[:,0]
            boxes_x_1 = width - 1 - boxes[:,2]
            boxes[:,0] = boxes_x_1
            boxes[:,2] = boxes_x_0

        #=== random zoom and crop ===
        zoom = np.random.uniform(1.0,self.max_zoom)
        img = img.resize((int(zoom*width),int(zoom*height)),resample=Image.BILINEAR)
        crop_pt = np.random.uniform(0.0,0.9,(2))
        # crop_pt = [0,0]
        crop_pt = (crop_pt * (zoom*width - width, zoom*height - height)).astype(np.int32)

        crop_pt[0] = np.clip(crop_pt[0],0,img.size[0] - width)
        crop_pt[1] = np.clip(crop_pt[1],0,img.size[1] - height)

        # img = img[crop_pt[1],crop_pt[1]+height,crop_pt[0],crop_pt[0]+width,:]
        img = img.crop((crop_pt[0],crop_pt[1],crop_pt[0]+width,crop_pt[1]+height))
        boxes = (boxes * zoom).astype(np.int32)
        boxes = boxes - (crop_pt[0],crop_pt[1],crop_pt[0],crop_pt[1])
        boxes[:,0] = np.clip(boxes[:,0],0,width-1) #clip x0 value
        boxes[:,2] = np.clip(boxes[:,2],0,width-1) #clip x1 value 
        boxes[:,1] = np.clip(boxes[:,1],0,height-1) #clip y0 value
        boxes[:,3] = np.clip(boxes[:,3],0,height-1) #clip y1 value 
        for i in range(len(classes)):
            #if box moved out of image, set label to 0
            if(abs(boxes[i,0] - boxes[i,2]) < 0.5 or abs(boxes[i,1] - boxes[i,3]) < 0.5):
                classes[i] = 0  #reset label
                boxes[i,:] = np.array([0,1,0,1]) #reset to valid box coords

        #=== random translation ===
        #get translation parameters
        t_x,t_y = np.random.uniform(-1,1,size=2) * self.max_translation * (height,width)
        t_x = int(t_x)
        t_y = int(t_y)
        #apply translation to img
        img = img.transform(img.size,Image.AFFINE,(1,0,t_x,0,1,t_y))
        # img = img.rotate(1,translate=(t_x,t_y))
        #apply translation to boxes, cutting any that fully leave the image
        boxes = boxes - np.array([t_x,t_y,t_x,t_y])
        boxes[:,0] = np.clip(boxes[:,0],0,width-1) #clip x0 value
        boxes[:,2] = np.clip(boxes[:,2],0,width-1) #clip x1 value 
        boxes[:,1] = np.clip(boxes[:,1],0,height-1) #clip y0 value
        boxes[:,3] = np.clip(boxes[:,3],0,height-1) #clip y1 value 
        for i in range(len(classes)):
            #if box moved out of image, set label to 0
            if(abs(boxes[i,0] - boxes[i,2]) < 0.5 or abs(boxes[i,1] - boxes[i,3]) < 0.5):
                classes[i] = 0  #reset label
                boxes[i,:] = np.array([0,1,0,1]) #reset to valid box coords

        #=== random brightness, hue, saturation changes ===
        brighten = ImageEnhance.Brightness(img)
        img = brighten.enhance(np.random.uniform(self.brightness[0],self.brightness[1],size=1))

        sharpen = ImageEnhance.Sharpness(img)
        img = sharpen.enhance(np.random.uniform(self.sharpness[0],self.sharpness[1],size=1))

        saturate = ImageEnhance.Color(img)
        img = saturate.enhance(np.random.uniform(self.saturation[0],self.saturation[1],size=1))

        contrast = ImageEnhance.Contrast(img)
        img = saturate.enhance(np.random.uniform(self.contrast[0],self.contrast[1],size=1))

        return img,boxes,classes


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load files
        img = Image.open(self.imgs[idx])        
        boxes = np.asarray([.5, .5, .2, .2]*self.max_boxes).reshape((self.max_boxes, 4))
        classes = np.zeros(self.max_boxes)

        #if the boxes file doesn't exist, it means there were no boxes in that image
        if(not os.path.exists(self.labels[idx])):
            #ensure correct datatypes
            img = np.asarray(img)/255.0
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)[0:3, :, :]
            boxes = np.asarray([0, 0, 1, 1]*self.max_boxes).reshape((self.max_boxes, 4))
            boxes = boxes.astype(np.int32)
            classes = classes.astype(np.int64)
            return img,boxes,classes,self.imgs[idx]

        classes_and_boxes = np.loadtxt(self.labels[idx]).reshape(-1,5)

        classes[0:classes_and_boxes.shape[0]] = classes_and_boxes[:,0] + 1
        boxes[0:classes_and_boxes.shape[0],:] = classes_and_boxes[:,1:5]

        #convert normalized boxes to index-based boxes
        height = np.asarray(img).shape[0]
        width = np.asarray(img).shape[1]

        center_x = boxes[:,0].copy()
        center_y = boxes[:,1].copy()
        size_x = boxes[:,2].copy()
        size_y = boxes[:,3].copy()

        boxes[:,0] = np.clip(np.round((center_x-size_x/2) * width),0,width-2)
        boxes[:,1] = np.clip(np.round((center_y-size_y/2) * height),0,height-2)
        boxes[:,2] = np.clip(np.round((center_x+size_x/2) * width),boxes[:,0]+1,width-1)
        boxes[:,3] = np.clip(np.round((center_y+size_y/2) * height),boxes[:,1]+1,height-1)
        
        #ensure correct datatypes and formats
        boxes = boxes.astype(np.int32)
        classes = classes.astype(np.int64)

        #apply transforms
        if(self.use_transforms):
            img,boxes,classes = self.ApplyTransforms(img,boxes,classes)

        #TEMPORARILY BLUR IMAGES
        # img = img.filter(ImageFilter.GaussianBlur(radius=1))

        #ensure correct image format and channels first        
        img = np.asarray(img)/255.0
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)[0:3, :, :]

        return img, boxes, classes, self.imgs[idx]




class ObjectDetectionImgLoaderWithPreds(torch.utils.data.Dataset):
    def __init__(self, data_root, pred_dir, max_boxes, apply_transforms=False, max_samples=-1, box_format=".txt", shuffle=False):
        self.name = "Object Detection Image Loader"
        self.data_root = data_root
        self.pred_dir = pred_dir
        self.max_boxes = max_boxes
        self.max_samples = max_samples

        self.img_dir = os.path.join(self.data_root, "imgs")
        self.label_dir = os.path.join(self.data_root, "labels")
        self.pred_dir = os.path.join(self.pred_dir, "labels")

        self.imgs = []
        self.labels = []
        self.preds = []

        if(not os.path.exists(self.data_root)):
            print("Error: directory not found. Data root = {}".format(self.data_root))
            exit(1)
        if(not os.path.exists(self.img_dir)):
            print("Error: directory not found. Image directory = {}".format(self.img_dir))
            exit(1)
        if(not os.path.exists(self.label_dir)):
            print("Error: directory not found. Label directory = {}".format(self.label_dir))
            exit(1)
        if(not os.path.exists(self.pred_dir)):
            print("Error: directory not found. Prediction directory = {}".format(self.pred_dir))
            exit(1)

        self.imgs = glob.glob(os.path.join(self.data_root, "imgs/*"))

        #sort the images for consistency
        def check_num(s):
            try:
                f = float(s)
                return f
            except:
                return s
        def frame_counter(frame):
            return [check_num(s) for s in re.split('_|\.',frame)]
        self.imgs.sort(key=frame_counter)

        if(shuffle):
            np.random.shuffle(self.imgs)

        if(len(self.imgs) == 0):
            print("Error: no images found in {}".format(
                os.path.join(self.data_root, "imgs/*")))
            exit(1)

        for f in self.imgs:
            basename = os.path.splitext(os.path.basename(f))[0]
            self.labels.append(os.path.join(self.label_dir, basename+box_format))
            self.preds.append(os.path.join(self.pred_dir, basename+box_format))

        if(self.max_samples > 0 and self.max_samples < len(self.imgs)):
            self.imgs = self.imgs[0:self.max_samples]
            self.labels = self.labels[0:self.max_samples]
            self.preds = self.preds[0:self.max_samples]

        print("Data loaded. Imgs={}, Labels={}, Predictions={}".format(
            len(self.imgs), len(self.labels), len(self.preds)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load files
        img = Image.open(self.imgs[idx])        
        gt_boxes = np.asarray([.5, .5, .2, .2]*self.max_boxes).reshape((self.max_boxes, 4))
        gt_classes = np.zeros(self.max_boxes)

        pred_boxes = None #np.asarray([.5, .5, .2, .2]*self.max_boxes).reshape((self.max_boxes, 4))
        pred_classes = None #np.zeros(self.max_boxes)

        #if the boxes file doesn't exist, it means there were no boxes in that image
        if(not os.path.exists(self.labels[idx])):
            #ensure correct datatypes
            img = np.asarray(img)/255.0
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)[0:3, :, :]
            gt_boxes = np.asarray([0, 0, 1, 1]*self.max_boxes).reshape((self.max_boxes, 4))
            gt_boxes = gt_boxes.astype(np.int32)
            gt_classes = gt_classes.astype(np.int64)
            return img,[],[],[],[]

        #convert normalized boxes to index-based boxes
        height = np.asarray(img).shape[0]
        width = np.asarray(img).shape[1]

        gt_classes_and_boxes = np.loadtxt(self.labels[idx]).reshape(-1,5)
        gt_classes[0:gt_classes_and_boxes.shape[0]] = gt_classes_and_boxes[:,0] + 1
        gt_boxes[0:gt_classes_and_boxes.shape[0],:] = gt_classes_and_boxes[:,1:5]

        center_x = gt_boxes[:,0].copy()
        center_y = gt_boxes[:,1].copy()
        size_x = gt_boxes[:,2].copy()
        size_y = gt_boxes[:,3].copy()

        gt_boxes[:,0] = np.clip(np.round((center_x-size_x/2) * width),0,width-2)
        gt_boxes[:,1] = np.clip(np.round((center_y-size_y/2) * height),0,height-2)
        gt_boxes[:,2] = np.clip(np.round((center_x+size_x/2) * width),gt_boxes[:,0]+1,width-1)
        gt_boxes[:,3] = np.clip(np.round((center_y+size_y/2) * height),gt_boxes[:,1]+1,height-1)
        
        #ensure correct datatypes and formats
        gt_boxes = gt_boxes.astype(np.int32)
        gt_classes = gt_classes.astype(np.int64)


        pred_classes_and_boxes = np.loadtxt(self.preds[idx]).reshape(-1,5)
        pred_classes = pred_classes_and_boxes[:,0] + 1
        pred_boxes = pred_classes_and_boxes[:,1:5]

        center_x = pred_boxes[:,0].copy()
        center_y = pred_boxes[:,1].copy()
        size_x = pred_boxes[:,2].copy()
        size_y = pred_boxes[:,3].copy()

        pred_boxes[:,0] = np.clip(np.round((center_x-size_x/2) * width),0,width-2)
        pred_boxes[:,1] = np.clip(np.round((center_y-size_y/2) * height),0,height-2)
        pred_boxes[:,2] = np.clip(np.round((center_x+size_x/2) * width),pred_boxes[:,0]+1,width-1)
        pred_boxes[:,3] = np.clip(np.round((center_y+size_y/2) * height),pred_boxes[:,1]+1,height-1)
        
        #ensure correct datatypes and formats
        pred_boxes = pred_boxes.astype(np.int32)
        pred_classes = pred_classes.astype(np.int64)

        #ensure correct image format and channels first        
        img = np.asarray(img)/255.0
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)[0:3, :, :]

        return img, gt_boxes, gt_classes, pred_boxes, pred_classes
