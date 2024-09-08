import os
import glob
from random import random
# from requests import patch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import time
import sys
from tqdm import tqdm
import random

import scipy as sp
import torch

from common.loader import cityscapes_labels2colors
from common.logger import Logger
from common.cityscapes_labels import *

def random_crop(img,label,pred,crop_size):
    i, j, h, w = transforms.RandomCrop.get_params(img, output_size=crop_size)
    out_img = F.crop(img, i, j, h, w)
    out_label = F.crop(label, i, j, h, w)
    out_pred = F.crop(pred, i, j, h, w)
    return out_img,out_label,out_pred

class SegmentationComparitor():
    def __init__(self,output_path="output",write=False,nameA="Dataset A",nameB="Dataset B"):
        self.nameA = nameA
        self.nameB = nameB

        self.a_imgs = None
        self.a_gt_labels = None
        self.a_pred_labels = None

        self.b_imgs = None
        self.b_gt_labels = None
        self.b_pred_labels = None

        self.logger = Logger(filename=os.path.join(output_path,"results.txt"),write=write,display=True)

    def load_dataset(self,path,predictions,n):
        
        # print(pred_path)
        img_path = "imgs"
        gt_path = "semantic_maps"
        pred_path = "preds"

        img_path = os.path.join(path,img_path)
        gt_path = os.path.join(path,gt_path)
        pred_path = os.path.join(predictions,pred_path)

        #ensure dataset paths exist
        if(not os.path.exists(img_path)):
            raise FileNotFoundError(img_path)
        if(not os.path.exists(gt_path)):
            raise FileNotFoundError(gt_path)
        if(not os.path.exists(pred_path)):
            raise FileNotFoundError(pred_path)

        self.a_imgs = glob.glob(os.path.join(img_path,"*.png"))
        self.a_imgs.sort()
        self.a_imgs = self.a_imgs[0:n]

        #semantic maps have same name as img
        self.a_gt_labels = []
        self.a_pred_labels = []
        for i,img in enumerate(self.a_imgs):
            img_name = os.path.basename(img).split('.')[0]

            # print("basename:",img_name)

            gt_file = os.path.join(gt_path,img_name+".png")
            pred_file = os.path.join(pred_path,img_name+"_prediction.png")

            if(not os.path.exists(gt_file)):
                self.logger.log("Error: missing semantic map: {}. Needed for image: {}".format(gt_file,img))
                exit(1)

            if(not os.path.exists(pred_file)):
                self.logger.log("Error: missing prediction map: {}. Needed for image: {}".format(pred_file,img))
                exit(1)

            self.a_gt_labels.append(gt_file)
            self.a_pred_labels.append(pred_file)

        self.logger.log("Loaded datasets. Imgs: {}, GT labels: {}, Pred Labels: {}".format(len(self.a_imgs),len(self.a_gt_labels),len(self.a_pred_labels)))        

    def load_img_list(self, img_list):
        imgs = []
        for file_name in img_list:
            imgs.append(np.asarray(Image.open(file_name),np.int32))
        return np.asarray(imgs)


    def map_to_trainid(self, gt_labels, pred_labels):
        # self.logger.log("Converting patches to training id to correct for any ignored labels")
        # for i,img in enumerate(tqdm(gt_labels)):
        for i,img in enumerate(gt_labels):

            gt_copy = gt_labels[i].copy()
            pred_copy = pred_labels[i].copy()
            for k, v in label2trainid.items():  
                
                #convert gt_labels
                binary_mask = (gt_copy == k)
                gt_labels[i][binary_mask] = v

            #mask ignored label to say it is same
            pred_labels[i][gt_labels[i]==255] = 255

        return gt_labels,pred_labels

    def map_to_color(self, mask):

        h,w = mask.shape
        color_mask = np.zeros((h,w,3),dtype=np.uint8)

        for k, v in trainId2color.items():  

            #convert gt_labels
            binary_mask = (mask == k)
            color_mask[binary_mask] = v

        return color_mask


    def load_patches(self,pathA,pathB,n,shuffle=False):

        img_path = "img_patches"
        gt_path = "gt_patches"
        pred_path = "pred_patches"

        a_img_path = os.path.join(pathA,img_path)
        a_gt_path = os.path.join(pathA,gt_path)
        a_pred_path = os.path.join(pathA,pred_path)

        b_img_path = os.path.join(pathB,img_path)
        b_gt_path = os.path.join(pathB,gt_path)
        b_pred_path = os.path.join(pathB,pred_path)

        #ensure dataset paths exist
        if(not os.path.exists(a_img_path)):
            raise FileNotFoundError(a_img_path)
        if(not os.path.exists(a_gt_path)):
            raise FileNotFoundError(a_gt_path)
        if(not os.path.exists(a_pred_path)):
            raise FileNotFoundError(a_pred_path)
        if(not os.path.exists(b_img_path)):
            raise FileNotFoundError(b_img_path)
        if(not os.path.exists(b_gt_path)):
            raise FileNotFoundError(b_gt_path)
        if(not os.path.exists(b_pred_path)):
            raise FileNotFoundError(b_pred_path)

        self.a_imgs = glob.glob(os.path.join(a_img_path,"*.png"))
        self.a_imgs.sort()

        p = np.random.permutation(len(self.a_imgs))
        if(shuffle):
            self.a_imgs = [self.a_imgs[x] for x in p]

        self.a_imgs = self.a_imgs[0:n]
        self.a_imgs = self.load_img_list(self.a_imgs)

        self.a_gt_labels = glob.glob(os.path.join(a_gt_path,"*.png"))
        self.a_gt_labels.sort()
        if(shuffle):
            self.a_gt_labels = [self.a_gt_labels[x] for x in p]
        self.a_gt_labels = self.a_gt_labels[0:n]
        self.a_gt_labels = self.load_img_list(self.a_gt_labels)

        self.a_pred_labels = glob.glob(os.path.join(a_pred_path,"*.png"))
        self.a_pred_labels.sort()
        if(shuffle):
            self.a_pred_labels = [self.a_pred_labels[x] for x in p]
        self.a_pred_labels = self.a_pred_labels[0:n]
        self.a_pred_labels = self.load_img_list(self.a_pred_labels)

        self.a_gt_labels,self.a_pred_labels = self.map_to_trainid(self.a_gt_labels,self.a_pred_labels)
        
        self.b_imgs = glob.glob(os.path.join(b_img_path,"*.png"))
        self.b_imgs.sort()
        p = np.random.permutation(len(self.b_imgs))
        if(shuffle):
            self.b_imgs = [self.b_imgs[x] for x in p]
        self.b_imgs = self.b_imgs[0:n]
        self.b_imgs = self.load_img_list(self.b_imgs)

        self.b_gt_labels = glob.glob(os.path.join(b_gt_path,"*.png"))
        self.b_gt_labels.sort()
        if(shuffle):
            self.b_gt_labels = [self.b_gt_labels[x] for x in p]
        self.b_gt_labels = self.b_gt_labels[0:n]
        self.b_gt_labels = self.load_img_list(self.b_gt_labels)

        self.b_pred_labels = glob.glob(os.path.join(b_pred_path,"*.png"))
        self.b_pred_labels.sort()
        if(shuffle):
            self.b_pred_labels = [self.b_pred_labels[x] for x in p]
        self.b_pred_labels = self.b_pred_labels[0:n]
        self.b_pred_labels = self.load_img_list(self.b_pred_labels)

        self.b_gt_labels,self.b_pred_labels = self.map_to_trainid(self.b_gt_labels,self.b_pred_labels)

        self.logger.log("Loaded datasets. \
            \n\t A: Imgs: {}, GT labels: {}, Pred Labels: {} \
            \n\t B: Imgs: {}, GT labels: {}, Pred Labels: {}".format(\
            len(self.a_imgs),len(self.a_gt_labels),len(self.a_pred_labels),
            len(self.b_imgs),len(self.b_gt_labels),len(self.b_pred_labels)))        

    def generate_patches(self,patch_size=(64,64),patches_per_img=8,output_path="output_patches"):


        if(len(patch_size) != 2 or patch_size[0]<1 or patch_size[1]<1):
            self.logger.log("Invalid patch size: {}. Must be positive tuple of length 2".format(patch_size))

        if(self.a_imgs == None or self.a_gt_labels==None or self.a_pred_labels == None):
            self.logger.log("Must load images (imgs, gt_labels, pred_labels) before generating patches")
            return -1

        img_path = os.path.join(output_path,"img_patches")
        label_path = os.path.join(output_path,"gt_patches")
        pred_path = os.path.join(output_path,"pred_patches")
        if(not os.path.exists(img_path)):
            os.mkdir(img_path)
        if(not os.path.exists(label_path)):
            os.mkdir(label_path)
        if(not os.path.exists(pred_path)):
            os.mkdir(pred_path)

        for i,img_name in enumerate(tqdm(self.a_imgs)):

            #load an img, label, pred 
            img = Image.open(self.a_imgs[i])
            label = Image.open(self.a_gt_labels[i])
            pred = Image.open(self.a_pred_labels[i])
            
            for j in range(patches_per_img):
                #random crop a section out of each
                patch_img,patch_label,patch_pred = random_crop(img,label,pred,patch_size)

                #save patch
                patch_img.save(os.path.join(img_path,"img_patch_{}_{}.png".format(i,j)))
                patch_label.save(os.path.join(label_path,"gt_patch_{}_{}.png".format(i,j)))
                patch_pred.save(os.path.join(pred_path,"pred_patch_{}_{}.png".format(i,j)))

    def evaluate_dataset_performance(self,args):
        
        per_image_accuracy = []
                        
        for i,img_name in enumerate(tqdm(self.a_imgs)):

            #load an img, label, pred 
            # img = np.array(Image.open(self.a_imgs[i]))
            label = np.array(Image.open(self.a_gt_labels[i]),dtype=np.int16)
            pred = np.array(Image.open(self.a_pred_labels[i]),dtype=np.int16)
            
            label,pred = self.map_to_trainid([label],[pred])
            
            label = label[0]
            pred = pred[0]
            
            acc = np.sum(np.equal(label,pred),axis=(0,1)) / float(label.shape[0]*label.shape[1])
            
            per_image_accuracy.append(acc)
            
        print(f"Total acc = {np.mean(per_image_accuracy):.3f}")

    def calculate_performance(self,gt_patches,pred_patches,metric):
        num_train_classes=19
        performance = []
        if(metric=="IOU"):

            for i in range(len(gt_patches)):
                gt_patch = gt_patches[i]
                pred_patch = pred_patches[i]

                gt_mask = (gt_patch >= 0) & (gt_patch < num_train_classes)

                data = num_train_classes * gt_patch[gt_mask].flatten().astype(int) + pred_patch[gt_mask].flatten()

                hist_data = np.bincount(data, minlength=num_train_classes**2)
                hist_data = hist_data.reshape(num_train_classes, num_train_classes)

                divisor = hist_data.sum(axis=1) + hist_data.sum(axis=0) - np.diagonal(hist_data, axis1=0, axis2=1)
                mean_iou = np.nanmean(np.diagonal(hist_data, axis1=0, axis2=1) / divisor)

                performance.append(mean_iou)
            performance = np.asarray(performance)

        else:
            performance = np.sum(gt_patches == pred_patches,axis=(1,2)) / float(gt_patches.shape[1]*gt_patches.shape[2])
        
        return performance

    def find_near_patches(self,gt_ref,gt_patches,threshold,metric):

        # t0 = time.time()
        
        dist_list = []
        # ids = []

        if(metric=="IOU"):
            num_train_classes=19

            gt_patches_copy = gt_patches.copy()
            ref_copy = gt_ref.copy()

            gt_patches_copy[gt_patches_copy==255]=num_train_classes
            ref_copy[ref_copy==255]=num_train_classes

            patch_pixels = gt_patches_copy.shape[1]*gt_patches_copy.shape[2]

            data = (num_train_classes+1) * gt_patches_copy.reshape(-1,patch_pixels).astype(int) + ref_copy.reshape(-1,patch_pixels)

            hist_data = np.apply_along_axis(np.bincount,axis=1,arr=data,minlength=(num_train_classes+1)**2)
            hist_data = hist_data.reshape(-1,num_train_classes+1, num_train_classes+1)

            divisor = hist_data.sum(axis=2) + hist_data.sum(axis=1) - np.diagonal(hist_data, axis1=1, axis2=2)
            dist_list = np.nanmean(np.diagonal(hist_data, axis1=1, axis2=2) / divisor,axis=1)

        else:
            dist_list = np.sum(gt_patches == gt_ref,axis=(1,2)) / float(gt_ref.shape[0]*gt_ref.shape[1])

        ids = np.where(dist_list>threshold)[0]

        return ids,dist_list[ids]

    def compare_patches(self,args):
        
        #set parameters from args
        threshold=args.patch_threshold
        vis=args.vis
        save=args.save
        output_dir=args.output_path
        sample_max=args.save_max
        min_batch_size=args.min_batch_size
        use_weighting=args.weight
        perf_measure=args.perf_measure
        ntest=args.ntest

        reduce = args.reduce_gt
        reduced_size=(32,32)


        self.logger.log("Comparing patches. Save= {}, Vis={}".format(save,vis))
        sys.stdout.flush()

        #only do this here as it downloads vgg net and loads onto GPU
        if(args.skvd):
            from metrics.ml_metrics import sKVD

        if(save and not os.path.exists(output_dir)):
            os.makedirs(output_dir, exist_ok=True)

        if(save and not os.path.exists(os.path.join(output_dir,"overlap"))):
            os.mkdir(os.path.join(output_dir,"overlap"))

        if(save and not os.path.exists(os.path.join(output_dir,"nonoverlap"))):
            os.mkdir(os.path.join(output_dir,"nonoverlap"))
        
        patch_w1 = []
        patch_mean = []
        batch_size = []
        aligned_patches = []

        #calculate performance for the predictions
        self.a_perf = self.calculate_performance(self.a_gt_labels,self.a_pred_labels,perf_measure)
        self.b_perf = self.calculate_performance(self.b_gt_labels,self.b_pred_labels,perf_measure)
        used_A = np.zeros(len(self.a_perf))
        used_B = np.zeros(len(self.b_perf))

        if(reduce):
            resized_labels = []
            for g,gt in enumerate(self.a_gt_labels):
                resized_gt = np.asarray(Image.fromarray(gt).resize(reduced_size,Image.NEAREST))
                resized_labels.append(resized_gt)
            self.a_gt_labels = np.asarray(resized_labels)

            resized_labels = []
            for g,gt in enumerate(self.b_gt_labels):
                resized_gt = np.asarray(Image.fromarray(gt).resize(reduced_size,Image.NEAREST))
                resized_labels.append(resized_gt)
            self.b_gt_labels = np.asarray(resized_labels)

        weights = []

        saved_sample_counter=0
        
        to_test = random.sample(range(len(self.a_imgs)),ntest)

        #for each patch, findings its near neighbors        
        for i in tqdm(to_test):
        
            similar_ids_A,similarities_A = self.find_near_patches(self.a_gt_labels[i], self.a_gt_labels, threshold, perf_measure)
            similar_ids_B,similarities_B = self.find_near_patches(self.a_gt_labels[i], self.b_gt_labels, threshold, perf_measure)


            if(len(similar_ids_A)>=min_batch_size and len(similar_ids_B)>=min_batch_size):
                used_A[similar_ids_A] = 1
                used_B[similar_ids_B] = 1

                weights.append(len(similar_ids_A))

                mb_perf_A = self.a_perf[similar_ids_A]
                mb_perf_B = self.b_perf[similar_ids_B]

                mb_w1 = wasserstein_distance(mb_perf_A,mb_perf_B)
                mb_mean_diff = np.mean(mb_perf_A) - np.mean(mb_perf_B)

                most_similar_B = similar_ids_B[np.argmax(similarities_B)]


                img_a0 = self.a_imgs[i,:,:,:] #.astype(np.float32)/127.0 - 1
                img_b0 = self.b_imgs[most_similar_B,:,:,:] #.astype(np.float32)/127.0 - 1

                img_a0 = img_a0.transpose(2,0,1)
                img_b0 = img_b0.transpose(2,0,1)

                if(args.skvd):
                    aligned_patches.append([img_a0,img_b0])

                patch_w1.append(mb_w1)
                patch_mean.append(mb_mean_diff)
                batch_size.append([len(similar_ids_A),len(similar_ids_B)])

                if(saved_sample_counter < sample_max):
                    saved_sample_counter+=1
                    f,axs = plt.subplots(6,10,figsize=(16,9))
                    for a in axs.flat:
                        a.axis('off')

                    for a,sim_id in enumerate(random.sample(range(len(similar_ids_A)),min(len(similar_ids_A),axs.shape[1]))):

                        # print("a={}, sim id={}, all_sim_ids={}".format(a,sim_id,similar_ids_A))
                        axs[0,a].imshow(self.map_to_color(self.a_gt_labels[similar_ids_A[sim_id],:,:]),interpolation='nearest')
                        # axs[0,a].imshow(self.a_gt_labels[similar_ids_A[a],:,:])
                        axs[0,a].set_title("GT, Sim={:.2f}".format(similarities_A[sim_id]),fontsize=8,pad=-100)
                        axs[1,a].imshow(self.map_to_color(self.a_pred_labels[similar_ids_A[sim_id],:,:]),interpolation='nearest')
                        # axs[1,a].imshow(self.a_pred_labels[similar_ids_A[a],:,:])
                        axs[1,a].set_title("Prediction",fontsize=8,pad=-100)
                        axs[2,a].imshow(self.a_imgs[similar_ids_A[sim_id],:,:,:])
                        axs[2,a].set_title("Performance={:.2f}".format(self.a_perf[similar_ids_A[sim_id]]),fontsize=8,pad=-100)

                    # Overwrite top left with reference sample
                    axs[0,0].imshow(self.map_to_color(self.a_gt_labels[i,:,:]),interpolation='nearest')
                    axs[0,0].set_title("GT Reference",fontsize=8,pad=-100)
                    axs[1,0].imshow(self.map_to_color(self.a_pred_labels[i,:,:]),interpolation='nearest')
                    axs[1,0].set_title("Prediction",fontsize=8,pad=-100)
                    axs[2,0].imshow(self.a_imgs[i,:,:,:])
                    axs[2,0].set_title("Performance={:.2f}".format(self.a_perf[i]),fontsize=8,pad=-100)

                    # exit(1)
                    for a,sim_id in enumerate(random.sample(range(len(similar_ids_B)),min(len(similar_ids_B),axs.shape[1]))):
                        axs[3,a].imshow(self.map_to_color(self.b_gt_labels[similar_ids_B[sim_id],:,:]),interpolation='nearest')
                        # axs[3,a].imshow(self.b_gt_labels[similar_ids_B[a],:,:])
                        axs[3,a].set_title("GT, Sim={:.2f}".format(similarities_B[sim_id]),fontsize=8,pad=-100)
                        axs[4,a].imshow(self.map_to_color(self.b_pred_labels[similar_ids_B[sim_id],:,:]),interpolation='nearest')
                        # axs[4,a].imshow(self.b_pred_labels[similar_ids_B[a],:,:])
                        axs[4,a].set_title("Prediction",fontsize=8,pad=-100)
                        axs[5,a].imshow(self.b_imgs[similar_ids_B[sim_id],:,:,:])
                        axs[5,a].set_title("Performance={:.2f}".format(self.b_perf[similar_ids_B[sim_id]]),fontsize=8,pad=-100)

                    # t2 = time.time()
                    plt.suptitle("From {}, showing {}/{} similar examples. \n From {}, showing {}/{} similar examples examples \n \
                        W1={:.3f} | Mean diff={:.3f}".format(self.nameA,min(len(similar_ids_A),10),len(similar_ids_A),self.nameB, \
                        min(len(similar_ids_B),10),len(similar_ids_B),mb_w1,mb_mean_diff))
                    if(vis):
                        plt.show()
                    elif(save):
                        plt.savefig(os.path.join(output_dir,"overlap",'comparison_{}.png'.format(i)), dpi=200)
                    plt.close("all")

        weights = np.asarray(weights)
        batch_size = np.asarray(batch_size)
        patch_w1 = np.asarray(patch_w1)
        patch_mean = np.asarray(patch_mean)  
        # patch_sKVD = np.asarray(patch_sKVD)


        #save intermediate values in case we want to plot them at some point
        np.savetxt(os.path.join(output_dir,"batch_size.csv"),batch_size,delimiter=",")
        np.savetxt(os.path.join(output_dir,"patch_w1.csv"),patch_w1,delimiter=",")
        np.savetxt(os.path.join(output_dir,"patch_mean.csv"),patch_mean,delimiter=",")
        np.savetxt(os.path.join(output_dir,"weights.csv"),weights,delimiter=",")

        skvd = [-1,-1]
        if (args.skvd and len(aligned_patches) > 5):
            skvd = sKVD(aligned_patches,kvd_subset_size=min(1000,int(len(aligned_patches)/2)),kvd_subsets=1000)

        ids_not_used_A = np.where(used_A==0)[0]
        ids_not_used_B = np.where(used_B==0)[0]
        
        if(len(batch_size) < 1):
            self.logger.log("No overlap found between datasets")
            return

        remaining_performance_A = []
        for id in ids_not_used_A:
            if(id < sample_max):

                f,axs = plt.subplots(1,3,figsize=(6,2))
                for a in axs.flat:
                    a.axis('off')

                axs[0].imshow(self.map_to_color(self.a_gt_labels[id,:,:]),interpolation='nearest')
                axs[0].set_title("Seg Map",fontsize=8,pad=-100)
                axs[1].imshow(self.map_to_color(self.a_pred_labels[id,:,:]),interpolation='nearest')
                axs[1].set_title("Prediction",fontsize=8,pad=-100)
                axs[2].imshow(self.a_imgs[id,:,:,:])
                axs[2].set_title("Performance={:.2f}".format(self.a_perf[id]),fontsize=8,pad=-100)

                plt.suptitle("Dataset {}. Sample {} not in overlap".format(self.nameA,id))   

                if(vis):
                    plt.show()
                elif(save):
                    plt.savefig(os.path.join(output_dir,"nonoverlap",'unused_sample_A_{}.png'.format(id)), dpi=200)
                plt.close("all")

            remaining_performance_A.append(self.a_perf[id])
    
        remaining_performance_B = []
        for id in ids_not_used_B:
            if(id < sample_max):
                f,axs = plt.subplots(1,3,figsize=(6,2))
                for a in axs.flat:
                    a.axis('off')

                axs[0].imshow(self.map_to_color(self.b_gt_labels[id,:,:]),interpolation='nearest')
                axs[0].set_title("Seg Map",fontsize=8,pad=-100)
                axs[1].imshow(self.map_to_color(self.b_pred_labels[id,:,:]),interpolation='nearest')
                axs[1].set_title("Prediction",fontsize=8,pad=-100)
                axs[2].imshow(self.b_imgs[id,:,:,:])
                axs[2].set_title("Performance={:.2f}".format(self.b_perf[id]),fontsize=8,pad=-100)

                plt.suptitle("Dataset {}. Sample {} not in overlap".format(self.nameB,id))   

                if(vis):
                    plt.show()
                elif(save):
                    plt.savefig(os.path.join(output_dir,"nonoverlap",'unused_sample_B_{}.png'.format(id)), dpi=200)
                plt.close("all")   

            remaining_performance_B.append(self.b_perf[id])

        self.logger.log("=== Comparison Results ===")

        self.logger.log("\t Samples with overlap A= {}/{}".format(np.count_nonzero(used_A==1),len(self.a_perf)))
        self.logger.log("\t Samples with overlap B= {}/{}".format(np.count_nonzero(used_B==1),len(self.b_perf)))
        self.logger.log("\t Mean Batch size A= {:.2f}".format(np.mean(batch_size[:,0])))
        self.logger.log("\t Mean Batch size B= {:.2f}".format(np.mean(batch_size[:,1])))

        self.logger.log("\t Median Batch size A= {:.2f}".format(np.median(batch_size[:,0])))
        self.logger.log("\t Median Batch size B= {:.2f}".format(np.median(batch_size[:,1])))
        
        self.logger.log("\t Weighted Average W1={:.3f}".format(np.average(patch_w1,weights=1/weights)))
        self.logger.log("\t Mean W1= {:.3f}".format(np.mean(patch_w1)))
        self.logger.log("\t Median W1= {:.3f}".format(np.median(patch_w1)))
        self.logger.log("\t Stdev W1= {:.3f}".format(np.std(patch_w1)))
        self.logger.log("\t Max W1= {:.3f}".format(np.max(patch_w1)))

        self.logger.log("\t Mean Absolute Difference of Means= {:.3f}".format(np.mean(np.abs(patch_mean))))
        self.logger.log("\t Median Absolute Difference of Means= {:.3f}".format(np.median(np.abs(patch_mean))))
        self.logger.log("\t Stdev Absolute Difference of Means= {:.3f}".format(np.std(np.abs(patch_mean))))
        self.logger.log("\t Max Absolute Difference of Means= {:.3f}".format(np.max(np.abs(patch_mean))))

        if(args.skvd):
            self.logger.log("\t Mean sKVD= {:.6f}".format(skvd[0]))       
            self.logger.log("\t Std sKVD= {:.6f}".format(skvd[1]))       

        self.logger.log("\t Non-overlapping region mean performance A:= {:.3f}".format(np.mean(remaining_performance_A)))
        self.logger.log("\t Non-overlapping region stdev performance A:= {:.3f}".format(np.std(remaining_performance_A)))
        self.logger.log("\t Non-overlapping region mean performance B:= {:.3f}".format(np.mean(remaining_performance_B)))
        self.logger.log("\t Non-overlapping region stdev performance B:= {:.3f}".format(np.std(remaining_performance_B)))

        self.logger.log("\t Diff of performance means = {:.3f}".format(np.mean(self.a_perf) - np.mean(self.b_perf)))

        self.logger.close()
