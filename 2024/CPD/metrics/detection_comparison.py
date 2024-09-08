
import re
from common.loader import ObjectDetectionImgLoaderWithPreds
from common.logger import Logger
from metrics.measures import AssociateObjectsAndGetMetrics

import torch

import os, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import random

class DetectionComparitor():

    def __init__(self, output_path="output",write=False,verbose=False):
        self.verbose = False
        self.logger = Logger(filename=os.path.join(output_path,"results.txt"),write=write,display=True)

    def SetDatasets(self, dirA: os.PathLike, dirB: os.PathLike, predsA: os.PathLike, predsB: os.PathLike, nameA: str,
                    nameB: str, threads=1, max_samples=-1, boxes=100, shuffle_a=False, shuffle_b=False):
        """ Adds datasets to the comparitor. The comparitor will the an A vs B comparison """

        self.nameA = nameA
        self.nameB = nameB

        datasetA = ObjectDetectionImgLoaderWithPreds(
            data_root=dirA,
            pred_dir=predsA,
            max_boxes=boxes,
            max_samples=max_samples,
            apply_transforms=False,
            shuffle=shuffle_a)

        self.loaderA = torch.utils.data.DataLoader(
            datasetA, 1, shuffle=False, num_workers=threads, drop_last=True)

        datasetB = ObjectDetectionImgLoaderWithPreds(
            data_root=dirB,
            max_boxes=boxes,
            pred_dir=predsB,
            max_samples=max_samples,
            apply_transforms=False,
            shuffle=shuffle_b)

        self.loaderB = torch.utils.data.DataLoader(
            datasetB, 1, shuffle=False, num_workers=threads, drop_last=True)

    def GetPatchFeatures(self,per_object_data,patch_size,img,num_classes=2):
        #generate segmentation layes
        buffer = (np.asarray(patch_size)/2).astype(np.int32)
        gt_map = np.zeros((img.shape[1]+patch_size[1],img.shape[2]+patch_size[0],num_classes))

        padded_img = np.pad(img.transpose(1,2,0), ((buffer[1], buffer[1]), (buffer[0], buffer[0]), (0, 0)), mode='constant', constant_values=0)

        #add all GT objects
        #gt_class, gt_pos_x, gt_pos_y, gt_size_x, gt_size_y, pr_class, pr_pos_x, pr_pos_y, pr_size_x, pr_size_y, object_iou
        for data in per_object_data:
            gt,x,y,sx,sy = data[0:5]

            if(patch_size[0] < sx or patch_size[1] < sy):
                print("WARNING: object size {}x{} was larger than patch size {}x{}. Must choose a patch size larger than the object for it to be added.".format(int(sx),int(sy),patch_size[0],patch_size[1]))

            x0 = int(np.round(x - sx/2)) + buffer[0]
            x1 = int(np.round(x + sx/2)) + buffer[0]
            y0 = int(np.round(y - sy/2)) + buffer[1]
            y1 = int(np.round(y + sy/2)) + buffer[1]
            gt_map[y0:y1,x0:x1,int(gt)-1] = 1

        features = []
        patches = []
        pred_boxes = []
        gt_boxes = []
        for data in per_object_data:
            gt,x,y,sx,sy = data[0:5].astype(np.int32)
            feature_map = gt_map[y:y+patch_size[1],x:x+patch_size[0],:].flatten()
            img_patch = padded_img[y:y+patch_size[1],x:x+patch_size[0],:]

            features.append(feature_map)
            patches.append(img_patch)

            pr,px,py,psx,psy = data[5:10].astype(np.int32)
            px = px - x + buffer[0]
            py = py - y+ buffer[1]

            x = buffer[0]
            y = buffer[1]
            pred_boxes.append(np.array([pr,px,py,psx,psy]))
            gt_boxes.append(np.array([gt,x,y,sx,sy]))

        return np.asarray(features),np.asarray(patches),np.asarray(pred_boxes),np.asarray(gt_boxes)
    
    
    def GetPerfAndFeatures(self,loader,patch_size,mask_factor,verbose):
        perf = np.array([]) #list of [class id, IOU]
        features = np.array([]) #list of [features]
        imgPatches = np.array([]) #list of image patches
        box_list = np.array([])

        n_imgs = len(loader)
        it = iter(loader)

        for i in tqdm(range(n_imgs),"extracting patches"):
            imgs, gt_boxes, gt_labels, pred_boxes, pred_labels = next(it)
            
            if(len(gt_labels)==0):
                continue

            target_list = {}
            target_list["boxes"] = gt_boxes
            target_list["labels"] = gt_labels

            predictions = {}
            predictions["boxes"] = pred_boxes
            predictions["labels"] = pred_labels
            img = imgs[0,:,:,:].numpy()

            per_object_data = AssociateObjectsAndGetMetrics(predictions,target_list,img,display=False,verbose=False)

            #TODO: we currently only look at GT objects. Should we also be able to consider false positive predictions?
            if(per_object_data.size>0 and len(per_object_data[:,0]>0)>0):
                per_object_data = per_object_data[per_object_data[:,0]>0,:]

                if(perf.size == 0):
                    perf = per_object_data[:,[0,-1]]
                else:
                    perf = np.concatenate((perf,per_object_data[:,[0,-1]]),axis=0)


                feature,patch,pred_box,gt_box = self.GetPatchFeatures(per_object_data,patch_size,img)

                #mask the features and image based on object size and mask size
                for p,pa in enumerate(patch):
                    gt,x,y,sx,sy = gt_box[p]
                    sx *= mask_factor[0]
                    sy *= mask_factor[1]

                    y_min = max(0,int(y-sy/2))
                    y_max = min(patch_size[1]-1,int(y+sy/2))
                    x_min = max(0,int(x-sx/2))
                    x_max = min(patch_size[0]-1,int(x+sx/2))

                    mask_box = np.zeros((patch_size[1],patch_size[0],1))
                    mask_box[y_min:y_max+1,x_min:x_max+1] = 1

                    # patch[p] = patch[p]*mask_box

                    feature_map = feature[p].reshape(patch_size[1],patch_size[0],-1)
                    feature_map = feature_map*mask_box
                    feature[p] = feature_map.flatten()

                if(features.size == 0):
                    features = feature
                    imgPatches = patch
                    box_list = pred_box
                else:
                    features = np.concatenate((features,feature),axis=0)
                    imgPatches = np.concatenate((imgPatches,patch),axis=0)

                    box_list = np.concatenate((box_list,pred_box),axis=0)
        #correct for extra large datatypes
        perf = perf.astype(np.float32)
        features = features.astype(np.uint8)
        imgPatches = (imgPatches*255.0).astype(np.uint8)

        return perf,features,imgPatches,box_list

    def PatchComparison(self, args): 
        #patch_size=[64,64],mask_factor=[1.5,1.5],patch_threshold=.5,save=False,display=False,verbose=False,output_dir="output",min_batch_size=1) ->None:
        ##patch_size=args.patch_size,mask_factor=args.mask_factor,patch_threshold=args.patch_threshold,save=args.save,display=args.display,verbose=args.verbose,output_dir=args.output_path,min_batch_size=args.min_batch_size)
        
        '''
            patch_size: the size of the patch to be extracted around the object as features
            patch_threshold: the similarity between two patches to be considered as part of a mini batch
        '''
        t00 = time.time()

        #get params from args
        patch_size=args.patch_size
        mask_factor=args.mask_factor
        patch_threshold=args.patch_threshold
        save=args.save
        display=args.display
        verbose=args.verbose
        output_dir=args.output_path
        min_batch_size=args.min_batch_size 


        #only do this here as it downloads vgg net and loads onto GPU
        if(args.skvd):
            from metrics.ml_metrics import sKVD

        if(len(patch_size) != 2):
            self.logger.log("Error: Unrecognized patch size:",patch_size)
            exit(1)

        if(save and not os.path.exists(output_dir)):
            os.mkdir(output_dir)

        if(save and not os.path.exists(os.path.join(output_dir,"overlap"))):
            os.mkdir(os.path.join(output_dir,"overlap"))

        if(save and not os.path.exists(os.path.join(output_dir,"nonoverlap"))):
            os.mkdir(os.path.join(output_dir,"nonoverlap"))
        
    
        if(verbose):
            self.logger.log("Patch Comparison Setting")
            self.logger.log("\tPatch size: {}".format(patch_size))
            self.logger.log("\tMask factor: {}".format(mask_factor))
            self.logger.log("\tPatch threshold: {}".format(patch_threshold))
            self.logger.log("Generating intermediate results for dataset A")
        
        # generate data for set A
        perfA,featuresA,patchesA,boxesA = self.GetPerfAndFeatures(self.loaderA,patch_size,mask_factor,verbose)
        del self.loaderA
        # print("Performance Data A:",perfA.shape)
        # print("Feature Data A:",featuresA.shape)

        if(verbose):
            self.logger.log("Generating intermediate results for dataset B")

        perfB,featuresB,patchesB,boxesB = self.GetPerfAndFeatures(self.loaderB,patch_size,mask_factor,verbose)
        del self.loaderB
        # del self.model
        torch.cuda.empty_cache()
        # print("Performance Data B:",perfB.shape)
        # print("Feature Data B:",featuresB.shape)

        if(verbose):
            self.logger.log("Performing comparison using set A")
            
        for c in np.unique(perfA[:,0]):
            class_perf_a = perfA[perfA[:,0] == c]
            class_perf_b = perfB[perfB[:,0] == c]
            self.logger.log("Mean class performance. A = {:.3f} | B = {:.3f} ".format(np.mean(class_perf_a[:,1]),np.mean(class_perf_b[:,1])))

        tested_class = []
        patch_w1 = []
        patch_mean = []
        batch_size = []
        aligned_patches = []
        weights = []
        mse = []

        used_A = np.zeros(len(perfA))
        used_B = np.zeros(len(perfB))

        t01 = time.time()

        self.logger.log("Prep time: {:.2f}".format(t01-t00))

        self.logger.log("Number of A samples: {}".format(len(perfA)))

        count = 0
        for i,example in enumerate(tqdm(perfA)):
            class_id = perfA[i,0].astype(int)

            #find all similar in set A
            feature = featuresA[i,:]
            similarities_intersection_A = featuresA * feature
            similarities_union_A = featuresA + feature

            #use IOU between the two gt maps
            similarities_A = np.sum(similarities_intersection_A>0,axis=1) / np.sum(similarities_union_A>0,axis=1)
            similar_ids_A = np.where(similarities_A>patch_threshold)[0]

            #find all similar in set B
            similarities_intersection_B = featuresB * feature
            similarities_union_B = featuresB + feature

            #use IOU between the two gt maps
            similarities_B = np.sum(similarities_intersection_B>0,axis=1) / np.sum(similarities_union_B>0,axis=1)
            similar_ids_B = np.where(similarities_B>patch_threshold)[0]

            if(len(similar_ids_A)>=min_batch_size and len(similar_ids_B)>=min_batch_size):

                #tag the samples as being overlapping
                used_A[similar_ids_A] = 1
                used_B[similar_ids_B] = 1

                weights.append(len(similar_ids_A))

                if(verbose):
                    self.logger.log("Object instance {}/{} had {} and {} similar instances".format(i+1,len(perfA), len(similar_ids_A),len(similar_ids_B)))

                #compare W1
                mb_perf_A = perfA[similar_ids_A,1]
                mb_perf_B = perfB[similar_ids_B,1]
                mb_w1 = wasserstein_distance(mb_perf_A,mb_perf_B)

                most_similar_B = similar_ids_B[np.argmax(similarities_B[similar_ids_B])]

                img_a0 = patchesA[i,:,:,:]
                img_b0 = patchesB[most_similar_B,:,:,:]
                img_a0 = img_a0.transpose(2,0,1)
                img_b0 = img_b0.transpose(2,0,1)

                if(args.skvd):
                    aligned_patches.append([img_a0,img_b0])

                #compare Mean
                mb_mean_diff = np.mean(mb_perf_A) - np.mean(mb_perf_B)

                patch_w1.append(mb_w1)
                patch_mean.append(mb_mean_diff)
                batch_size.append([len(similar_ids_A),len(similar_ids_B)])
                tested_class.append(class_id)
                mse.append(np.mean(np.square((img_a0 - img_b0) / 255.0)))

                #show some examples
                if(display or (save and count<50) ):
                    count +=1
                    f,axs = plt.subplots(6,10,figsize=(16,9))
                    for a in axs.flat:
                        a.axis('off')

                    # for a in range(axs.shape[1]):
                    for a,sim_id in enumerate(random.sample(range(len(similar_ids_A)),min(len(similar_ids_A),axs.shape[1]))):
                        # if(a >= len(similar_ids_A)):
                        #     break
                        feature_map = featuresA[similar_ids_A[sim_id],:].reshape(patch_size[1],patch_size[0],-1)
                        # print(feature_map.shape)
                        axs[0,a].imshow(feature_map[:,:,0])
                        axs[0,a].set_title("Class 1",fontsize=8,pad=-100)
                        axs[1,a].imshow(feature_map[:,:,1])
                        axs[1,a].set_title("Class 2",fontsize=8,pad=-100)
                        axs[2,a].imshow(patchesA[similar_ids_A[sim_id],:,:,:])
                        axs[2,a].set_title("IOU={:.2f}".format(perfA[similar_ids_A[sim_id],1]),fontsize=8,pad=-100)         

                        pred_box = boxesA[similar_ids_A[sim_id],:]
                        color = 'g' if pred_box[0]==2 else 'r'
                        rect = patches.Rectangle((pred_box[1]-pred_box[3]/2,pred_box[2]-pred_box[4]/2), pred_box[3], pred_box[4], linewidth=1, edgecolor=color, facecolor='none')
                        axs[2,a].add_patch(rect)
                    # Overwrite top left with reference sample
                    feature_map = featuresA[i,:].reshape(patch_size[1],patch_size[0],-1)
                    axs[0,0].imshow(feature_map[:,:,0])
                    axs[0,0].set_title("GT Reference",fontsize=8,pad=-100)
                    axs[1,0].imshow(feature_map[:,:,1])
                    axs[1,0].set_title("GT Reference",fontsize=8,pad=-100)
                    axs[2,a].imshow(patchesA[i,:,:,:])
                    axs[2,0].set_title("IOU={:.2f}".format(perfA[i,1]),fontsize=8,pad=-100)  


                    # for a in range(axs.shape[1]):
                    for a,sim_id in enumerate(random.sample(range(len(similar_ids_B)),min(len(similar_ids_B),axs.shape[1]))):
                        # if(a >= len(similar_ids_B)):
                        #     break
                        feature_map = featuresB[similar_ids_B[sim_id],:].reshape(patch_size[1],patch_size[0],-1)
                        axs[3,a].imshow(feature_map[:,:,0])
                        axs[3,a].set_title("Class 1",fontsize=8,pad=-100)
                        axs[4,a].imshow(feature_map[:,:,1])
                        axs[4,a].set_title("Class 2",fontsize=8,pad=-100)
                        axs[5,a].imshow(patchesB[similar_ids_B[sim_id],:,:,:])
                        axs[5,a].set_title("IOU={:.2f}".format(perfB[similar_ids_B[sim_id],1]),fontsize=8,pad=-100)

                        pred_box = boxesB[similar_ids_B[sim_id],:]
                        color = 'g' if pred_box[0]==2 else 'r'
                        rect = patches.Rectangle((pred_box[1]-pred_box[3]/2,pred_box[2]-pred_box[4]/2), pred_box[3], pred_box[4], linewidth=1, edgecolor=color, facecolor='none')
                        axs[5,a].add_patch(rect)

                    plt.suptitle("Showing similar examples from A ({} examples) and B ({} examples) \n W1={:.3f} | Mean diff={:.3f}".format(len(similar_ids_A),len(similar_ids_B),mb_w1,mb_mean_diff))

                    if(display):
                        plt.show()
                    elif(save):
                        plt.savefig(os.path.join(output_dir,"overlap",'comparison_{}.png'.format(i)), dpi=200)
                        plt.close("all")
                    else:
                        plt.close("all")

            elif(verbose):
                self.logger.log("Object instance {}/{} didn't have enough similar instances in both datasets".format(i+1,len(perfA)))


        if(verbose):
            self.logger.log("Generating output and samples")

        batch_size = np.asarray(batch_size)
        tested_class = np.asarray(tested_class)
        patch_w1 = np.asarray(patch_w1)
        patch_mean = np.asarray(patch_mean)    
        weights = np.asarray(weights)
        mse = np.asarray(mse)

        #save intermediate values in case we want to plot them at some point
        np.savetxt(os.path.join(output_dir,"batch_size.csv"),batch_size,delimiter=",")
        np.savetxt(os.path.join(output_dir,"tested_class.csv"),tested_class,delimiter=",")
        np.savetxt(os.path.join(output_dir,"patch_w1.csv"),patch_w1,delimiter=",")
        np.savetxt(os.path.join(output_dir,"patch_mean.csv"),patch_mean,delimiter=",")
        np.savetxt(os.path.join(output_dir,"weights.csv"),weights,delimiter=",")
        np.savetxt(os.path.join(output_dir,"mse.csv"),mse,delimiter=",")

        #look at untested regions in each dataset
        ids_not_used_A = np.where(used_A==0)[0]
        ids_not_used_B = np.where(used_B==0)[0]

        if(args.skvd):
            skvd = sKVD(aligned_patches,kvd_subset_size=min(1000,int(len(aligned_patches)/2)),kvd_subsets=1000)


        #save the patches that were not overlapping from A and B
        

        count = 0
        for id in ids_not_used_A:
            if((display or save) and count <50):
                count+=1
                f,axs = plt.subplots(1,3,figsize=(6,2))
                for a in axs.flat:
                    a.axis('off')

                feature_map = featuresA[id,:].reshape(patch_size[1],patch_size[0],-1)
                axs[0].imshow(feature_map[:,:,0])
                axs[0].set_title("Class 1",fontsize=8,pad=-100)
                axs[1].imshow(feature_map[:,:,1])
                axs[1].set_title("Class 2",fontsize=8,pad=-100)
                axs[2].imshow(patchesA[id,:,:,:])
                axs[2].set_title("IOU={:.2f}".format(perfA[id,1]),fontsize=8,pad=-100)
                
                pred_box = boxesA[id,:]
                color = 'g' if pred_box[0]==2 else 'r'
                rect = patches.Rectangle((pred_box[1]-pred_box[3]/2,pred_box[2]-pred_box[4]/2), pred_box[3], pred_box[4], linewidth=1, edgecolor=color, facecolor='none')
                axs[2].add_patch(rect)

                plt.suptitle("Dataset A. Sample {} not in overlap".format(id))   

                if(display):
                    plt.show()
                elif(save):
                    plt.savefig(os.path.join(output_dir,"nonoverlap",'unused_sample_A_{}.png'.format(id)), dpi=200)
                    plt.close("all")
                else:
                    plt.close("all")           
            count = 0
        for id in ids_not_used_B:

            # remaining_performance_B.append(self.b_perf[id])
            if((display or save) and count <50):
                count+=1
                f,axs = plt.subplots(1,3,figsize=(6,2))
                for a in axs.flat:
                    a.axis('off')

                feature_map = featuresB[id,:].reshape(patch_size[1],patch_size[0],-1)
                # print(feature_map.shape)
                axs[0].imshow(feature_map[:,:,0])
                axs[0].set_title("Class 1",fontsize=8,pad=-100)
                axs[1].imshow(feature_map[:,:,1])
                axs[1].set_title("Class 2",fontsize=8,pad=-100)
                axs[2].imshow(patchesB[id,:,:,:])
                axs[2].set_title("IOU={:.2f}".format(perfB[id,1]),fontsize=8,pad=-100) 
                pred_box = boxesB[id,:]
                color = 'g' if pred_box[0]==2 else 'r'
                rect = patches.Rectangle((pred_box[1]-pred_box[3]/2,pred_box[2]-pred_box[4]/2), pred_box[3], pred_box[4], linewidth=1, edgecolor=color, facecolor='none')
                axs[2].add_patch(rect)
        
                plt.suptitle("Dataset B. Sample {} not in overlap".format(id))     

                if(display):
                    plt.show()
                elif(save):
                    plt.savefig(os.path.join(output_dir,"nonoverlap",'unused_sample_B_{}.png'.format(id)), dpi=200)
                    plt.close("all")
                else:
                    plt.close("all")         

        self.logger.log("=== Comparison Results ===")

        if(args.skvd):
            self.logger.log("\t Mean sKVD= {:.3f}".format(skvd[0]))
            self.logger.log("\t Std sKVD= {:.3f}".format(skvd[1]))


        for c in np.unique(tested_class):
            patch_w1_class = patch_w1[tested_class==c]
            patch_weights_class = weights[tested_class==c]
            batch_size_class = batch_size[tested_class==c,:]
            patch_mean_class = patch_mean[tested_class==c]
            perfA_class = perfA[perfA[:,0]==c]
            perfB_class = perfB[perfB[:,0]==c]
            b_used_class = perfB[used_B==1,0]==c


            a_slice_not_used = perfA[ids_not_used_A,:]
            remaining_perf_a = a_slice_not_used[a_slice_not_used[:,0]==c,1]
            b_slice_not_used = perfB[ids_not_used_B,:]
            remaining_perf_b = b_slice_not_used[b_slice_not_used[:,0]==c,1]

            self.logger.log("\t Test Class: {}".format(c))  
            self.logger.log("\t\t Samples with overlap A= {}/{}".format(len(patch_w1_class),len(perfA_class)))
            self.logger.log("\t\t Samples with overlap B= {}/{}".format(np.count_nonzero(b_used_class),len(perfB_class)))
            self.logger.log("\t\t Mean Batch size A:B= {:.2f}:{:.2f}".format(np.mean(batch_size_class[:,0]),np.mean(batch_size_class[:,1])))
            self.logger.log("\t\t Median Batch size A:B= {:.2f}:{:.2f}".format(np.median(batch_size_class[:,0]),np.median(batch_size_class[:,1])))
            self.logger.log("\t\t Weighted Avg W1= {:.3f}".format(np.average(patch_w1_class,weights=patch_weights_class)))
            self.logger.log("\t\t Mean W1= {:.3f}".format(np.mean(patch_w1_class)))
            self.logger.log("\t\t Median W1= {:.3f}".format(np.median(patch_w1_class)))
            self.logger.log("\t\t Max W1= {:.3f}".format(np.max(patch_w1_class)))
            self.logger.log("\t\t Mean Absolute Difference of Means= {:.3f}".format(np.mean(np.abs(patch_mean_class))))
            self.logger.log("\t\t Median Absolute Difference of Means= {:.3f}".format(np.median(np.abs(patch_mean_class))))
            self.logger.log("\t\t Max Absolute Difference of Means= {:.3f}".format(np.max(np.abs(patch_mean_class))))

            self.logger.log("\t\t Non-overlapping region mean performance A:= {:.3f}".format(np.mean(remaining_perf_a)))
            self.logger.log("\t\t Non-overlapping region stdev performance A:= {:.3f}".format(np.std(remaining_perf_a)))
            self.logger.log("\t\t Non-overlapping region mean performance B:= {:.3f}".format(np.mean(remaining_perf_b)))
            self.logger.log("\t\t Non-overlapping region stdev performance B:= {:.3f}".format(np.std(remaining_perf_b)))

            class_perf_a = perfA[perfA[:,0] == c]
            class_perf_b = perfB[perfB[:,0] == c]
            self.logger.log("\t\t Diff of mean class performance = {:.3f}".format(np.mean(class_perf_a[:,1]) - np.mean(class_perf_b[:,1])))


            if(display or save):

                # f,axs = plt.subplots(1,3)
                f,axs = plt.subplots(1,3,figsize=(9,4))
                axs[0].hist(patch_w1)
                axs[0].set_title("W1 Distribution")
                axs[0].set_ylabel("Count")
                axs[0].set_xlabel("W1")

                axs[1].hist(patch_mean)
                axs[1].set_title("Mean Difference Distribution")
                axs[1].set_ylabel("Count")
                axs[1].set_xlabel("Difference in Means")

                axs[2].hist(batch_size)
                axs[2].set_title("Batch Size Distribution")
                axs[2].set_ylabel("Count")
                axs[2].set_xlabel("Batch Size")
                axs[2].legend(["A","B"])

                plt.suptitle("Performance Metric Distribution. Class {}".format(int(c)))

                if(save):
                    plt.savefig(os.path.join(output_dir,'metric_distribution_class_{}.png'.format(int(c))), dpi=200)

                if(display):
                    plt.show()
                elif(save):
                    plt.close("all")
                else:
                    plt.close("all")   
        self.logger.close()