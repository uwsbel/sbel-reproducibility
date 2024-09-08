import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def AssociateObjectsAndGetMetrics(pred,gt,img,display=False,verbose=False):
    c = img.shape[0]
    h = img.shape[1]
    w = img.shape[2]

    # perform box matching for each class
    gt_boxes = gt["boxes"].cpu().detach().numpy()
    gt_labels = gt["labels"].cpu().detach().numpy()
    pred_boxes = pred["boxes"].cpu().detach().numpy()
    pred_labels = pred["labels"].cpu().detach().numpy()

    # do matching on all classes together to include mislabeling in error metrics
    gt_boxes = gt_boxes[gt_labels > 0]
    gt_labels = gt_labels[gt_labels > 0]
    pred_boxes = pred_boxes[pred_labels > 0]
    pred_labels = pred_labels[pred_labels > 0]

    if(verbose and display):
        #display image with boxes
        f,ax = plt.subplots()
        ax.imshow(img.transpose((1,2,0)))
        for g,gt in enumerate(gt_boxes):
            color = 'g' if gt_labels[g]==2 else 'r'
            rect1 = patches.Rectangle((gt[0],gt[1]), gt[2]-gt[0], gt[3]-gt[1], linewidth=1, edgecolor=color,linestyle='dashed', facecolor='none')
            ax.add_patch(rect1)
        for p,pr in enumerate(pred_boxes):
            color = 'g' if pred_labels[p]==2 else 'r'
            rect2 = patches.Rectangle((pr[0],pr[1]), pr[2]-pr[0], pr[3]-pr[1], linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect2)
        plt.show()

    iou_matrix = np.zeros((len(pred_labels), len(gt_labels)))

    # calculate iou for predicted box vs each gt box
    for p in range(len(pred_labels)):
        for g in range(len(gt_labels)):
            gx0, gy0, gx1, gy1 = gt_boxes[g, :].astype(np.int32)
            px0, py0, px1, py1 = pred_boxes[p, :].astype(np.int32)

            # do mask comparison
            mask = np.zeros((w, h)).astype(np.int32)
            mask[gx0:gx1, gy0:gy1] += 1
            mask[px0:px1, py0:py1] += 1

            intersection = np.count_nonzero(mask[mask == 2])
            union = np.count_nonzero(mask[mask > 0])

            iou_matrix[p, g] = intersection / float(union)

    # greedily find the max IOU to match boxes
    pairs = []

    gt_ids = list(range(len(gt_labels)))
    pred_ids = list(range(len(pred_labels)))

    if(len(pred_labels) > 0 and len(gt_labels) > 0):
        for p in range(len(pred_labels)):
            if(np.max(iou_matrix) < 1e-9):
                break

            id_max = np.unravel_index(
                iou_matrix.argmax(), iou_matrix.shape)
            pairs.append(id_max)

            pred_ids.remove(id_max[0])
            gt_ids.remove(id_max[1])

            # zero out the row and column selected
            iou_matrix[id_max[0], :] = -1
            iou_matrix[:, id_max[1]] = -1

    # handle any remaining predictions or gt boxes with no pairs
    for p in pred_ids:
        pairs.append((p, -1))
    for g in gt_ids:
        pairs.append((-1, g))

    per_cone_data = []
    for p, pair in enumerate(pairs):
        p_id = pair[0]
        g_id = pair[1]

        # === calculate errors and metrics
        # default metric values
        gt_class = 0
        gt_pos_x = -1
        gt_pos_y = -1
        gt_size_x = -1
        gt_size_y = -1
        pr_class = 0
        pr_pos_x = -1
        pr_pos_y = -1
        pr_size_x = -1
        pr_size_y = -1
        object_iou = 0

        #predicted exclusive information
        if(p_id > -1):
            pr_class = pred_labels[p_id].astype(np.int32)
            px0, py0, px1, py1 = pred_boxes[p_id, :].astype(np.int32)
            pr_pos_x = (px1 + px0 ) / 2.0
            pr_pos_y = (py1 + py0 ) / 2.0
            pr_size_x = px1 - px0
            pr_size_y = py1 - py0


        #ground truth exlusive information
        if(g_id > -1):
            gt_class = gt_labels[g_id].astype(np.int32)
            gx0, gy0, gx1, gy1 = gt_boxes[g_id, :].astype(np.int32)
            gt_pos_x = (gx1 + gx0 ) / 2.0
            gt_pos_y = (gy1 + gy0 ) / 2.0
            gt_size_x = gx1 - gx0
            gt_size_y = gy1 - gy0

        if(pr_class > 0 and gt_class > 0 and pr_class == gt_class):
            px0, py0, px1, py1 = pred_boxes[p_id, :].astype(np.int32)
            gx0, gy0, gx1, gy1 = gt_boxes[g_id, :].astype(np.int32)

            mask = np.zeros((w, h)).astype(np.int32)
            mask[gx0:gx1, gy0:gy1] += 1
            mask[px0:px1, py0:py1] += 1

            intersection = np.count_nonzero(mask[mask == 2])
            union = np.count_nonzero(mask[mask > 0])
            object_iou = intersection / union

        per_cone_data.append([gt_class, gt_pos_x, gt_pos_y, gt_size_x, gt_size_y,
                                pr_class, pr_pos_x, pr_pos_y, pr_size_x, pr_size_y, object_iou])

    per_cone_data = np.asarray(per_cone_data)
    if(len(per_cone_data.shape)==1):
        per_cone_data = per_cone_data.reshape(1,per_cone_data.shape[0])
        
    return per_cone_data