import numpy as np
import matplotlib.pyplot as plt
import json
import copy
import operator

from task2_tools import read_predicted_boxes, read_ground_truth_boxes

def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """

    prediction_box_area = (prediction_box[2]-prediction_box[0])*(prediction_box[3]-prediction_box[1])
    gt_box_area = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
    #no overlap

    # determine the (x, y)-coordinates of the intersection rectangle
    xmin = max(prediction_box[0], gt_box[0])
    ymin = max(prediction_box[1], gt_box[1])
    xmax = min(prediction_box[2], gt_box[2])
    ymax = min(prediction_box[3], gt_box[3])

    # compute the area of intersection rectangle
    if xmin > xmax or ymin > ymax:
        intersection = 0
    else:
        intersection = (xmax-xmin)*(ymax-ymin)

    union = (prediction_box_area+gt_box_area)-intersection

    return intersection/union


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    return num_tp/(num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    return num_tp/(num_tp + num_fn)


def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    matches = []
    for i in range(len(gt_boxes)):
        for j in range(len(prediction_boxes)):
            if calculate_iou(prediction_boxes[j],gt_boxes[i]) >= iou_threshold:
                matches.append((calculate_iou(prediction_boxes[j],gt_boxes[i]),j,i))

    matches.sort(key = operator.itemgetter(1), reverse = True)
    #matches is now a list of ALL matches sorted by IoU
    prediction_matches = []
    gt_matches = []

    while True:
        #check if best match is accepted
        if len(matches) == 0:
            break
        prediction_match_index = matches[0][1]
        gt_match_index = matches[0][2]
        prediction_matches.append(prediction_boxes[prediction_match_index])
        gt_matches.append(gt_boxes[gt_match_index])
        #remove all matches from same prediction box or to same ground truth box
        updated_matches = []
        for i in range(len(matches)):
            if matches[i][1] != prediction_match_index and matches[i][2] != gt_match_index:
                updated_matches.append(matches[i])
        matches = updated_matches
    prediction_matches = np.asarray(prediction_matches)
    gt_matches = np.asarray(gt_matches)

    return (prediction_matches, gt_matches)


def calculate_individual_image_result(
        prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """
    (prediction_matches, gt_matches) = get_all_box_matches(
            prediction_boxes, gt_boxes, iou_threshold)
    if len(prediction_matches)-len(gt_matches) != 0:
        print(len(prediction_matches),len(gt_matches))
    true_positives = len(gt_matches)
    false_negatives = len(gt_boxes)-len(gt_matches)
    false_positives = len(prediction_boxes)-len(prediction_matches)

    return {"true_pos": true_positives, "false_pos": false_positives, "false_neg": false_negatives}


def calculate_precision_recall_all_images(
        all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    num_images = len(all_gt_boxes)

    for i in range(num_images):
        values = calculate_individual_image_result(all_prediction_boxes[i], all_gt_boxes[i], iou_threshold)
        false_positives += values["false_pos"]
        false_negatives += values["false_neg"]
        true_positives += values["true_pos"]
    #print("FP: ", false_positives)
    #print("FN: ", false_negatives)
    #print("TP: ", true_positives)
    precision = calculate_precision(true_positives, false_positives, false_negatives)
    recall = calculate_recall(true_positives, false_positives, false_negatives)

    #print("Precision: ", precision)
    #print("Recall: ", recall)
    return (precision, recall)


def get_precision_recall_curve(all_prediction_boxes, all_gt_boxes,
                               confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the precision-recall curve over all images. Use the given
       confidence thresholds to find the precision-recall curve.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, xmax, ymin, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        tuple: (precision, recall). Both np.array of floats.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    confidence_thresholds = np.linspace(0, 1, 500)
    precisions = []
    recalls = []
    for i in range(len(confidence_thresholds)):
        confident_predictions = []
        for j in range(len(all_prediction_boxes)):
            filter_mask = np.greater_equal(confidence_scores[j], confidence_thresholds[i])
            confident_predictions.append(all_prediction_boxes[j][filter_mask])
        (this_precision, this_recall) = calculate_precision_recall_all_images(
                confident_predictions, all_gt_boxes, iou_threshold)
        #print(len(confident_predictions[0]))
        precisions.append(this_precision)
        recalls.append(this_recall)
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    return (precisions, recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    # No need to edit this code.
    plt.figure(figsize=(20, 20))
    plt.plot(recalls,precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")



def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    # DO NOT CHANGE. If you change this, the tests will not pass when we run the final
    # evaluation
    recall_levels = np.linspace(0, 1.0, 11)

    recalls, precisions = zip(*sorted(zip(recalls, precisions)))
    #calculate average precision for each class
    sum = 0
    for i in range(len(recall_levels)):
        min_precision_index = -1
        for j in range(len(recalls)):
            if recalls[j] >= recall_levels[i]:
                min_precision_index = j
                break
        if min_precision_index == -1:
            sum += 0
            print("Precision for recall >= ",recall_levels[i], " : ", 0)
        else:
            sum += max(precisions[min_precision_index:])
            print("Precision for recall >= ",recall_levels[i], " : ", max(precisions[min_precision_index:]))

    return sum/len(recall_levels)





def mean_average_precision(ground_truth_boxes, prediction_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        prediction_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = prediction_boxes[image_id]["boxes"]
        scores = prediction_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)
    iou_threshold = 0.5
    precisions, recalls = get_precision_recall_curve(all_prediction_boxes,
                                                     all_gt_boxes,
                                                     confidence_scores,
                                                     iou_threshold)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions,
                                                              recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    prediction_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, prediction_boxes)
