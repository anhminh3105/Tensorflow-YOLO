import numpy as np
import math
from images import scale_bbox

def to_vertices(centers, sizes):
    return centers - sizes/2, centers + sizes/2

def confidence_filter(predictions, confidence_threshold=None):
    if confidence_threshold == None:
        confidence_threshold=.6
    centers, sizes, confidence, class_scores = predictions
    mask = np.squeeze(confidence > confidence_threshold)
    centers = centers[mask]
    sizes = sizes[mask]
    confidence = confidence[mask]
    class_scores = class_scores[mask]
    classes = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)
    return centers, sizes, confidence, classes, scores

def iou(box1, box2):
    top1, bot1 = box1[0], box1[1]
    top2, bot2 = box2[0], box2[1]
    x1 = np.maximum(top1[0], top2[:, 0])
    y1 = np.maximum(top1[1], top2[:, 1])
    x2 = np.minimum(bot1[0], bot2[:, 0])
    y2 = np.minimum(bot1[1], bot2[:, 1])
    i_areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    b1_area = np.maximum(0, bot1[0] - top1[0]) * np.maximum(0, bot1[1] - top1[1])
    b2_area = np.maximum(0, bot2[:, 0] - top2[:, 0]) * np.maximum(0, bot2[:, 1] - top2[:, 1])
    u_areas = b1_area + b2_area - i_areas
    return i_areas / u_areas  
    
def non_max_suppression(tops, bots, confidence, classes, scores, iou_threshold=.5):
    confidence = np.squeeze(confidence)
    # print(confidence)
    idxs = np.argsort(confidence)[::-1]
    pick = []
    while len(idxs) > 0:
        first = idxs[0]
        idxs = idxs[1:]
        pick.append(first)
        box1 = (tops[first], bots[first])
        box2 = (tops[idxs], bots[idxs])
        ious = iou(box1, box2)
        keep_idxs = ious < iou_threshold
        idxs = idxs[keep_idxs]
    try:
        classes = classes[pick]
    except:
        classes = np.full(np.array(pick).shape, classes)
    return tops[pick], bots[pick], scores[pick], classes

def per_class_non_max_suppression(tops, bots, confidence, classes, scores, iou_threshold=.5):
    unique_classes = np.unique(classes)
    keep_tops = []
    keep_bots = []
    keep_cls = []
    keep_scores = []
    for cls in unique_classes:
        selection_mask = classes == cls
        selected_tops = tops[selection_mask]
        selected_bots = bots[selection_mask]
        selected_confidence = confidence[selection_mask]
        selected_scores = scores[selection_mask]
        ptops, pbots, pscores, pcls = non_max_suppression(selected_tops, selected_bots, selected_confidence, cls, selected_scores, iou_threshold)
        keep_tops.extend(ptops)
        keep_bots.extend(pbots)
        keep_cls.extend(pcls)
        keep_scores.extend(pscores)
    return keep_tops, keep_bots, keep_scores, keep_cls

def predict(batch_predictions, confidence_threshold=.6, iou_threshold=.5):
    pred_list = []
    for predictions in batch_predictions:
        preds = np.split(predictions, [2, 4, 5], axis=1)
        centers, sizes, confidence, classes, scores = confidence_filter(preds, confidence_threshold)
        tops, bots = to_vertices(centers, sizes)
        keep_preds = per_class_non_max_suppression(tops, bots, confidence, classes, scores)
        pred_list.append(keep_preds)
    return pred_list

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def region_np(predictions, anchor_list, input_size, num_predictions):
    num_anchors = len(anchor_list)
    output_shape = predictions.shape # if output_shape=(m, 13, 13, 255)
    grid_sz = output_shape[1:3] # grid_sz = 13, 13
    grid_dim = np.prod(grid_sz) # grid_dim = 169
    strides = np.array(input_size) / np.array(grid_sz)

    predictions = np.reshape(predictions, [-1, grid_dim*num_anchors, num_predictions]) # predictions = (m, 507, 85)
    
    anchors_xy, anchors_hw, confidence, classes = np.split(predictions, [2, 4, 5], axis=-1) # split along the last dimension

    anchors_xy = sigmoid(anchors_xy)
    confidence = sigmoid(confidence)
    classes = sigmoid(classes)
    
    grid_x_range = range(grid_sz[0])
    grid_y_range = range(grid_sz[1])
    grid_x, grid_y = np.meshgrid(grid_x_range, grid_y_range)
    grid_x = np.reshape(grid_x, [-1, 1])
    grid_y = np.reshape(grid_y, [-1, 1])
    grid_offset = np.concatenate([grid_x, grid_y], axis=-1)
    grid_offset = np.tile(grid_offset, [1, num_anchors])
    grid_offset = np.reshape(grid_offset, [1, -1, 2])
    
    anchors_xy = anchors_xy + grid_offset
    anchors_xy = anchors_xy * strides
    
    anchors = [tuple(a/strides) for a in anchor_list]
    anchors = np.tile(anchors, [grid_dim, 1])
    
    anchors_hw = anchors * np.exp(anchors_hw) * strides
    
    return np.concatenate([anchors_xy, anchors_hw, confidence, classes], axis=-1) # concat along the last dimension


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2
    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def intersection_over_union(box_1, box_2):
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape    
    resized_image_h, resized_image_w = resized_image_shape  
    scale_ratio = min(resized_image_h / orig_im_h, resized_image_w / orig_im_w)
    paddings = [(resized_image_dim - orig_im_dim*scale_ratio) / 2 for \
                resized_image_dim, orig_im_dim in zip(resized_image_shape, (orig_im_h, orig_im_w))]  
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            x = (col + predictions[box_index + 0 * side_square]) / params.side * resized_image_w
            y = (row + predictions[box_index + 1 * side_square]) / params.side * resized_image_h
            # Value for exp is very big number in some cases so following construction is using here
            try:
                w_exp = math.exp(predictions[box_index + 2 * side_square])
                h_exp = math.exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            w = w_exp * params.anchors[params.anchor_offset + 2 * n]
            h = h_exp * params.anchors[params.anchor_offset + 2 * n + 1]
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                        scale_ratio=scale_ratio, paddings=paddings))
    return objects
