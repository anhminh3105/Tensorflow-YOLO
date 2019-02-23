import numpy as np

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
    if type(classes) is np.int64:
        classes = np.full(np.array(pick).shape, classes)
    else:
        classes = classes[pick]
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
        centers, sizes, confidence, classes, scores = confidence_filter(preds)
        tops, bots = to_vertices(centers, sizes)
        keep_preds = per_class_non_max_suppression(tops, bots, confidence, classes, scores)
        pred_list.append(keep_preds)
    return pred_list

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def region_np(predictions, anchor_list, input_size):
    num_anchors = len(anchor_list)
    output_shape = predictions.shape # if output_shape=(m, 13, 13, 255)
    grid_sz = output_shape[1] # grid_sz = 13
    grid_dim = grid_sz * grid_sz # grid_dim = 169
    stride = input_size // grid_sz
    
    predictions = np.reshape(predictions, [-1, grid_dim*num_anchors, 85]) # predictions = (m, 507, 85)
    
    anchors_xy, anchors_hw, confidence, classes = np.split(predictions, [2, 4, 5], axis=-1) # split along the last dimension

    anchors_xy = sigmoid(anchors_xy)
    confidence = sigmoid(confidence)
    classes = sigmoid(classes)
    
    grid_sz_range = range(grid_sz)
    grid_x, grid_y = np.meshgrid(grid_sz_range, grid_sz_range)
    grid_x = np.reshape(grid_x, [-1, 1])
    grid_y = np.reshape(grid_y, [-1, 1])
    grid_offset = np.concatenate([grid_x, grid_y], axis=-1)
    grid_offset = np.tile(grid_offset, [1, num_anchors])
    grid_offset = np.reshape(grid_offset, [1, -1, 2])
    
    anchors_xy = anchors_xy + grid_offset
    anchors_xy = anchors_xy * stride
    
    anchors = [(a[0] / stride, a[1] / stride) for a in anchor_list]
    anchors = np.tile(anchors, [grid_dim, 1])
    
    anchors_hw = anchors * np.exp(anchors_hw) * stride
    
    return np.concatenate([anchors_xy, anchors_hw, confidence, classes], axis=-1) # concat along the last dimension