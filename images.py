import os
from pathlib import Path
from utils import *
import cv2
import numpy as np

def imresize(im, to_sizes):
    '''
    Resize the image to the specified square sizes but keeping the original aspect ratio using padding.

    Args:
        im -- input image.
        to_sizes -- output sizes, can be an integer or a tuple.

    Returns:
        resized image.
    '''
    if type(to_sizes) is int:
        to_sizes = (to_sizes, to_sizes)
        
    im_h, im_w, _ = im.shape
    to_w, to_h = to_sizes
    scale_ratio = min(to_w/im_w, to_h/im_h)
    new_im = cv2.resize(im,(0, 0), fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_CUBIC)
    new_h, new_w, _ = new_im.shape

    padded_im = np.full((to_h, to_w, 3), 128)
    x1 = (to_w-new_w)//2
    x2 = x1 + new_w
    y1 = (to_h-new_h)//2
    y2 = y1 + new_h
    padded_im[y1:y2, x1:x2, :] = new_im 
    
    return padded_im

def improcess(ims, to_sizes, to_rgb=True, normalise=True):
    '''
    Prepare an image for model's input (using OpenCV).
    Args:
        ims -- input images.
        to_sizes -- output sizes, can be an integer or a tuple.
        flip_color_channel -- flip the colour channel from BGR to RGB, set this to False if use other image processing libraries.
    
    Returns:
        A resized and normalised image.
    '''
    imlist = []
    for im in ims:
        if to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = imresize(im, to_sizes)
        imlist.append(im)
    imlist = np.array(imlist)
    if normalise: imlist = imlist / 255
    return imlist


def imread_from_path(im_path):
    '''
    read one or all images from im_path.

    Args:
        dir_path -- path to image or image directory.
    
    Returns:
        list of images.
    '''
    p = Path(os.path.abspath(im_path))
    if os.path.isdir(p):
        ims = [p.joinpath(imname) for imname in os.listdir(p)]
    else:
        ims = [p]
    return [cv2.imread(str(im), cv2.IMREAD_COLOR) for im in ims]
    

def rescale_vertex(vtx, from_wh, to_wh):
    if from_wh is int:
        from_wh = (from_wh, from_wh)
    if to_wh is int:
        to_wh = (to_wh, to_wh)
    from_wh = np.array(from_wh)
    to_wh = np.array(to_wh)
    scale_ratio = min(from_wh[0]/to_wh[0], from_wh[1]/to_wh[1])
    pad = (from_wh - scale_ratio*to_wh) // 2
    vtx = (vtx - pad) / scale_ratio
    return vtx.astype(np.int32)


def add_overlays_v1(frame, preds, pred_wh, labels, palette):
    tops, bots, scores, classes = preds
    if not tops:
        return frame
    frame_wh = frame.shape[:-1][::-1]
    vtcs = np.concatenate([tops, bots], axis=0)
    vtcs = rescale_vertex(vtcs, pred_wh, frame_wh)
    tops, bots = np.split(vtcs, 2)
    b_thick = np.int(np.sum(frame_wh) // 1000)
    t_thick = (b_thick//3)+1
    # print('b_thick={}'.format(b_thick))
    # print('t_thick={}'.format(t_thick))
    t_scale = 8e-4*np.min(frame_wh)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    for top, bot, cls in zip(tops, bots, classes):
        colour = palette[cls]
        top = tuple(top)
        bot = tuple(bot)
        # txt = '{}:{}%'.format(labels[cls], int(round(score*100)))
        frame = cv2.rectangle(frame, top, bot, colour, b_thick)
        txt = '{}'.format(labels[cls])
        t_size = cv2.getTextSize(txt, font_face, t_scale, t_thick)[0]
        t_box_bot = top
        t_box_top = (t_box_bot[0] + t_size[0] + b_thick*4, t_box_bot[1] - t_size[1] - b_thick*6)
        t_orig = top[0]+b_thick*2, top[1]-b_thick*4
        if t_box_top[1] < 0:
            t_box_top = top
            t_box_bot = (t_box_top[0] + t_size[0] + b_thick*4, t_box_top[1] + t_size[1] + b_thick*6)
            t_orig = top[0] + b_thick*2, top[1] + t_size[1] + b_thick*2
        frame = cv2.rectangle(frame, t_box_top, t_box_bot, colour, -1)
        frame = cv2.putText(frame, txt, t_orig, font_face, t_scale, (255, 255, 255), t_thick)
    return frame


b_thick = 3
t_thick = 2
t_scale = 1
def add_overlays_v2(obj, orig_frame, labels_map, palette):
    origin_im_size = orig_frame.shape[:-1]
    if obj['ymax'] > origin_im_size[0]:
        obj['ymax'] = origin_im_size[0] - b_thick
    if obj['xmax'] > origin_im_size[1]:
        obj['xmax'] = origin_im_size[1] - b_thick
    if obj['xmin'] < 0:
        obj['xmin'] = 0 + b_thick
    if obj['ymin'] < 0:
        obj['ymin'] = 0 + b_thick
    det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
        str(obj['class_id'])
    txt = '{}'.format(det_label)
    colour = palette[obj['class_id']]
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(orig_frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), colour, b_thick)
    t_size = cv2.getTextSize(txt, font_face, t_scale, t_thick)[0]
    if obj['xmin'] + t_size[0] > origin_im_size[1]:
        obj['xmin'] = origin_im_size[1] - t_size[0] - b_thick*2
    t_box_bot = obj['xmin'], obj['ymin']
    t_box_top = (t_box_bot[0] + t_size[0] + b_thick, t_box_bot[1] - t_size[1] - b_thick*3)
    
    t_orig = t_box_bot[0]+b_thick, t_box_bot[1]-b_thick*2
    if t_box_top[1] < 0:
        t_box_top = t_box_bot
        t_box_bot = (t_box_top[0] + t_size[0] + b_thick, t_box_top[1] + t_size[1] + b_thick*3)
        t_orig = t_box_top[0] + b_thick, t_box_top[1] + t_size[1] + b_thick
    cv2.rectangle(orig_frame, t_box_top, t_box_bot, colour, -1)
    cv2.rectangle(orig_frame, t_box_top, t_box_bot, colour, b_thick)
    cv2.putText(orig_frame, txt, t_orig, font_face, t_scale, (0, 0, 0), t_thick)
    return orig_frame


def visualise(orig_imlist, pred_list, input_size, labels, palette):
    '''
    Visualise predictions on original images.
    '''
    imlist = []
    for im, preds in zip(orig_imlist, pred_list):
        tops, bots, scores, classes = preds
        if tops:
            im_size = im.shape[:-1][::-1]
            vtcs = np.concatenate([tops, bots], axis=0)
            vtcs = rescale_vertex(vtcs, input_size, im_size)
            tops, bots = np.split(vtcs, 2)
            for top, bot, score, cls in zip(tops, bots, scores, classes):
                obj = dict(xmin=top[0], ymin=top[1], xmax=bot[0], ymax=bot[1],
                            class_id=cls, confidence=score)
                im = add_overlays_v2(obj, im, labels, palette)
                
        imlist.append(im)
        # imlist.append(add_overlays(im, preds, input_size, labels, palette))
    return imlist


def scale_bbox(x, y, h, w, class_id, confidence, scale_ratio, paddings):
    ypad, xpad = paddings
    xmin = int((x - w / 2 - xpad) / scale_ratio)
    ymin = int((y - h / 2 - ypad) / scale_ratio)
    xmax = int(xmin + w / scale_ratio)
    ymax = int(ymin + h / scale_ratio)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def imwrite(ims):
    p = Path('.').absolute()
    save_dir = p.joinpath('outputs').as_posix()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    [cv2.imwrite(save_dir + '/{}.jpg'.format(i), im) for i, im in zip(range(len(ims)), ims)]
    print('Images have been saved to {}'.format(save_dir))

def display_mess(frame, mess):
    cv2.putText(frame, mess, (2, 20), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, 2)
    return frame
