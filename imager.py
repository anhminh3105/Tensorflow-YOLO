import numpy as np
from utils import *
from images import *

np.random.seed(2)

class Imager(object):
    
    def __init__(self, input_size, labels):
        if type(input_size) is int:
            self.input_size = (input_size, input_size)
        else:
            self.input_size = input_size
        self.labels = labels
        self.palette = np.random.randint(0, 256, (len(self.labels), 3)).tolist()

    def imset_from_path(self, path):
        ims = np.array(imread_from_path(path))
        if len(ims.shape) == 3:
            ims = [ims]
        self.ims = ims

    def imset(self, ims):
        ims = np.array(ims)
        if len(ims.shape) == 3:
            ims = [ims]
        self.ims = ims

    def preprocess(self):
        return improcess(self.ims, self.input_size)
    
    def ncs_preprocess(self):
        ims = improcess(self.ims, self.input_size, to_rgb=False, normalise=False) # ims are normalised by the ncs.
        ims = np.transpose(np.array(ims), [0, 3, 1, 2])
        return np.expand_dims(ims, 1)

    def visualise_preds(self, pred_list):
        self.ims = visualise(self.ims, pred_list, self.input_size, self.labels, self.palette)
        return self.ims

    def ncs_visualise_preds(self, objects_list):
        imlist = list()
        for im, objects in zip(self.ims, objects_list):
            if not objects:
                imlist.append(im)
                continue
            for obj in objects:
                add_overlays_v2(obj, im, self.labels, self.palette)
            imlist.append(im)
        self.ims = imlist
        return self.ims

    def imsave(self, ims):
        imwrite(ims)

