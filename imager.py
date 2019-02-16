import numpy as np
from utils import *
from images import *
from yolov3 import *
np.random.seed(2)

class Imager(object):
    
    def __init__(self, transform_sizes=416, class_name_file='./data/coco.names'):
        if type(transform_sizes) is int:
            self.transform_sizes = (transform_sizes, transform_sizes)
        else:
            self.transform_sizes = transform_sizes
        self.namelist = load_class_names(class_name_file)
        self.palette = np.random.randint(0, 256, (len(self.namelist), 3)).tolist()

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

    def preproces(self):
        return improcess(self.ims, self.transform_sizes)

    def visualise_preds(self, pred_list):
        return visualise(self.ims, pred_list, self.transform_sizes, self.namelist, self.palette)

    def imsave(self, ims):
        imwrite(ims)
