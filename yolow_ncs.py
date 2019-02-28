import os, sys, logging
from argparse import ArgumentParser
from openvino.inference_engine import IEPlugin, IENetwork
from utils import *
from predict import *

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO, stream=sys.stdout)
log=logging.getLogger()

class YolowNCS(object):

    _ANCHORS = anchors_for_yolov3()

    def __init__(self, model_name=None):
        if model_name is None:
            model_name = 'ir/frozen_yolow'
        self.model=model_name + '.xml'
        self.weights=model_name + '.bin'
        self.plugin=IEPlugin(device='MYRIAD')
        log.info('Loading network files:\n\t{}\n\t{}'.format(self.model, self.weights))
        self.net=IENetwork(model=self.model, weights=self.weights)
        log.info('Preparing inputs')
        self.input_blob=next(iter(self.net.inputs))
        self.net.batch_size=1
        log.info('Loading model to the plugin')
        self.exec_net=self.plugin.load(network=self.net)

    def predict(self, input_list, confidence_theshold=.6, iou_theshould=.5):
        batch_predictions = []
        get_from = 0
        input_size = input_list.shape[2]
        input_dict = {self.input_blob: input_list}
        pred_dict = self.exec_net.infer(inputs=input_dict)
        for preds in pred_dict.values():
            preds = np.transpose(preds, [0, 2, 3, 1])
            get_to = get_from + 3
            batch_predictions.append(region_np(preds, self._ANCHORS[get_from:get_to], input_size))
            get_from = get_to
        batch_predictions = np.concatenate(batch_predictions, axis=1)
        return predict(batch_predictions, confidence_theshold, iou_theshould)
