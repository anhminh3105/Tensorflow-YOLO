import os, sys, logging
from argparse import ArgumentParser
from openvino.inference_engine import IEPlugin, IENetwork
from utils import *
from predict import *

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO, stream=sys.stdout)
log=logging.getLogger()

class YolowNCS(object):

    def __init__(self, model_type='full', num_requests=2):
        model_name = 'data/ir/frozen_yolow_' + model_type
        self._ANCHORS = anchors_for_yolov3(model_type)
        self.model=model_name + '.xml'
        self.weights=model_name + '.bin'
        self.plugin=IEPlugin(device='MYRIAD')
        log.info('Loading network files:\n\t{}\n\t{}'.format(self.model, self.weights))
        self.net=IENetwork(model=self.model, weights=self.weights)
        log.info('Preparing inputs')
        self.input_blob=next(iter(self.net.inputs))
        self.net.batch_size=1
        log.info('Loading model to the plugin')
        self.current_request_id = 0
        self.next_request_id = 1
        self.num_requests = num_requests
        self.exec_net=self.plugin.load(network=self.net, num_requests=self.num_requests)

    def predict(self, input_list, confidence_theshold=.6, iou_theshould=.5, async_mode=False):
        batch_predictions = []
        get_from = 0
        input_size = input_list.shape[2]
        input_dict = {self.input_blob: input_list}
        request_handle = self.exec_net.requests[self.current_request_id]
        if async_mode:
            next_request_id = self.current_request_id + 1
            if next_request_id == self.num_requests:
                next_request_id = 0
        else:
            request_handle.wait()
            next_request_id = self.current_request_id
        self.exec_net.start_async(request_id=next_request_id,
                                  inputs=input_dict)
        if async_mode:
            self.current_request_id = next_request_id
        request_handle.wait()
        pred_dict = request_handle.outputs
        for preds in pred_dict.values():
            preds = np.transpose(preds, [0, 2, 3, 1])
            get_to = get_from + 3
            batch_predictions.append(region_np(preds, self._ANCHORS[get_from:get_to], input_size))
            get_from = get_to
        batch_predictions = np.concatenate(batch_predictions, axis=1)
        return predict(batch_predictions, confidence_theshold, iou_theshould)
