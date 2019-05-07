import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from utils import *
import numpy as np
import tensorflow as tf

class WeightLoader(object):
    
    def __init__(self, var_list, model_type='full', path=None):
        self.model_type = model_type
        filenames = load_file_names()
        self.filename = filenames[self.model_type]
        if path is None:
            path = './data/' + self.filename
        self.url = 'https://pjreddie.com/media/files/' + self.filename
        self.var_list = var_list
        self.file = Path(path)
        self.major = 0
        self.minor = 0
        self.revision = 0
        self.seen = 0
        self.offset = 0
        self.weights = []
        self.weights_size = 0
        
    def load(self, var):
        var_shape = var.shape.as_list()
        var_size = np.prod(var_shape)
        read_from = self.offset
        read_to = read_from + var_size
        val = self.weights[read_from:read_to]
        if 'weights' in var.name:
            val = val.reshape(var_shape[3], var_shape[2], var_shape[0], var_shape[1])
            val = np.transpose(val, (2, 3, 1, 0))
        else:
            val = val.reshape(var_shape)
        self.offset = read_to
        return tf.assign(var, val, validate_shape=True)
        
    def load_now(self):
        print('\n\nChecking weights from {}\n'.format(self.file))        
        if (self.file.is_file() is False 
        or (os.path.getsize(self.file) != 248007048 and self.model_type is 'full') 
        or (os.path.getsize(self.file) != 35434956 and self.model_type is 'tiny')):
            #mess = "'{}' unexisted!".format(weight_file_path)
            #raise Exception(mess)
            print('\n{} was not found or was incomplete! Start to download from the internet.'.format(self.file))
            self.download_weights()
        print('\n\nLoading weights from {}\n'.format(self.file))
        with open(self.file, 'rb') as f:
            self.major, self.minor, self.revision = np.fromfile(f, dtype=np.int32, count=3)
            self.seen = np.fromfile(f, dtype=np.float64, count=1)
            self.weights = np.fromfile(f, dtype=np.float32)
            self.weights_size = self.weights.shape[0]
        
        load_ops = []
        now = 0
        while now < len(self.var_list):
            
            var_now = self.var_list[now]
            print(var_now)
            if 'weights' in var_now.name:
                next = now + 1
                var_next = self.var_list[next]
                if 'batch_normalization' in var_next.name:
                    num_bn_vars = 4
                    gamma, beta, moving_mean, moving_variance = self.var_list[next:next+num_bn_vars]
                    bn_vars = [beta, gamma, moving_mean, moving_variance]
                    
                    for var in bn_vars:
                        load_ops.append(self.load(var))
                        print('{} variable loaded -- read {}/{} total bytes.'.format(var.name, self.offset, self.weights_size))
                    now += num_bn_vars

                elif 'biases' in var_next.name:
                    load_ops.append(self.load(var_next))
                    print('{} variable loaded -- read {}/{} total bytes.'.format(var_next.name, self.offset, self.weights_size))
                    now = next
                
                else:
                    mess = 'Encountered unexpected next variable {}.'.format(var_next.name)
                    assert Exception(mess)
                
                load_ops.append(self.load(var_now))
                print('{} variable loaded -- read {}/{} total bytes.'.format(var_now.name, self.offset, self.weights_size))
                now += 1
                print('total loaded variables = ' + str(now))
                
            else:
                mess = 'Encountered unexpected variable {}'.format(var_now.name)
                assert Exception(mess)
                        
        #assert self.offset == self.weights_size, 'Failed! -- read {}/{} total bytes'.format(self.offset, self.weights_size)
        print('Done!')
        return load_ops
    
    
    def download_weights(self):
    
        # callback function to report the download progress.
        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = "\r%5.1f%% %*d / %d" % (
                    percent, len(str(totalsize)), readsofar, totalsize)
                sys.stderr.write(s)
                if readsofar >= totalsize: # near the end
                    sys.stderr.write("\n")
            else: # total size is unknown
                sys.stderr.write("\r_%% %d/Unknown\n" % (readsofar))
        
        print('Downloading ' + self.filename + ' from ' + self.url)
        urlretrieve(self.url, filename=self.file, reporthook=reporthook)
        print('Download complete!')
        