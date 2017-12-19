from __future__ import division
import os
import cPickle as pickle
import sys
sys.path.append('../')
import caffe

class Net(object):
    """
    This class is defined to store net structure and parameters for pickle using
    """
    def __init__(self):
        self.net = None
        self.data = {}

def save_pickle(filepath, data):
    fid = open(filepath, 'wb')
    p = pickle.dump(data, fid)
    fid.close()
    return p

def read_pickle(filepath):
    fod = open(filepath, 'rb')
    p = pickle.load(fod)
    fod.close()
    return p

def save_net(net_file, param_file, dst_path):
    src_net = caffe.Net(net_file, param_file, caffe.TEST)
    src_layers = [k for k in src_net.params.keys()]
    dst_net = Net()
    dst_net.net = open(net_file, 'r').readlines()
    for i in range(len(src_layers)):
        layer_data = []
        for j in range(len(src_net.params[src_layers[i]])):
            data = src_net.params[src_layers[i]][j].data[...]
            layer_data.append(data)
        dst_net.data[src_layers[i]] = layer_data
    save_pickle(dst_path, dst_net)
    print dst_net

def tocaffe(src_filepath, dst_dir):
    dst_net = os.path.join(dst_dir, 'test.p')
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    pickle_net = read_pickle(src_filepath)
    print pickle_net.net
    outnet = open(dst_net, 'w')
    outnet.writelines(pickle_net.net)
    outnet.close()
    outdata = pickle_net.data

    net = caffe.Net(dst_net, caffe.TEST)

    for layername in outdata:
        for channel in range(len(outdata[layername])):
            #print "outdata[%s][%d]:" % (layername, channel), outdata[layername][channel]
            net.params[layername][channel].data[...] = outdata[layername][channel]

    dst_model = os.path.join(dst_dir, 'test.m')
    net.save(dst_model)

def remove(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

def process():
    MODEL_FILE = '../models/ResNet-50-deploy.prototxt'
    PRETRAINED = '../models/train_iter_90000.caffemodel'
    filepath = 'net'

    caffenet_path = '../models/test.p'
    caffemodel_path = '../models/test.m'

    save_net(MODEL_FILE, PRETRAINED, filepath)
    read_pickle(filepath)
    tocaffe(filepath, '../models')
    test_result(caffenet_path, PRETRAINED, caffemodel_path)
    remove(caffenet_path)
    remove(caffemodel_path)

def test_result(net, srcmodel, dstmodel):
    src_net = caffe.Net(net, srcmodel, caffe.TEST)
    dst_net = caffe.Net(net, dstmodel, caffe.TEST)
    src_layers = [k for k in src_net.params.keys()]
    dst_layers = [k for k in dst_net.params.keys()]
    for layer_index in range(len(src_layers)):
        for channel in range(len(src_net.params[src_layers[layer_index]])):
            print dst_net.params[dst_layers[layer_index]][channel].data[0]
            print src_net.params[src_layers[layer_index]][channel].data[0]

if __name__ == '__main__':
    process()
