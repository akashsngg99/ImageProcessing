"""
    net_compiler.py
    Copyright 2017 Junhui Zhang <mrlittlezhu@gmail.com>
    Portions Copyright 2017 Xianyi Zhang <http://xianyi.github.io> and Chaowei Wang <wangchaowei@ncic.ac.cn>

This script made to build caffe net protobuf file to inferxlite[website] .c file

"""

__author__ = "Junhui Zhang <https://mrlittlepig.github.io>"
__version__ = "0.1"
__date__ = "March 4,2017"
__copyright__ = "Copyright: 2017 Junhui Zhang; Portions: 2017 Xianyi Zhang <http://xianyi.github.io>; Portions: 2017 Chaowei Wang;"

import re
import sys

from abc import abstractmethod


def isac(c):
    """
    A simple function, which determine whether the
    string element char c is belong to a decimal number
    :param c: a string element type of char
    :return: a bool type of determination
    """
    try:
        int(c)
        return True
    except:
        if c == '.' or c == '-' or c == 'e':
            return True
        else:
            return False

class LayerFactory(object):
    """
    Layer factory used to connect layer and sublayer.

    Members
    ----------
    __layer_register: a list to store layer type, which is registered in layer system.
    layer_string: contain a whole layer information.
    type: all the layers type are included in __layer_register.
    layer: which sotres layer object by __gen_layer__ function, using
        statement exec ('self.layer = %s(self.layer_string)'%self.__type)
        as self.layer = Convolution(self.layer_string) an example.
    ----------
    """

    __layer_register = ['Input', 'Convolution', 'Deconvolution', 'Pooling',
                        'Crop', 'Eltwise', 'ArgMax', 'BatchNorm', 'Concat',
                        'Scale', 'Sigmoid', 'Softmax', 'TanH', 'ReLU']

    def __init__(self, layer_string=None):
        self.layer_string = layer_string
        self.type = None

        self.layer = None
        self.__init_type__()
        self.__gen_layer__()

    def __init_type__(self):
        phase_list = self.layer_string.split('type')
        phase_num = len(phase_list)
        if phase_num == 1:
            self.type = "Input"
        elif phase_num >= 2:
            self.type = phase_list[1].split('\"')[1]

    def __gen_layer__(self):
        if self.type in self.__layer_register:
            exec ('self.layer = %s(self.layer_string)'%self.type)

class Layer(object):
    """Layer parent class"""

    __phases_string = ['name', 'type', 'bottom', 'top']
    def __init__(self, layer_string=None):
        self.layer_string = layer_string

        self.type = None
        self.name = None
        self.bottom = None
        self.top = None

        self.bottom_layer = None
        self.num_input = None
        self.num_output = None

        self.interface_c = None
        self.other = None
        self.__init_string_param__()
        #self.__list_all_member__()

    @abstractmethod
    def __calc_ioput__(self):
        """Calculate num_input and num_output"""
        pass

    @abstractmethod
    def __interface_c__(self):
        """Write the predestinate parameter into c type layer function"""
        pass

    def __init_bottom__(self):
        """Sometimes a layer has more than one bottom, so we pull it out alone"""
        bottoms_tmp = self.layer_string.split('bottom')
        bottom_num = len(bottoms_tmp)
        bottoms = []
        if bottom_num == 1:
            self.bottom = None
        else:
            for index in range(1, bottom_num):
                bottoms.append(bottoms_tmp[index].split('\"')[1])
            self.bottom = bottoms

    def __init_string_param__(self):
        """
        String parameters like name: "layername", key is name the value
        is the string type "layername", this function finds string parameters,
        which are stored in private list __phases_string, then stores the keys
        values in member variables by using exec function.
        """
        for phase in self.__phases_string:
            if phase == 'bottom':
                self.__init_bottom__()
                continue
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                continue
            elif phase_num == 2:
                exec ('self.%s=phase_list[1].split(\'\"\')[1]' % phase)
            else:
                member = []
                for index in range(1, phase_num):
                    member.append(phase_list[index].split('\"')[1])
                exec ('self.%s=member' % phase)
        print("Init string param.")

    def __init_number_param__(self, phases_number):
        """
        Number parameters like num_output: 21, key is num_output the value
        is the number 21, this function finds number parameters, which are
        stored in list phases_number, then stores the keys values in member
         variables by using exec function.
        """
        for phase in phases_number:
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                continue
            elif phase_num == 2:
                exec ('self.%s = self.__find_all_num__(phase_list[1])[0]' % phase)
            else:
                print("Error phase_num:%d" % phase_num)
        print("Init number param.")

    def __init_decimal_param__(self, phases_decimal):
        """
        Decimal parameters like eps: 0.0001, key is eps the value is the
        decimal 0.0001, this function finds decimal parameters, which are
        stored in list phases_decimal, then stores the keys values in member
         variables by using exec function.
        """
        for phase in phases_decimal:
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                continue
            elif phase_num == 2:
                exec ('self.%s = self.__find_all_decimal__(phase_list[1].split(\':\')[1])[1]' % phase)
            else:
                for index in range(1, phase_num):
                    exec ('self.%s = []' % phase)
                    exec ('self.%s.append(self.__find_all_decimal__(phase_list[index].split(\':\')[1])[1]' % phase)
        print("Init decimal param.")

    def __init_binary_param__(self, phase, default='false'):
        """
        Binary parameters like bias_term: false, key is bias_term the value
        is the bool type false, this function finds binary parameter, which
        pass in as phase, then stores the keys values in member variable by
        using exec function. Parameter default to set the default satus of
        the phase parameter
        """
        if default == 'false':
            neg_default = 'true'
        else:
            neg_default = 'false'
        phase_list = self.layer_string.split(phase)
        phase_num = len(phase_list)
        if phase_num == 1:
            exec ('self.%s = \'%s\'' % (phase, default))
        elif phase_num >= 2:
            if len(phase_list[1].split(':')[1].split(default)) == 1:
                exec ('self.%s = \'%s\'' % (phase, neg_default))
            else:
                exec ('self.%s = \'%s\'' % (phase, default))

    def __find_all_num__(self, string_phase):
        """
        A function to find series of numbers
        :param string_phase: string type key like num_output
        :return: a list stores numbers found in string_phase
        """
        number = re.findall(r'(\w*[0-9]+)\w*', string_phase)
        return number

    def __find_all_decimal__(self, string_phase):
        """
        A function to find series of decimal
        :param string_phase: string type key like moving_average_fraction
        :return: a list stores decimals found in string_phase
        """
        decimals = ""
        for index in range(len(string_phase)):
            if isac(string_phase[index]):
                decimals += string_phase[index]
            else:
                decimals += ' '
        return decimals.split(' ')


    def __list_all_member__(self):
        """Show all member variables"""
        for name, value in vars(self).items():
            if value == None:
                continue
            print('%s = %s' % (name, value))


class Input(Layer):
    """Input layer"""

    __phases_string = ['name', 'type', 'top']
    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        self.dim = []
        self.__init_dim__()
        #self.__list_all_member__()

    def __init_dim__(self):
        phase_list = self.layer_string.split('dim:')
        phase_num = len(phase_list)
        if phase_num == 1:
            print("Input layer %s has no input dims" % self.name)
        elif phase_num >= 2:
            for index in range(1, phase_num):
                self.dim.append(self.__find_all_num__(phase_list[index]))

    def __init_string_param__(self):
        if len(self.layer_string.split("type")) == 1 \
            and len(self.layer_string.split("top")) == 1:
            self.name = "input"
            self.type = "Input"
            self.top = "data"
            return
        for phase in self.__phases_string:
            if phase == 'bottom':
                self.__init_bottom__()
                continue
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                continue
            elif phase_num == 2:
                exec ('self.%s=phase_list[1].split(\'\"\')[1]' % phase)
            else:
                member = []
                for index in range(1, phase_num):
                    member.append(phase_list[index].split('\"')[1])
                exec ('self.%s=member' % phase)
        print("Init string param.")

    def __interface_c__(self):
        self.interface_c = "Input("
        for d in self.dim:
            self.interface_c += "%d" % int(d[0])
            self.interface_c += ','
        self.interface_c += '\"data_cat\",'
        self.interface_c += '\"data\");'
        print(self.interface_c)

    def __calc_ioput__(self):
        self.num_input = None
        self.num_output = int(self.dim[1][0])


class Convolution(Layer):
    """Convolution layer"""

    __phases_number = ['num_output', 'kernel_size', 'stride', 'pad', 'dilation',
                       'group', 'dilation', 'axis']
    __phases_binary = ['bias_term', 'force_nd_im2col']
    def __init__(self, layer_string=None):
        self.bias_term = None
        self.group = None
        self.axis = None
        self.kernel_size = None
        self.dilation = None
        self.stride = 1
        self.pad = 0

        Layer.__init__(self, layer_string)
        self.__init_number_param__(self.__phases_number)
        self.__init_binary_param__(self.__phases_binary[0], default='true')
        self.__init_binary_param__(self.__phases_binary[1], default='false')
        #self.__list_all_member__()

        self.kernel_h = self.kernel_size
        self.kernel_w = self.kernel_size
        self.stride_h = self.stride
        self.stride_w = self.stride
        self.pad_h = self.pad
        self.pad_w = self.pad

    def __interface_c__(self):
        self.interface_c = "Convolution("
        self.interface_c += "{},{},{},{},{},{},{},{}".\
            format(self.num_input,self.num_output,self.kernel_h,self.kernel_w,
                   self.stride_h,self.stride_w,self.pad_h,self.pad_w)
        self.interface_c += ",\"{}\",\"{}\");".format(self.bottom_layer[0].top,self.top)
        print(self.interface_c)

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output


class Deconvolution(Convolution):
    """Deconvolution layer"""

    __phases_number = ['num_output', 'kernel_size', 'stride', 'pad', 'dilation']
    def __init__(self, layer_string=None):
        Convolution.__init__(self, layer_string)

    def __interface_c__(self):
        self.interface_c = "Deconvolution("
        self.interface_c += "{},{},{},{},{},{},{},{}". \
            format(self.num_input,self.num_output,self.kernel_h,self.kernel_w,
                   self.stride_h,self.stride_w,self.pad_h,self.pad_w)
        self.interface_c += ",\"{}\",\"{}\");".format(self.bottom_layer[0].top, self.top)
        print(self.interface_c)


class Pooling(Layer):
    """Pooling layer"""

    __phases_number = ['kernel_size', 'stride', 'pad']
    __phases_binary = ['global_pooling']
    __pool_phases = ['MAX', 'AVE', 'STOCHASTIC']
    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        self.kernel_size = None
        self.stride = None
        self.pool = None
        self.global_pooling = None
        self.pad = 0

        self.__init_number_param__(self.__phases_number)
        self.__init_binary_param__(self.__phases_binary[0], default='false')
        self.__init_pool__()
        #self.__list_all_member__()

        self.kernel_h = self.kernel_size
        self.kernel_w = self.kernel_size
        self.stride_h = self.stride
        self.stride_w = self.stride
        self.pad_h = self.pad
        self.pad_w = self.pad

    def __init_pool__(self):
        for phase in self.__pool_phases:
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                print("Pooling layer %s has no pool method %s." % (self.name, phase))
            elif phase_num == 2:
                self.pool = phase
                print("Pooling layer %s has pool method %s." % (self.name, self.pool))
            else:
                print("Pool layer method error.")

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_c = "Pooling("
        self.interface_c += "{},{},{},{},{},{},{},{}". \
            format(self.num_input,self.num_output,self.kernel_h,self.kernel_w,
                   self.stride_h,self.stride_w,self.pad_h,self.pad_w)
        self.interface_c += ",\"{}\",\"{}\");".format(self.bottom_layer[0].top, self.top)
        print(self.interface_c)

class Crop(Layer):
    """Crop layer"""

    __phases_number =  ['axis', 'offset']
    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        self.__axis = None
        self.__offset = None

        self.__init_number_param__(self.__phases_number)
        #self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_c = "Crop("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\");".format(self.top)
        print(self.interface_c)


class Eltwise(Layer):
    """Eltwise layer"""

    __eltwise_phases = ['PROD', 'SUM', 'MAX']
    __phases_decimal = ['coeff']
    __phases_binary = ['stable_prod_grad']
    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        self.operation = None
        self.stabel_prod_grad = None
        self.coeff = None

        self.__init_eltwise__()
        self.__init_binary_param__(self.__phases_binary[0], default='true')
        self.__init_decimal_param__(self.__phases_decimal)
        #self.__list_all_member__()

        if not self.coeff == None:
            self.coeffa = float(self.coeff[0])
            self.coeffb = float(self.coeff[1])

    def __init_eltwise__(self):
        for phase in self.__eltwise_phases:
            phase_list = self.layer_string.split(phase)
            phase_num = len(phase_list)
            if phase_num == 1:
                print("Eltwise layer %s has no eltwise operations named %s." % (self.name, phase))
            elif phase_num == 2:
                self.operation = phase
                print("Eltwise layer %s has eltwise operations %s." % (self.name, self.operation))
            else:
                print("Eltwise layer %s layer method error." % self.name)

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_c = "Eltwise("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\");".format(self.top)
        print(self.interface_c)


class ReLU(Layer):
    """ReLU layer"""

    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        #self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_c = "ReLU("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\");".format(self.top)
        print(self.interface_c)

class ArgMax(Layer):
    """ArgMax layer"""

    __phases_number = ['top_k', 'axis']
    __phases_binary = ['out_max_val']
    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        self.out_max_val = None
        self.top_k = None
        self.axis = None

        self.__init_binary_param__(self.__phases_binary[0], default='false')
        self.__init_number_param__(self.__phases_number)
        #self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_c = "ArgMax("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\");".format(self.top)
        print(self.interface_c)


class BatchNorm(Layer):
    """BatchNorm layer"""

    __phases_decimal = ['moving_average_fraction', 'eps']
    __phases_binary = ['use_global_stats']
    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        self.use_global_stats = 'true'
        self.moving_average_fraction = '0.999'
        self.eps = '1e-5'

        self.__init_decimal_param__(self.__phases_decimal)
        self.__init_binary_param__(self.__phases_binary[0], default='true')
        #self.__list_all_member__()

    def __calc_ioput__(self):
        print(self.name)
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_c = "BatchNorm("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        if not self.moving_average_fraction == None:
            self.interface_c += ",{}".format(self.moving_average_fraction)
        if not self.eps == None:
            self.interface_c += ",{}".format(self.eps)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\");".format(self.top)
        print(self.interface_c)


class Concat(Layer):
    """Concat layer"""

    __phases_number = ['axis', 'concat_dim']
    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        self.axis = None
        self.concat_dim = None
        self.__init_number_param__(self.__phases_number)
        #self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = []
        self.num_output = 0
        for bottom in self.bottom_layer:
            self.num_input.append(bottom.num_output)
        for input in self.num_input:
            self.num_output += input

    def __interface_c__(self):
        self.interface_c = "Concat("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\");".format(self.top)
        print(self.interface_c)

class Scale(Layer):
    """Scale layer"""

    __phases_number = ['axis', 'num_axes']
    __phases_binary = ['bias_term']
    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        self.axis = None
        self.num_axes = None
        self.bias_term = None
        self.__init_number_param__(self.__phases_number)
        self.__init_binary_param__(self.__phases_binary[0], default='false')
        #self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_c = "Scale("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\");".format(self.top)
        print(self.interface_c)

class Sigmoid(Layer):
    """Sigmoid layer"""

    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        #self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer.num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_c = "Sigmoid("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\");".format(self.top)
        print(self.interface_c)

class Softmax(Layer):
    """Softmax layer"""

    __phases_number = ['axis']
    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        self.axis = None
        self.__init_number_param__(self.__phases_number)
        #self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer[0].num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_c = "Softmax("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\");".format(self.top)
        print(self.interface_c)

class TanH(Layer):
    """TanH layer"""

    def __init__(self, layer_string=None):
        Layer.__init__(self, layer_string)
        #self.__list_all_member__()

    def __calc_ioput__(self):
        self.num_input = self.bottom_layer.num_output
        self.num_output = self.num_input

    def __interface_c__(self):
        self.interface_c = "TanH("
        self.interface_c += "{},{}".format(self.num_input, self.num_output)
        for index in range(len(self.bottom_layer)):
            self.interface_c += ",\"{}\"".format(self.bottom_layer[index].top)
        self.interface_c += ",\"{}\");".format(self.top)
        print(self.interface_c)

class Net(object):
    """Convert caffe net protobuf file to inferxlite net.c file"""

    def __init__(self, proto=None):
        self.__loaded = False
        self.__proto = proto

        self.__name = None
        self.__layers_string = None
        self.__layers = []
        self.__layernum = None
        self.__txt = None
        self.__log = []
        self.__cfile = []

        self.__read_proto__()
        self.__init_layers_()
        self.__link_layers__()
        self.__write_c_format__()

    def __update_log__(self, log):
        """Print log from here"""
        print(log)
        self.__log.append(log)

    def __update_line__(self, line, outlines):
        """Print line from here"""
        print(line)
        outlines.append(line)

    def __read_proto__(self):
        """Read caffe net protobuf file"""
        try:
            self.__net = open(self.__proto, "r").read()
        except IOError:
            self.__update_log__("IOError file {} not opened." % self.__proto)
            return
        self.__layers_string = self.__net.split('layer')
        self.__update_log__("Net has been loaded successfully.")
        self.__loaded = True

    def __init_layers_(self):
        if not self.__loaded:
            self.__update_log__("Net not loaded, please check your net proto file.")
        else:
            if len(self.__layers_string[0].split("dim:")) >= 2:
                self.__layers.append(LayerFactory(layer_string=self.__layers_string[0]).layer)
            for layer_string in self.__layers_string[1:]:
                self.__layers.append(LayerFactory(layer_string=layer_string).layer)
            self.__update_log__("Layers has initialized successfully.")

    def __link_layers__(self):
        for index_i in range(len(self.__layers)):
            if self.__layers[index_i].bottom == None:
                self.__layers[index_i].__calc_ioput__()
                self.__layers[index_i].__interface_c__()
                continue
            bottom_num = len(self.__layers[index_i].bottom)
            self.__layers[index_i].bottom_layer = []
            for index_ib in range(bottom_num):
                for index_j in range(index_i):
                    if self.__layers[index_i].bottom[index_ib] == self.__layers[index_j].top:
                        self.__layers[index_i].bottom_layer.append(self.__layers[index_j])
                        break
            self.__layers[index_i].__calc_ioput__()
            self.__layers[index_i].__interface_c__()
            #print(self.__layers[index_i].name,self.__layers[index_i].num_input,self.__layers[index_i].num_output)

    def __write_c_format__(self):
        outf = open("net.c", 'w+')
        line = "#include \"common.h\"\n"
        self.__update_line__(line, self.__cfile)
        line = "#include \"interface.h\"\n\n"
        self.__update_line__(line, self.__cfile)
        line = "void net()\n{\n"
        self.__update_line__(line, self.__cfile)
        for index in range(len(self.__layers)):
            self.__update_line__("\t{}\n".format(self.__layers[index].interface_c), self.__cfile)
        self.__update_line__("\n\treturn 0;\n}", self.__cfile)
        outf.writelines(self.__cfile)


if __name__ == "__main__":
    #net = Net("deploy_resnet.prototxt")
    net = Net(str(sys.argv[1]))