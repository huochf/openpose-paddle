# ------------------------------------------------------------------------
# This code is based on Openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/face/pose_deploy.prototxt)
# Modified from https://github.com/Hzzone/pytorch-openpose
# ------------------------------------------------------------------------

import paddle.fluid.layers as L
import paddle.fluid.dygraph as dg


def maxpool(kernel_size, stride, padding):
    return dg.Pool2D(pool_size=kernel_size, pool_type='max',
                     pool_stride=stride, pool_padding=padding,)


def conv2d(in_channels, out_channels, kernel_size, stride, padding):
    return dg.Conv2D(num_channels=in_channels, 
                     num_filters=out_channels, 
                     filter_size=kernel_size, 
                     stride=stride, 
                     padding=padding,)


def conv1x1(in_channels, out_channels, stride=1):
    return conv2d(in_channels, out_channels, 1, stride, 0)


def conv3x3(in_channels, out_channels, stride=1):
    return conv2d(in_channels, out_channels, 3, stride, 1)


def conv7x7(in_channels, out_channels, stride=1):
    return conv2d(in_channels, out_channels, 7, stride, 3)


class ReLU(dg.Layer):

    def forward(self, x):
        return L.relu(x)


def make_layers(block,):
    layers = []
    for layer_name, v in block:
        params, conv_type, activ_type = v
        if 'pool' in layer_name:
            layer = maxpool(kernel_size=params[0], stride=params[1], padding=params[2])
            layers.append((layer_name, layer))
        else:
            if conv_type == 'conv3x3':
                conv = conv3x3(params[0], params[1])
            elif conv_type == 'conv1x1':
                conv = conv1x1(params[0], params[1])
            elif conv_type == 'conv7x7':
                conv = conv7x7(params[0], params[1])
            
            layers.append((layer_name, conv))

            if activ_type == 'relu':
                activ = ReLU()
            else:
                assert activ_type == 'no_relu'
                activ = None
            
            if activ is not None:
                new_name = 'relu' + (layer_name[5:] if 'M' in layer_name else layer_name[4:])
                layers.append((new_name, activ))
    
    return layers


class PoseBlock(dg.Layer):

    def __init__(self, in_channel, hidden_channel, out_channel, 
        stage_idx, branch_idx, layer_num=5):
        super(PoseBlock, self).__init__()

        in_channels = [in_channel] + [hidden_channel] * (layer_num - 1)
        sub_layers = []
        for i in range(0, layer_num):
            sub_layers.append(('Mconv%d_stage%d_L%d' % (i + 1, stage_idx, branch_idx),
                ([in_channels[i], hidden_channel], 'conv7x7', 'relu')))
        sub_layers.append(('Mconv6_stage%d_L%d' % (stage_idx, branch_idx),
            ([hidden_channel, hidden_channel], 'conv1x1', 'relu')))
        sub_layers.append(('Mconv7_stage%d_L%d' % (stage_idx, branch_idx),
            ([hidden_channel, out_channel], 'conv1x1', 'no_relu')))
        
        sub_layers = make_layers(sub_layers)
        self.sub_layers = dg.Sequential(*sub_layers)
    

    def forward(self, x):
        return self.sub_layers(x)


class FacePose(dg.Layer):

    def __init__(self, ):
        super(FacePose, self).__init__()

        features = [
            ('conv1_1',      ([3, 64],    'conv3x3', 'relu')),
            ('conv1_2',      ([64, 64],   'conv3x3', 'relu')),
            ('pool1_stage1', ([2, 2, 0],       None,   None)),
            ('conv2_1',      ([64, 128], 'conv3x3', 'relu')),
            ('conv2_2',      ([128, 128], 'conv3x3', 'relu')),
            ('pool2_stage1', ([2, 2, 0],       None,   None)),
            ('conv3_1',      ([128, 256], 'conv3x3', 'relu')),
            ('conv3_2',      ([256, 256], 'conv3x3', 'relu')),
            ('conv3_3',      ([256, 256], 'conv3x3', 'relu')),
            ('conv3_4',      ([256, 256], 'conv3x3', 'relu')),
            ('pool3_stage1', ([2, 2, 0],       None,   None)),
            ('conv4_1',      ([256, 512], 'conv3x3', 'relu')),
            ('conv4_2',      ([512, 512], 'conv3x3', 'relu')),
            ('conv4_3',      ([512, 512], 'conv3x3', 'relu')),
            ('conv4_4',      ([512, 512], 'conv3x3', 'relu')),
            ('conv5_1',      ([512, 512], 'conv3x3', 'relu')),
            ('conv5_2',      ([512, 512], 'conv3x3', 'relu')),
            ('conv5_3_CPM',  ([512, 128], 'conv3x3', 'relu')),
        ]

        features = make_layers(features)
        self.feature = dg.Sequential(*features)

        # PAF: Part Affinity Field
        stage1 = [
            ('conv6_1_CPM', ([128, 512], 'conv1x1',   'relu')),
            ('conv6_2_CPM', ([512,  71], 'conv1x1', 'no_relu')),
        ]
        stage1 = make_layers(stage1)
        self.stage1 = dg.Sequential(*stage1)

        self.stage2 = PoseBlock(199, 128, 71, 2, 1)
        self.stage3 = PoseBlock(199, 128, 71, 3, 1)
        self.stage4 = PoseBlock(199, 128, 71, 4, 1)
        self.stage5 = PoseBlock(199, 128, 71, 5, 1)
        self.stage6 = PoseBlock(199, 128, 71, 6, 1)
    

    def forward(self, x):
        features = self.feature(x)
        x1 = self.stage1(features)
        x2 = L.concat([x1, features], 1)
        x2 = self.stage2(x2)
        x3 = L.concat([x2, features], 1)
        x3 = self.stage3(x3)
        x4 = L.concat([x3, features], 1)
        x4 = self.stage4(x4)
        x5 = L.concat([x4, features], 1)
        x5 = self.stage5(x5)
        x6 = L.concat([x5, features], 1)
        x6 = self.stage6(x6)

        return [x1, x2, x3, x4, x5, x6]
