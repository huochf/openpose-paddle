# ------------------------------------------------------------------------
# This code is based on Openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/mpi/pose_deploy_linevec.prototxt)
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


class BodyPose(dg.Layer):

    def __init__(self, ):
        super(BodyPose, self).__init__()

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
            ('conv4_3_CPM',  ([512, 256], 'conv3x3', 'relu')),
            ('conv4_4_CPM',  ([256, 128], 'conv3x3', 'relu')),
        ]

        features = make_layers(features)
        self.feature = dg.Sequential(*features)

        # PAF: Part Affinity Field
        PAF_stage1 = [
            ('conv5_1_CPM_L1', ([128, 128], 'conv3x3',   'relu')),
            ('conv5_2_CPM_L1', ([128, 128], 'conv3x3',   'relu')),
            ('conv5_3_CPM_L1', ([128, 128], 'conv3x3',   'relu')),
            ('conv5_4_CPM_L1', ([128, 512], 'conv1x1',   'relu')),
            ('conv5_5_CPM_L1', ([512,  28], 'conv1x1', 'no_relu')),
        ]
        PAF_stage1 = make_layers(PAF_stage1)
        self.PAF_stage1 = dg.Sequential(*PAF_stage1)

        # CHM: Confidence HeatMap
        CHM_stage1 = [
            ('conv5_1_CPM_L2', ([128, 128], 'conv3x3',   'relu')),
            ('conv5_2_CPM_L2', ([128, 128], 'conv3x3',   'relu')),
            ('conv5_3_CPM_L2', ([128, 128], 'conv3x3',   'relu')),
            ('conv5_4_CPM_L2', ([128, 512], 'conv1x1',   'relu')),
            ('conv5_5_CPM_L2', ([512,  16], 'conv1x1', 'no_relu')),
        ]
        CHM_stage1 = make_layers(CHM_stage1)
        self.CHM_stage1 = dg.Sequential(*CHM_stage1)

        self.PAF_stage2 = PoseBlock(172, 128, 28, 2, 1)
        self.PAF_stage3 = PoseBlock(172, 128, 28, 3, 1)
        self.PAF_stage4 = PoseBlock(172, 128, 28, 4, 1)
        self.PAF_stage5 = PoseBlock(172, 128, 28, 5, 1)
        self.PAF_stage6 = PoseBlock(172, 128, 28, 6, 1)

        self.CHM_stage2 = PoseBlock(172, 128, 16, 2, 2)
        self.CHM_stage3 = PoseBlock(172, 128, 16, 3, 2)
        self.CHM_stage4 = PoseBlock(172, 128, 16, 4, 2)
        self.CHM_stage5 = PoseBlock(172, 128, 16, 5, 2)
        self.CHM_stage6 = PoseBlock(172, 128, 16, 6, 2)
    

    def forward(self, x):
        features = self.feature(x)
        PAF1 = self.PAF_stage1(features)
        CHM1 = self.CHM_stage1(features)

        stage2_input = L.concat([PAF1, CHM1, features], 1)
        PAF2 = self.PAF_stage2(stage2_input)
        CHM2 = self.CHM_stage2(stage2_input)

        stage3_input = L.concat([PAF2, CHM2, features], 1)
        PAF3 = self.PAF_stage3(stage3_input)
        CHM3 = self.CHM_stage3(stage3_input)
        
        stage4_input = L.concat([PAF3, CHM3, features], 1)
        PAF4 = self.PAF_stage4(stage4_input)
        CHM4 = self.CHM_stage4(stage4_input)

        stage5_input = L.concat([PAF4, CHM4, features], 1)
        PAF5 = self.PAF_stage5(stage5_input)
        CHM5 = self.CHM_stage5(stage5_input)

        stage6_input = L.concat([PAF5, CHM5, features], 1)
        PAF6 = self.PAF_stage6(stage6_input)
        CHM6 = self.CHM_stage6(stage6_input)

        return [
            L.concat([CHM1, PAF1], 1),
            L.concat([CHM2, PAF2], 1),
            L.concat([CHM3, PAF3], 1),
            L.concat([CHM4, PAF4], 1),
            L.concat([CHM5, PAF5], 1),
            L.concat([CHM6, PAF6], 1),
        ]
