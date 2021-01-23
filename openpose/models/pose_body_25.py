# ------------------------------------------------------------------------
# This code is based on Openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/body_25/pose_deploy.prototxt)
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


def conv3x3(in_channels, out_channels, stride=1):
    return dg.Conv2D(num_channels=in_channels,
                     num_filters=out_channels,
                     filter_size=3,
                     stride=stride,
                     padding=1)


def conv1x1(in_channels, out_channels, stride=1):
    return dg.Conv2D(num_channels=in_channels,
                     num_filters=out_channels,
                     filter_size=1,
                     stride=stride,
                     padding=0)


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
            else:
                raise ValueError
            
            layers.append((layer_name, conv))

            if activ_type == 'relu':
                activ = ReLU()
            elif activ_type == 'prelu':
                activ = dg.PRelu(mode='channel', channel=params[1])
            else:
                assert activ_type == 'no_relu'
                activ = None
            
            if activ is not None:
                if layer_name[0] == 'M':
                    new_name = 'M' + activ_type + layer_name[5:]
                else:
                    new_name = activ_type + layer_name[4:]

                layers.append((new_name, activ))
    
    return layers


class DenseNetBlock(dg.Layer):

    def __init__(self, in_channel, hidden_channel):
        super(DenseNetBlock, self).__init__()
        self.conv0 = conv3x3(in_channel, hidden_channel)
        self.prelu0 = dg.PRelu(mode='channel', channel=hidden_channel)
        self.conv1 = conv3x3(hidden_channel, hidden_channel)
        self.prelu1 = dg.PRelu(mode='channel', channel=hidden_channel)
        self.conv2 = conv3x3(hidden_channel, hidden_channel)
        self.prelu2 = dg.PRelu(mode='channel', channel=hidden_channel)
    

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.prelu0(x0)
        x1 = self.conv1(x0)
        x1 = self.prelu1(x1)
        x2 = self.conv2(x1)
        x2 = self.prelu2(x2)
        return L.concat([x0, x1, x2], axis=1)


class PoseBlock(dg.Layer):

    def __init__(self, in_channel, hidden_channel1, hidden_channel2, out_channel):
        super(PoseBlock, self).__init__()

        self.sub_block1 = DenseNetBlock(in_channel         , hidden_channel1)
        self.sub_block2 = DenseNetBlock(hidden_channel1 * 3, hidden_channel1)
        self.sub_block3 = DenseNetBlock(hidden_channel1 * 3, hidden_channel1)
        self.sub_block4 = DenseNetBlock(hidden_channel1 * 3, hidden_channel1)
        self.sub_block5 = DenseNetBlock(hidden_channel1 * 3, hidden_channel1)
        self.sub_block6 = dg.Sequential(
                ('conv', conv1x1(hidden_channel1 * 3, hidden_channel2)),
                ('prelu', dg.PRelu(mode='channel', channel=hidden_channel2)),
            )
        self.sub_block7 = dg.Sequential(
                ('conv', conv1x1(hidden_channel2, out_channel))
            )
    

    def forward(self, x):
        x = self.sub_block1(x)
        x = self.sub_block2(x)
        x = self.sub_block3(x)
        x = self.sub_block4(x)
        x = self.sub_block5(x)
        x = self.sub_block6(x)
        x = self.sub_block7(x)
        return x


class BodyPose(dg.Layer):

    def __init__(self, ):
        super(BodyPose, self).__init__()

        feature = [
            ('conv1_1',      ([3, 64],    'conv3x3', 'relu')),
            ('conv1_2',      ([64, 64],   'conv3x3', 'relu')),
            ('pool1_stage1', ([2, 2, 0],       None,   None)),
            ('conv2_1',      ([64, 128],  'conv3x3', 'relu')),
            ('conv2_2',      ([128, 128], 'conv3x3', 'relu')),
            ('pool2_stage1', ([2, 2, 0],       None,   None)),
            ('conv3_1',      ([128, 256], 'conv3x3', 'relu')),
            ('conv3_2',      ([256, 256], 'conv3x3', 'relu')),
            ('conv3_3',      ([256, 256], 'conv3x3', 'relu')),
            ('conv3_4',      ([256, 256], 'conv3x3', 'relu')),
            ('pool3_state1', ([2, 2, 0],       None,   None)),
            ('conv4_1',      ([256, 512], 'conv3x3', 'relu')),
            ('conv4_2',      ([512, 512], 'conv3x3', 'relu')),
            ('conv4_3_CPM',  ([512, 256], 'conv3x3','prelu')),
            ('conv4_4_CPM',  ([256, 128], 'conv3x3','prelu')),
        ]
        feature = make_layers(feature)
        self.features = dg.Sequential(*feature)

        # Part Affinity Field blocks
        self.PAF_block0 = PoseBlock(128,  96, 256, 52)
        self.PAF_block1 = PoseBlock(180, 128, 512, 52)
        self.PAF_block2 = PoseBlock(180, 128, 512, 52)
        self.PAF_block3 = PoseBlock(180, 128, 512, 52)
        
        # Confidence Heatmap blocks
        self.CHM_block0 = PoseBlock(180,  96, 256, 26)
        self.CHM_block1 = PoseBlock(206, 128, 512, 26)


    def forward(self, img):
        feature = self.features(img) # [bs, 128, h, w]

        PAF0 = self.PAF_block0(feature)
        PAF1 = L.concat((feature, PAF0), 1)
        PAF1 = self.PAF_block1(PAF1)
        PAF2 = L.concat((feature, PAF1), 1)
        PAF2 = self.PAF_block2(PAF2)
        PAF3 = L.concat((feature, PAF2), 1)
        PAF3 = self.PAF_block3(PAF3)

        CHM0 = L.concat((feature, PAF3), 1)
        CHM0 = self.CHM_block0(CHM0)
        CHM1 = L.concat((feature, CHM0, PAF3), 1)
        CHM1 = self.CHM_block1(CHM1)

        return [
            [PAF0, PAF1, PAF2, PAF3, CHM0, CHM1],
            L.concat((CHM1, PAF3), 1)
        ]
