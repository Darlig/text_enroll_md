"""model.py - Model and module class for EfficientNet.
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import torch
import sys
sys.path.append('../')
import model.NetModules as NM
from torch import nn
from torch.nn import functional as F
from model.NetModules import make_mix_target
from model.efficient_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)


VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        #self._swish = MemoryEfficientSwish()
        self._swish = Swish()

    def forward(self, inputs, drop_connect_rate=None):

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=False):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    def __init__(
            self, model_name, num_classes, mute_class=None, task='base', input_type='fbank', reduction='mean', **override_params
        ):
        super().__init__()
        #assert isinstance(blocks_args, list), 'blocks_args should be a list'
        #assert len(blocks_args) > 0, 'block args must be greater than 0'
        #blocks_args, global_params = get_model_params(model_name, override_params) 
        blocks_args, global_params = self.from_name(model_name, override_params) 
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.task = task
        self.input_type = input_type
        assert (self.task in ['base', 'MT'])

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        image_size = override_params.get('image_size', None)
        if isinstance(image_size, list):
            image_size = tuple(image_size)
        if image_size == None:
            image_size = (98,80)
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 1  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        # TODO:   change input channels
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool3d((1280,1,1))
        self.mute_class = mute_class
        self._fc = nn.Linear(out_channels, num_classes)
        self.num_classes = num_classes
        self.one_hot_classes = num_classes + 1 if self.mute_class else num_classes
        #if self._global_params.include_top:
        #    self._dropout = nn.Dropout(self._global_params.dropout_rate)
        #    self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        self._swish = Swish()
        if self.task == 'base':
            self.crit = nn.CrossEntropyLoss()
            self.prob_factor = nn.Softmax(dim=-1)
        else:
            self.crit = nn.BCEWithLogitsLoss(reduction=reduction)
            self.prob_factor = nn.Sigmoid() 
        self.prob_factor = nn.Sigmoid()
        #self._swish = MemoryEfficientSwish

    def set_swish(self, memory_efficient=False):

        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x
    
    def forward_net(self, x):
        b = x.size(0)
        x = self.extract_features(x.unsqueeze(1))
        x = self._avg_pooling(x)
        x = self._fc(x.view(b,-1))
        return x
    
    def forward(self, inputs):

        detail_loss = {}
        if self.task == 'base':
            speech, target = inputs
            target = target.view(-1)
            mix_loss = 0
        elif self.task =='MT':
            if self.input_type == 'raw':
                mix_speech, speech, mix_target, target = NM.self_corrupt_input(inputs, self.num_classes)
                mix_speech = NM.compute_fbank(mix_speech)
                speech = NM.compute_fbank(speech)
            else:
                mix_speech, speech, mix_target, ratios = inputs
                mix_target = F.one_hot(mix_target, num_classes=self.one_hot_classes).to(torch.float)
                if self.mute_class: 
                    target = mix_target[:,0,0:self.mute_class]
                else:
                    target = mix_target[:,0]
                n_target = mix_target.size(1)
                mix_target = make_mix_target(mix_target, ratios[:,0:n_target], mute_class=self.mute_class, soft=False)
                mix_target = mix_target[:,0:self.mute_class]

            mix_speech = self.forward_net(mix_speech)
            mix_loss = self.crit(mix_speech, mix_target)
            detail_loss.update({'mix_det_loss': mix_loss.clone()})
        else:
            raise NotImplementedError("Only support Base and MT training method")
        speech = self.forward_net(speech)
        loss = self.crit(speech, target)
        detail_loss.update({'clean_det_loss': loss.clone()})
        loss = loss + mix_loss 
        return loss, detail_loss

    @torch.no_grad()
    def evaluate(self, inputs):
        speech, target = inputs
        logit = self.forward_net(speech)
        prob = self.prob_factor(logit)
        return prob, target

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        #model = cls(blocks_args, global_params)
        #model._change_in_channels(in_channels)
        return (blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):

        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):

        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):

        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

if __name__ == '__main__':
    m = EfficientNet(
        'efficientnet-b0', in_channels=1, num_classes=200, image_size=(98,80)
    )
    n = 0
    #m2 = EfficientNet.from_name('efficientnet-b0', in_channels=1)
    x = torch.rand(1, 98, 80)
    for k,v in m.state_dict().items():
        n += v.numel()
    print (n)
    m.set_swish(memory_efficient=False)
    o = m((x,x))
    #n = 0
    #k2n = {}
    #for k,v in m.state_dict().items():
    #    k2n[k] = v.numel()
    #    n += v.numel()
    #for k,v in k2n.items():
    #    if v > 100000:
    #        print (k, v)
