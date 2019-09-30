import torch.nn as nn
from typing import Callable, Union, Tuple

from pointcnn_utils.util_funcs import UFloatTensor
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

def EndChannels(f, make_contiguous = False):
    """ Class decorator to apply 2D convolution along end channels. """

    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, x):
            x = x.permute(0,3,1,2)
            x = self.f(x)
            x = x.permute(0,2,3,1)
            return x

    return WrappedLayer()

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(nonlinear_layer_org, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_tilda = F.tanh(self.fc1(x))
        g = F.sigmoid(self.gate(x))
        y = y_tilda * g
        return y

class _Dense(nn.Module):
    """
    Single layer perceptron with optional activation, batch normalization, and dropout.
    """

    def __init__(self, in_features : int, out_features : int,
                 drop_rate : int = 0, with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ELU(),
                 dim : int = 1
                ) -> None:
        """
        :param in_features: Length of input featuers (last dimension).
        :param out_features: Length of output features (last dimension).
        :param drop_rate: Drop rate to be applied after activation.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Dense, self).__init__()

        self.linear = weight_norm(nn.Linear(in_features, out_features))
        self.activation = activation
        self.bn1 = LayerNorm(out_features, dim=1, momentum=0.99)
        self.bn2 = LayerNorm(out_features, dim=2, momentum=0.99)
        self.with_bn = with_bn
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Linear.
        :return: Tensor with linear layer and optional activation, batchnorm,
        and dropout applied.
        """
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        if self.with_bn:
            _x = x
            try:
                if _x.dim() == 4:
                    _x = _x.permute(0, 3, 1, 2)
                    _x = self.bn2(_x)
                    _x = _x.permute(0, 2, 3, 1)
                elif _x.dim() == 3:
                    _x = _x.permute(0, 2, 1)
                    _x = self.bn1(_x)
                    _x = _x.permute(0, 2, 1)
            except:
                print(x.size())
            x = _x
        if self.drop:
            x = self.drop(x)
        return x


class Dense(nn.Module):
    """
    Single layer perceptron with optional activation, batch normalization, and dropout.
    """

    def __init__(self, in_features : int, out_features : int,
                 drop_rate : int = 0, with_bn : bool = True,
                 activation : bool = True,
                 dim : int = 1
                ) -> None:
        """
        :param in_features: Length of input featuers (last dimension).
        :param out_features: Length of output features (last dimension).
        :param drop_rate: Drop rate to be applied after activation.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Dense, self).__init__()

        self.linear1 = weight_norm(nn.Linear(in_features, out_features))
        self.linear2 = weight_norm(nn.Linear(in_features, out_features))
        self.activation = activation
        self.bn1 = LayerNorm(out_features, dim=1, momentum=0.99)
        self.bn2 = LayerNorm(out_features, dim=2, momentum=0.99)
        self.with_bn = with_bn
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, in_x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Linear.
        :return: Tensor with linear layer and optional activation, batchnorm,
        and dropout applied.
        """
        x = self.linear1(in_x)
        if self.activation:
            x2 = self.linear2(in_x)
            y_tilda = F.tanh(x)
            g = F.sigmoid(x2)
            x = y_tilda * g
        if self.with_bn:
            _x = x
            try:
                if _x.dim() == 4:
                    _x = _x.permute(0, 3, 1, 2)
                    _x = self.bn2(_x)
                    _x = _x.permute(0, 2, 3, 1)
                elif _x.dim() == 3:
                    _x = _x.permute(0, 2, 1)
                    _x = self.bn1(_x)
                    _x = _x.permute(0, 2, 1)
            except:
                print(x.size())
            x = _x
        if self.drop:
            x = self.drop(x)
        return x

class Conv(nn.Module):
    """
    2D convolutional layer with optional activation and batch normalization.
    """

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : Union[int, Tuple[int, int]], with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ELU()
                ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias = not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.99) if with_bn else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with convolutional layer and optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class SepConv(nn.Module):
    """ Depthwise separable convolution with optional activation and batch normalization"""

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : Union[int, Tuple[int, int]],
                 depth_multiplier : int = 1, with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ELU()
                 ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param depth_multiplier: Depth multiplier for middle part of separable convolution.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(SepConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups = in_channels),
            nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, bias = not with_bn)
        )

        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.99) if with_bn else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x


class ChanConv(nn.Module):
    """ Depthwise separable convolution with optional activation and batch normalization"""

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : Union[int, Tuple[int, int]],
                 depth_multiplier : int = 1, with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ELU()
                 ) -> None:
        """
        :param in_channels: Length of input features (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param depth_multiplier: Depth multiplier for middle part of separable convolution.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(ChanConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups = in_channels),
        )

        self.activation = activation
        self.bn = nn.BatchNorm2d(in_channels * depth_multiplier, momentum=0.99) if with_bn else None
        self.K = out_channels

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.size(0), x.size(2), self.K, self.K)
        return x


class LayerNorm(nn.Module):
    """
    Batch Normalization over ONLY the mini-batch layer (suitable for nn.Linear layers).
    """

    def __init__(self, N : int, dim : int, *args, **kwargs) -> None:
        """
        :param N: Batch size.
        :param D: Dimensions.
        """
        super(LayerNorm, self).__init__()
        if dim == 1:
            self.bn = nn.BatchNorm1d(N, *args, **kwargs)
        elif dim == 2:
            self.bn = nn.BatchNorm2d(N, *args, **kwargs)
        elif dim == 3:
            self.bn = nn.BatchNorm3d(N, *args, **kwargs)
        else:
            raise ValueError("Dimensionality %i not supported" % dim)

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        return self.bn(x)
