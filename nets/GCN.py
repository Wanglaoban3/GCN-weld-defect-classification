import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean
from torch_geometric.data import Data
import math

class GCN(torch.nn.Module):
    def __init__(self, batch_size, hidden_channels, out_channels, num_resblock, num_gcnlayer, patch=False, attention=False):
        super(GCN, self).__init__()
        # 将ResNet作为特征提取网络，得到256*31*25大小的特征图
        self.backbone = ResNet(BasicBlock, num_resblock)
        img_tensor = self.backbone(torch.zeros(batch_size, 1, 487, 400)) #Al5083
        # img_tensor = self.backbone(torch.zeros(batch_size, 1, 320, 175)) #ss304
        shape = list(img_tensor.shape)
        if patch:
            shape[1] = shape[1] * 4
            shape[2] = int(math.ceil(shape[2] / 2))
            shape[3] = int(math.ceil(shape[3] / 2))
        # 由于每张图的无向图都相同所以提前准备无向图和对应点集的batch索引
        self.batch, self.edge_index = self.get_batch_and_edge_index(shape)

        # 实例化两层图卷积神经网络
        gcn_layers = []
        if num_gcnlayer == 1:
            gcn_layers.append(GCN_block(shape[1], out_channels, act=False, dropout=True, attention=attention))
        else:
            gcn_layers.append(GCN_block(shape[1], hidden_channels, dropout=True))
            for i in range(1, num_gcnlayer-1):
                gcn_layers.append(GCN_block(hidden_channels, hidden_channels))
            gcn_layers.append(GCN_block(hidden_channels, out_channels, act=False, attention=attention))
        self.gcn_layers = nn.Sequential(*gcn_layers)

        self.patch = patch

    def get_batch_and_edge_index(self, shape):
        batch_size, channels, height, width = shape
        # 将像素张量转换为节点张量
        num_nodes = shape[2] * shape[3]
        # 构建边的列表
        edges = []
        for i in range(height):
            for j in range(width):
                # 上下左右4邻近
                if j > 0:
                    edges.append([i * width + j, i * width + j - 1])
                if i > 0:
                    edges.append([i * width + j, (i - 1) * width + j])
                if j < width - 1:
                    edges.append([i * width + j, i * width + j + 1])
                if i < height - 1:
                    edges.append([i * width + j, (i + 1) * width + j])
                # 左上、右上、左下、右下4邻近
                if i > 0 and j > 0:
                    edges.append([i * width + j, (i - 1) * width + j - 1])
                if i > 0 and j < width - 1:
                    edges.append([i * width + j, (i - 1) * width + j + 1])
                if i < height - 1 and j > 0:
                    edges.append([i * width + j, (i + 1) * width + j - 1])
                if i < height - 1 and j < width - 1:
                    edges.append([i * width + j, (i + 1) * width + j + 1])
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # 转换成无向图
        edge_index = to_undirected(edges)
        # 构建Data对象
        edge_indexes = []
        batch = []
        for i in range(batch_size):
            batch.append(torch.ones(num_nodes)*i)
            edge_indexes.append(edge_index+(i*num_nodes))
        # 构建Batch对象
        return torch.stack(batch, dim=0).view(-1).long(), torch.stack(edge_indexes, dim=1).view(2, -1)

    def to_patch(self, feat_map):
        # 假设原始特征图为feat_map，shape为(batch_size, channel, height, width)
        # 将特征图分割成2*2的patch，得到的新特征图shape为(batch_size, 4*channel, height/2, width/2)
        if feat_map.size(2) % 2 != 0:
            feat_map = F.pad(feat_map, (0, 0, 0, 1), mode='constant', value=0)
        if feat_map.size(3) % 2 != 0:
            feat_map = F.pad(feat_map, (0, 1, 0, 0), mode='constant', value=0)
        batch = feat_map.size(0)
        channel = feat_map.size(1)
        new_h = int(feat_map.size(2) / 2)
        new_w = int(feat_map.size(3) / 2)
        new_feat_map = feat_map.view(batch, channel, new_h, 2, new_w, 2)
        new_feat_map = new_feat_map.permute(0, 1, 2, 4, 3, 5).permute(0, 2, 3, 4, 5, 1).contiguous()
        new_feat_map = new_feat_map.view(batch, new_h, new_w, -1).permute(0, 3, 1, 2).contiguous()
        return new_feat_map

    def forward(self, x):
        x = self.backbone(x)
        if self.patch:
            x = self.to_patch(x)
        # 得到了n,c,h,w特征图后，转化为n*h*w,c
        channel = x.size(1)
        x = x.view(x.size(0), channel, -1)
        x = x.transpose(1, 2)
        x = x.reshape(-1, channel)
        data = Data(x=x, edge_index=self.edge_index)
        # 将图送入图卷积神经网络
        data = self.gcn_layers(data)
        # 最后一个全局池化得到n,num_classes的张量
        x = scatter_mean(data.x, self.batch, dim=0)
        return x


class GCN_block(nn.Module):
    def __init__(self, in_channel, out_channel, act=True, dropout=False, attention=False):
        super().__init__()
        if attention:
            self.conv = GATConv(in_channel, out_channel)
        else:
            self.conv = GCNConv(in_channel, out_channel)
        self.act = act
        self.dropout = dropout

    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        if self.act:
            data.x = F.relu(data.x)
        if self.dropout:
            data.x = F.dropout(data.x, training=self.training)
        return data

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock]],
        layers: List[int],
        num_classes: int = 6,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = []
        self.layers.append(self._make_layer(block, 64, layers[0]))
        last_channels = 64
        for i in layers[1:]:
            last_channels = last_channels * 2
            self.layers.append(self._make_layer(block, last_channels, layers[i], stride=2,
                                           dilate=replace_stride_with_dilation[i-1]))
        self.layers = nn.Sequential(*self.layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block: Type[Union[BasicBlock]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)
        # out -> Tensor: (n, 256, 31, 25)
        return x


if __name__ == "__main__":
    # model = ResNet(BasicBlock, [2, 2, 2, 2])
    batch_size = 1
    # x = torch.linspace(1, 49, 49).view(1, 1, 7, 7)
    # x = x.repeat(1, 2, 1, 1)
    # x = torch.cat((x, x+100), dim=0)
    # x = GCN.to_patch(x)
    model = GCN(batch_size, 512, 6, [2, 2, 2], 1, patch=True, attention=False)
    # model.eval()

    x = torch.rand([batch_size, 1, 487, 400])
    x1, x2 = x[0, :, :, :].repeat(2, 1, 1, 1), x[1, :, :, :].repeat(2, 1, 1, 1)
    result1 = model(x)
    result2 = torch.cat((model(x1)[0].unsqueeze(0), model(x2)[0].unsqueeze(0)), dim=0)
    print(result2 - result1)
