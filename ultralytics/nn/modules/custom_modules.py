import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Concat

from ultralytics.nn.modules import C3, C3k2, CBAM, Conv, SPPF, C2PSA
import inspect

__all__ = (
    "C3SE",  # 自定义 C3SE 模块
    "SEBlock",     # 自定义 SE 模块
    "CSConv",      # 自定义 CSConv 通道分离卷积 模块
    "ACConv",  # 自定义可调通道卷积模块
    "GConv",       # 自定义分组卷积模块
    "GConcat"     # 自定义 GConcat 模块
)


# 定义 SE (Squeeze-and-Excitation) 模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        s = self.squeeze(x).view(b, c)
        # Excitation
        e = self.excitation(s).view(b, c, 1, 1)
        return x * e

# 定义使用 SE 模块的自定义 C3 模块 (C3SE)
class C3SE(C3):  # 继承自官方的 C3 模块
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # 在官方 C3 的基础上添加 SE 模块
        self.se = SEBlock(c2)

    def forward(self, x):
        # 先通过官方 C3 的前向传播
        out = super().forward(x)
        # 然后应用 SE 注意力
        return self.se(out)

# 定义通道分离卷积模块
class CSConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1,
                 use_bn=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 模式判断：expand / compress / identity
        if out_channels > in_channels:
            assert out_channels % in_channels == 0, \
                f"out_channels({out_channels}) 必须是 in_channels({in_channels}) 的倍数"
            self.mode = "expand"
            group = in_channels
            in_per_group = 1
            out_per_group = out_channels // in_channels

        elif out_channels < in_channels:
            assert in_channels % out_channels == 0, \
                f"in_channels({in_channels}) 必须是 out_channels({out_channels}) 的倍数"
            self.mode = "compress"
            group = out_channels
            in_per_group = in_channels // out_channels
            out_per_group = 1

        else:
            self.mode = "identity"
            group = in_channels
            in_per_group = 1
            out_per_group = 1

        # 构建每组卷积模块
        self.convs = nn.ModuleList()
        for _ in range(group):
            layers = [nn.Conv2d(in_per_group, out_per_group, kernel_size, padding=padding)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_per_group))
            layers.append(nn.SiLU(inplace=True))  # 固定为 SiLU 激活
            self.convs.append(nn.Sequential(*layers))

        self.group = group
        self.in_per_group = in_per_group

    def forward(self, x):
        outputs = []
        for i in range(self.group):
            x_part = x[:, i * self.in_per_group : (i + 1) * self.in_per_group]
            y = self.convs[i](x_part)
            outputs.append(y)
        return torch.cat(outputs, dim=1)


# 定义可调通道卷积模块
class ACConv(nn.Module):
    def __init__(self, in_channels, out_channels, shift_m, kernel_size=3, stride=1, padding=1, use_bn=True, act=True):
        """
        可调通道卷积模块
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            shift_m (int): 卷积通道位移
            kernel_size (int): 卷积核大小
            stride (int): 卷积步长
            padding (int): 卷积填充
            use_bn (bool): 是否使用批归一化
            act (bool): 是否使用激活函数
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shift_m = shift_m
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bn = use_bn
        self.act = act
        
        # 计算中间参数n
        self.n = in_channels - shift_m * (out_channels - 1)
        
        # 检查n是否大于0
        if self.n <= 0:
            raise ValueError(f"计算得到的n={self.n}必须大于0。请调整shift_m或输入/输出通道数。")
        
        # 构建卷积层列表
        self.convs = nn.ModuleList()
        for i in range(out_channels):
            # 计算当前输出通道对应的输入通道范围
            start_channel = i * shift_m
            end_channel = start_channel + self.n
            
            # 确保不超出输入通道范围
            if end_channel > in_channels:
                raise ValueError(f"通道索引超出范围: {end_channel} > {in_channels}")
            
            # 为每个输出通道创建一个卷积层
            # layers = [nn.Conv2d(self.n, 1, self.kernel_size, self.stride, self.padding)]
            # 使用C3k2模块卷积
            layers = [C3k2(self.n, 1, c3k=False, e=0.25)]
            if use_bn:
                layers.append(nn.BatchNorm2d(1))
            if act:
                layers.append(nn.SiLU(inplace=True))
            self.convs.append(nn.Sequential(*layers))
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        outputs = []
        
        for i in range(self.out_channels):
            # 计算当前输出通道对应的输入通道范围
            start_channel = i * self.shift_m
            end_channel = start_channel + self.n
            
            # 提取对应的输入通道
            x_part = x[:, start_channel:end_channel]
            
            # 应用卷积
            y = self.convs[i](x_part)
            outputs.append(y)
        
        # 拼接所有输出通道
        return torch.cat(outputs, dim=1)

class GConv(nn.Module):
# 定义分组式卷积模块
    def __init__(self, c1, c2, groups, module_config):
        """
        优化版分组卷积，所有组使用相同的YOLO模块和参数
        参数：
            c1: 输入通道数
            c2: 输出通道数
            groups: 组数
            module_config: 数组，格式为 [模块名, 参数1, 参数2, ...]，如 ['Conv', 3, 2] 表示Conv模块，核大小3，步长2
        """
        super(GConv, self).__init__()

        # 验证输入和输出通道数是否能被组数整除
        if c1 % groups != 0 or c2 % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        if not isinstance(module_config, (list, tuple)) or len(module_config) < 1:  # 支持列表/元组，至少包含模块名
            raise ValueError("module_config must be a list/tuple like [module_name, param1, param2, ...]")

        self.groups = groups
        self.in_channels_per_group = c1 // groups
        self.out_channels_per_group = c2 // groups

        # 解析模块配置（数组格式）
        module_name = module_config[0]  # 模块名是数组第一个元素
        module_params = module_config[1:]  # 后续元素是参数列表

        # 支持的YOLO模块映射（保持不变）
        self.module_map = {
            'C3': C3,
            'C3k2': C3k2,
            'CBAM': CBAM,
            'Conv': Conv,
            'SPPF': SPPF,
            'C2PSA': C2PSA
        }

        # 检查模块是否支持（保持不变）
        if module_name not in self.module_map:
            raise ValueError(f"Unsupported module: {module_name}. Supported: {list(self.module_map.keys())}")

        # 根据模块名动态获取参数要求（保持不变）
        module_class = self.module_map[module_name]
        init_signature = inspect.signature(module_class.__init__)
        valid_params = list(init_signature.parameters.keys())[1:]  # 跳过self

        # 构造模块参数字典（保持逻辑，仅调整参数来源）
        params = {}
        if 'c1' in valid_params:
            params['c1'] = self.in_channels_per_group
        if 'c2' in valid_params:
            params['c2'] = self.out_channels_per_group

        # 模块参数匹配（根据数组参数调整索引）
        if module_name == 'Conv':
            params['k'] = module_params[0]  # 数组第二个元素是核大小
            params['s'] = module_params[1]  # 数组第三个元素是步长
        elif module_name == 'C3':
            params['n'] = module_params[0]   # 数组第二个元素是重复次数
            params['shortcut'] = module_params[1]  # 数组第三个元素是是否shortcut
            params['g'] = module_params[2]  # 数组第四个元素是组卷积数
        elif module_name == 'C3k2':
            params['n'] = module_params[0]   # 数组第二个元素是重复次数
            params['c3k'] = module_params[1]  # 数组第三个元素是是否使用C3k2
            params['e'] = module_params[2]   # 数组第四个元素是扩展因子
        elif module_name == 'CBAM':
            params['kernel_size'] = module_params[0]  # 数组第二个元素是核大小
        elif module_name == 'SPPF':
            params['k'] = module_params[0]   # 数组第二个元素是核大小
        elif module_name == 'C2PSA':
            params['n'] = module_params[0]   # 数组第二个元素是重复次数
            params['e'] = module_params[1]   # 数组第三个元素是扩展因子
        else:
            raise ValueError(f"Unsupported module: {module_name}")

        # 创建每组的模块（保持不变）
        module_class = self.module_map[module_name]
        self.conv_layers = nn.ModuleList([
            module_class(**params) for _ in range(groups)
        ])

    def forward(self, x):
        """
        前向传播
        参数：
            x: 输入张量，形状为 (batch_size, in_channels, height, width)
        返回：
            输出张量，形状为 (batch_size, out_channels, height, width)
        """
        batch_size, in_channels, height, width = x.size()

        # 验证输入通道数
        if in_channels != self.in_channels_per_group * self.groups:
            raise ValueError(f"Expected {self.in_channels_per_group * self.groups} input channels, got {in_channels}")

        # 将输入按通道分组
        group_inputs = torch.split(x, self.in_channels_per_group, dim=1)

        # 每组通过对应的模块处理
        group_outputs = []
        for i in range(self.groups):
            out = self.conv_layers[i](group_inputs[i])
            group_outputs.append(out)

        # 沿着通道维度拼接输出
        output = torch.cat(group_outputs, dim=1)
        return output

class GConcat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension after grouping.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
        g (int): Number of groups.
    """

    def __init__(self, dimension=1, groups=1):
        """
        Initialize GroupedConcat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
            groups (int): Number of groups.
        """
        super().__init__()
        self.d = dimension
        self.g = groups

    def forward(self, x):
        """
        Concatenate input tensors along specified dimension after grouping.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        grouped_tensors = [torch.chunk(tensor, self.g, dim=self.d) for tensor in x]
        concatenated_groups = [torch.cat(group, self.d) for group in zip(*grouped_tensors)]
        return torch.cat(concatenated_groups, self.d)
    
# if __name__ == "__main__":
    # # 示例 1: in_channels=64, out_channels=32, m=1
    # conv1 = ACConv(in_channels=64, out_channels=32, shift_m=1, kernel_size=3, padding=1)
    # x1 = torch.randn(1, 64, 20, 20)
    # y1 = conv1(x1)
    # print(f"Example 1 output shape: {y1.shape}")  # 应为 (1, 32, 20, 20)
    #
    # # 示例 2: in_channels=64, out_channels=32, m=2
    # conv2 = ACConv(in_channels=64, out_channels=32, shift_m=2, kernel_size=3, padding=1)
    # x2 = torch.randn(1, 64, 20, 20)
    # y2 = conv2(x2)
    # print(f"Example 2 output shape: {y2.shape}")  # 应为 (1, 32, 20, 20)
    #
    # # 示例 3: in_channels=64, out_channels=32, m=2, stride=2
    # conv3 = ACConv(in_channels=64, out_channels=32, shift_m=2, kernel_size=3, stride=2, padding=1)
    # x3 = torch.randn(1, 64, 20, 20)
    # y3 = conv3(x3)
    # print(f"Example 3 output shape: {y3.shape}")  # 应为 (1, 32, 10, 10)

    # # 示例 4: in_channels=64, out_channels=5, m=3, stride=2
    # conv4 = ACConv(in_channels=64, out_channels=32, shift_m=3, kernel_size=3, stride=2, padding=1)
    # x4 = torch.randn(1, 64, 20, 20)
    # y4 = conv4(x4)
    # print(f"Example 4 output shape: {y4.shape}")  # 应为 (1, 5, 10, 10)
    #

    # groupconcat示例
    # feature1 = torch.zeros((1, 8, 4, 4))  # 全为0的张量
    # feature2 = torch.ones((1, 8, 4, 4))  # 全为1的张量
    #
    # # 初始化GroupedConcat模块
    # grouped_concat = GConcat(dimension=1, groups=2)
    # normal_concat = Concat(dimension=1)
    # # 执行拼接操作
    # result1 = grouped_concat([feature1, feature2])
    # result2 = normal_concat([feature1, feature2])
    # # 输出结果
    # print(f"feature1: {feature1}")
    # print(f"feature2: {feature2}")
    # print(f"groupconcat: {result1}")
    # print(f"result1 shape: {result1.shape}")
    # print(f"normalconcat: {result2}")
    # print(f"result2 shape: {result2.shape}")