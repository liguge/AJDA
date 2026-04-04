
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, frequency, eps=0.3, mode='sigmoid'):
        super(Laplace_fast, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps
        self.mode = mode
        self.fre = frequency
        self.a_ = torch.linspace(0, self.out_channels, self.out_channels).view(-1, 1)
        self.b_ = torch.linspace(0, self.out_channels, self.out_channels).view(-1, 1)
        self.time_disc = torch.linspace(0, self.kernel_size - 1, steps=int(self.kernel_size))

    def Laplace(self, p):
        # m = 1000
        # ep = 0.03
        # # tal = 0.1
        # f = 80
        w = 2 * torch.pi * self.fre
        # A = 0.08
        q = torch.tensor(1 - pow(0.03, 2))

        if self.mode == 'vanilla':
            return ((1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))) * (torch.sin(w * (p - 0.1))))

        if self.mode == 'maxmin':
            a = (1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (
                torch.sin(w * (p - 0.1)))
            return (a-a.min())/(a.max()-a.min())

        if self.mode == 'sigmoid':
            return (1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).sigmoid()) * (
                torch.sin(w * (p - 0.1)))

        if self.mode == 'softmax':
            return (1/math.e) * torch.exp(F.softmax((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)), dim=-1)) * (
                torch.sin(w * (p - 0.1)))

        if self.mode == 'tanh':
            return (1/math.e) * torch.exp(((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1))).tanh()) * (torch.sin(w * (p - 0.1)))

        if self.mode == 'atan':
            return (1/math.e) * torch.exp((2 / torch.pi) * (((-0.03 / (torch.sqrt(q))) * (w * (p - 0.1)))).atan()) * (
                torch.sin(w * (p - 0.1)))

    def forward(self):
        p1 = (self.time_disc - self.b_) / (self.a_ + self.eps)
        return self.Laplace(p1).view(self.out_channels, 1, self.kernel_size)


# class CNN(nn.Module):
#     """可调试的故障诊断模型"""
#
#     def __init__(self, pretrained=False):
#         super().__init__()
#         self.output_num = 256
#         self.feature_extractor = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=31),
#             nn.BatchNorm1d(16),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2),
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten()
#         )
#         # 分类器
#         # self.fc1 = nn.Linear(self.output_dim, 128)
#         # self.fc2 = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         features = self.feature_extractor(x)
#         # features = self.fc1(features)
#         # features = F.relu(features)
#         # logits = self.fc2(features)
#         return features
class CNN(nn.Module):
    """可调试的故障诊断模型"""

    def __init__(self, wavelet_initialize_weights=True, zero_init_residual=False):
        super().__init__()
        self.output_num = 256

        # 🔥 将首层命名为 'conv1' 以便特殊初始化
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=31),  # 原来的 conv1
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv_layers = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        if wavelet_initialize_weights:
            self._initialize_weights()
            if zero_init_residual:
                self._zero_init_last_bn()

        self._initialized_weights = True

    def _initialize_weights(self):
        """初始化权重 - 包含特殊层初始化"""
        for name, module in self.named_children():
            if name == 'conv1':
                # 🔥 对 conv1 应用特殊初始化
                self._initialize_conv1(module)
            else:
                # 🔥 对其他层应用标准初始化
                self._initialize_other_modules(module)

    def _initialize_conv1(self, conv1_module):
        """对 conv1 应用 Laplace 滤波器初始化"""
        for m in conv1_module.modules():
            if isinstance(m, nn.Conv1d):
                # 🔥 检查是否为首层卷积 (kernel_size=64)
                if m.kernel_size == (64,):  # 首层卷积
                    try:
                        # 🔥 使用您的 Laplace 初始化
                        laplace_filter = Laplace_fast(
                            out_channels=m.out_channels,
                            kernel_size=m.kernel_size[0],
                            eps=1.0,
                            frequency=50000,
                            mode='sigmoid'
                        ).forward()

                        # 确保维度匹配
                        if laplace_filter.shape == m.weight.data.shape:
                            m.weight.data = laplace_filter
                            print(f"✅ Conv1D {m.kernel_size} 使用 Laplace 滤波器初始化")
                        else:
                            print(f"⚠️ Laplace 滤波器维度不匹配，使用默认初始化")
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    except Exception as e:
                        print(f"⚠️ Laplace 初始化失败: {e}，使用默认初始化")
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    # 其他 conv 层使用标准初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _initialize_other_modules(self, module):
        """对非 conv1 的模块应用标准初始化"""
        for m in module.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _zero_init_last_bn(self):
        """零初始化最后一个BN层权重"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_layers(x)
        features = self.global_pool(x)
        return features


if __name__ == "__main__":
    model = CNN()
    input = torch.randn((512, 1, 3072))
    output = model(input)
    print(output[0].shape)