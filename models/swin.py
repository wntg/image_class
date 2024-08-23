from timm import create_model
import timm
#from thop import profile
from torch import nn
import torch
#swin_models = timm.list_models('*efficientnet*')
# print(swin_models)
#base_model = create_model('swinv2_base_window16_384', pretrained=True)
#eff_model=create_model('tf_efficientnetv2_l', pretrained=True)
#eff_model=create_model('tf_efficientnet_b7_ns', pretrained=True)
class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.model = eff_model
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)

    def forward(self, x):
        output = self.model(x)
        return output


class Swin(nn.Module):
    def __init__(self):
        super(Swin, self).__init__()
        self.model = create_model('swinv2_base_window12to24_192to384', pretrained=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Linear(self.model.head.in_features, 2)  # 最终的二分类层
    def forward(self, x):
        x = self.model.forward_features(x)
        x = x.permute(0, 3, 1,2)  # [batch_size, num_patches, num_features]
        x = self.global_pool(x).view(x.size(0), -1)  # [batch_size, num_features]
        x = self.fc(x)
        return x
    

# # 创建一个随机输入张量，模拟一个批次大小为1的输入
# input_tensor = torch.ones(1, 3, 224, 224)
# model= EfficientNet()
# print(model)
# print(model(input_tensor))
# # 使用 thop 库计算 FLOPs 和参数量
# flops, params = profile(model, inputs=(input_tensor,))

# # 以百万（M）为单位打印参数量
# print(f'参数量: {params / 1e6:.2f} M')
# print(f'FLOPs: {flops / 1e9:.2f} G')  # FLOPs 以十亿（G）为单位