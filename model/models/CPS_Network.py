import torch.nn as nn
from .FCN_backbone import FCNs_VGG
from .SEUNet import se_resnext50_32x4d
import torch


class FCNs_CPS(nn.Module):
    def __init__(self, in_ch, out_ch, backbone='vgg16_bn', pretrained=True):
        super(FCNs_CPS, self).__init__()
        self.name = "FCNs_CPS_" + backbone
        self.backbone = backbone
        # self.branch1 = FCNs_VGG(in_ch, out_ch, backbone=backbone, pretrained=pretrained)
        # self.branch2 = FCNs_VGG(in_ch, out_ch, backbone=backbone, pretrained=pretrained)
        self.branch1 = se_resnext50_32x4d(num_classes=out_ch, pretrained=None)
        self.branch2 = se_resnext50_32x4d(num_classes=out_ch, pretrained=None)
        #
        self.branch1 = self.model_init(self.branch1)
        self.branch2 = self.model_init(self.branch2)

    def model_init(self, model):
        pretrained_dict = torch.load(
            '/data1/users/lvliang/project_123/ClassHyPer-master/ClassHyPer-master/examples/save/se_resnext50_32x4d-a260b3a4.pth')
        my_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in my_dict}
        my_dict.update(pretrained_dict)
        model.load_state_dict(my_dict)
        return model

    def forward(self, data, step=1):
        # if not self.training:
        #     pred1 = self.branch1(data)
        #     return pred1

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)

if __name__ == '__main__':
    model = FCNs_CPS(in_ch=3, out_ch=1)
    print(model)




