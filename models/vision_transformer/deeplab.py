import math
import torch.nn.functional as F
import torch
import torch.nn as nn


class ASPP_module(nn.Module):
    def __init__(self, in_channels, out_channels,temp_epo=0):
        super(ASPP_module, self).__init__()
        # 定义空洞率，根据图示空洞率为1 6 12 18 ，      说明：当空洞率为1时为普通卷积
        #动态膨胀率
        if (temp_epo+1)%10==1 or (temp_epo+1)%10==3 or (temp_epo+1)%10==5 or (temp_epo+1)%10==7:
           dilations = [1, 1, 2, 3]

        if (temp_epo+1)%10==2 or (temp_epo+1)%10==4 or (temp_epo+1)%10==6:
           dilations = [1, 1, 3, 5]
           
        if (temp_epo+1)%10==8 or (temp_epo+1)%10==9:
           dilations = [1, 3, 6, 9]
           
        if (temp_epo+1)%10==0:
           dilations = [1, 6, 12, 18]
        
           
      
#        print("**************使用ASPP对attn进行加强特征提取,膨胀率设置为:",dilations,".**************")
        self.Aspp1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0,
                      dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.Aspp2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                      # padding与dilation相同原因：当原始卷积核为3x3时，使输入输出尺寸大小相同，计算见3中说明。
                      padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Aspp3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                      padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.Aspp4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                      padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            # 输入通道2048，原因：deeplab v3+ 在进去ASPP前为2048(需要改)
            nn.Conv2d(6, 6, kernel_size=(1, 1), stride=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU()
        )
        # concat后通道为1280，用1x1卷积改变通道数60*5=300
        self.conv1 = nn.Conv2d(30, 6, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        # 初始化卷积核权重
        self._init_weight()

    def forward(self, x):
 #       print(x.shape)


        x1 = self.Aspp1(x);
 #       print("X1.shape", x1.size())
        x2 = self.Aspp2(x);
  #      print("X2.shape", x2.size())
        x3 = self.Aspp3(x);
 #       print("X3.shape", x3.size())
        x4 = self.Aspp4(x);
  #      print("X4.shape", x4.size())
        x5 = self.global_avg_pool(x);
  #      print('x5.shape', x5.size())
        # 利用双线性插值恢复x5的尺寸，再进行concat
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        cat = torch.cat((x1, x2, x3, x4, x5), dim=1);
    #    print('cat.shape', cat.size())

        # 此处的output，包含后面1x1卷积进行的通道数调整
        output = self.conv1(cat)
        output = self.bn1(output);
    #    print('output.shape', output.size())
        return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 初始化卷积核方式
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# if __name__ == '__main__':
#     # 测试输出尺寸使用（2048为Decoder输出的通道数）
#     # aspp = ASPP_module(2048, 256)
#     # input = torch.randn(2, 2048, 176, 240);  #batchsize=2，2048通道，176*240
#
#     aspp = ASPP_module(6, 6)
#     input = torch.randn(4, 6, 1601, 1601);  #batchsize=2，2048通道，176*240
#     print('input_size:', input.size())
