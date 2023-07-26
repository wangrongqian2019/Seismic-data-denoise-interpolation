import torch.nn as nn 
import torch

class CBDNet(nn.Module):
    def __init__(self):
        super(CBDNet, self).__init__()
        self.fcn = FCN()
        self.rdn = rdn()
        self._initialize_weights()
    
    def forward(self, x):
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.rdn(concat_img) + x
        return noise_level, out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

                
class one_conv(nn.Module):
    def __init__(self,inchanels,growth_rate,kernel_size = 3):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(inchanels,growth_rate,kernel_size=kernel_size,padding = kernel_size>>1,stride= 1)
        self.relu = nn.ReLU()
    def forward(self,x):
        output = self.relu(self.conv(x))
        return torch.cat((x,output),1)

class RDB(nn.Module):
    def __init__(self,G0,C,G,kernel_size = 3):
        super(RDB,self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G,G))
        self.conv = nn.Sequential(*convs)
        #local_feature_fusion
        self.LFF = nn.Conv2d(G0+C*G,G0,kernel_size = 1,padding = 0,stride =1)
    def forward(self,x):
        out = self.conv(x)
        lff = self.LFF(out)
        #local residual learning
        return lff + x

class rdn(nn.Module):
    def __init__(self):
        '''
        opts: the system para
        '''
        super(rdn,self).__init__()
        '''
        D: RDB number 20
        C: the number of conv layer in RDB 6
        G: the growth rate 32
        G0:local and global feature fusion layers 64filter
        '''
        self.D = 20
        self.C = 6
        self.G = 32
        self.G0 = 64
        
        kernel_size = 3
        input_channels = 2
        #shallow feature extraction 
        self.SFE1 = nn.Conv2d(input_channels,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride=  1)
        self.SFE2 = nn.Conv2d(self.G0,self.G0,kernel_size=kernel_size,padding = kernel_size>>1,stride =1)
        #RDB for paper we have D RDB block
        self.RDBS = nn.ModuleList()
        for d in range(self.D):
            self.RDBS.append(RDB(self.G0,self.C,self.G,kernel_size))
        #Global feature fusion
        self.GFF = nn.Sequential(
               nn.Conv2d(self.D*self.G0,self.G0,kernel_size = 1,padding = 0 ,stride= 1),
               nn.Conv2d(self.G0,self.G0,kernel_size,padding = kernel_size>>1,stride = 1),
        )
        self.up_net = nn.Sequential(
            nn.Conv2d(self.G0,1,kernel_size=kernel_size,padding = kernel_size>>1,stride = 1),
        )
        
        '''
        for para in self.modules():
            if isinstance(para,nn.Conv2d):
                nn.init.orthogonal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()
        '''


    def forward(self,x):
        #f-1
        f__1 = self.SFE1(x)
        out  = self.SFE2(f__1)
        RDB_outs = []
        for i in range(self.D):
            out = self.RDBS[i](out)
            RDB_outs.append(out)
        out = torch.cat(RDB_outs,1)
        out = self.GFF(out)
        out = f__1+out
        out = self.up_net(out)
        return out


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.outc = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        conv1 = self.inc(x)
        conv2 = self.conv(conv1)
        conv3 = self.conv(conv2)
        conv4 = self.conv(conv3)
        conv5 = self.outc(conv4)
        return conv5


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        h_x = est_noise.shape[2]
        w_x = est_noise.shape[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = torch.mean(torch.pow((out_image - gt_image), 2)) + 0.75 * torch.mean(torch.pow((est_noise - gt_noise), 2))
                #if_asym * 0.75 * torch.mean(torch.mul(torch.abs(0.3 - F.relu(gt_noise - est_noise)), torch.pow(est_noise - gt_noise, 2)))
        return loss

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]