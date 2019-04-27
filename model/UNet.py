import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self, img_chn=1, n_cls=2):
        super(UNet, self).__init__()

        # Encoder
        self.encoder_lyr1 = nn.Sequential(
            nn.Conv2d(img_chn, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))

        self.encoder_lyr2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True))

        self.encoder_lyr3  = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True))

        self.encoder_lyr4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True))
        
        self.encoder_lyr5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU(inplace=True))

        # Decoder
        self.decoder_lyr5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.decoder_lyr4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))

        self.decoder_lyr3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))

        self.decoder_lyr2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))

        self.decoder_lyr1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, n_cls, 1, padding=0))

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encode
        f1 = self.encoder_lyr1(x)
        f2 = self.encoder_lyr2(f1)
        f3 = self.encoder_lyr3(f2)
        f4 = self.encoder_lyr4(f3)
        f5 = self.encoder_lyr5(f4)
        # Decode
        y = self.decoder_lyr5(f5)
        y = torch.cat([f4, y], dim=1)
        y = self.decoder_lyr4(y)
        y = torch.cat([f3, y], dim=1)
        y = self.decoder_lyr3(y)
        y = torch.cat([f2, y], dim=1)
        y = self.decoder_lyr2(y)
        y = torch.cat([f1, y], dim=1)
        y = self.decoder_lyr1(y)

        return y

    def params_count(self):
        total_num = sum([param.nelement() for param in self.parameters()])
        return total_num
    

if __name__=='__main__':

    # 1. fake data
    images = torch.rand(32, 1, 256, 256)

    # 2. build the model
    net = unet()
    print('Number of parameters: {:.2f}M'.format(net.params_count() / 1e6))

    # 3. forward
    prob = net(images)

    # 4. info
    print(prob.shape)
