import numpy as np

import torch
import torch.nn as nn

import time


class Deconn(nn.Module):

    def __init__(self):
        super(Deconn, self).__init__()

        self.initialize_model()

    def initialize_model(self):

        self.ch = 4
        self.conv0 = nn.Conv2d(1, self.ch, 3, padding=1,\
                padding_mode="reflect") 
        self.conv1 = nn.Conv2d(self.ch, self.ch*2, 3, padding=1,\
                padding_mode="reflect") 
        self.conv2 = nn.Conv2d(self.ch*2, self.ch*4, 3, padding=1,\
                padding_mode="reflect") 
        self.conv3 = nn.Conv2d(self.ch*4, self.ch*2, 3, padding=1,\
                padding_mode="reflect") 
        self.conv4 = nn.Conv2d(self.ch*2, self.ch*2, 3, padding=1,\
                padding_mode="reflect") 

        self.conv_t5 = nn.ConvTranspose2d(self.ch*2, self.ch*2, 2,\
                stride=2, padding=0)
        self.conv_t6 = nn.ConvTranspose2d(self.ch*4, self.ch*2, 2,\
                stride=2, padding=0)
        self.conv_t7 = nn.ConvTranspose2d(self.ch*6, self.ch*2, 2,\
                stride=2, padding=0)
        self.conv_t8 = nn.ConvTranspose2d(self.ch*4, self.ch*2, 2,\
                stride=2, padding=0)
        self.conv_t9 = nn.ConvTranspose2d(self.ch*3, self.ch*2, 2,\
                stride=2, padding=0)

        self.conv_t10 = nn.ConvTranspose2d(self.ch*2+1, self.ch*2, 2,\
                stride=2, padding=0)

        self.conv11 = nn.Conv2d(self.ch*2, self.ch*2, 7, padding=3,\
                padding_mode="reflect") 
        self.conv12 = nn.Conv2d(self.ch*2, self.ch*2, 7, padding=3,\
                padding_mode="reflect") 
        self.conv13 = nn.Conv2d(self.ch*2, self.ch*2, 7, padding=3,\
                padding_mode="reflect") 
        self.conv_t14 = nn.ConvTranspose2d(self.ch*2, self.ch*2, 2, stride=2,\
                padding=0) 

        self.conv15 = nn.Conv2d(self.ch*4, 1, 3, padding=1, padding_mode="reflect") 

    def forward(self, x):
        """
                                                                                    2h x 2w -> conv10
        h x w -> conv0 ----------------------------------------------> cat h x w -> convT9 /      \ h x w -> conv11 --------------> cat  h x w -> conv14 -> y (h x w 
            \ h/2 x w/2 -> conv1 -------------------------------> cat h/2 x w/2 -> convT8 /           \ h/2 x w/2 -> conv12 --> convT13 /
                \ h/4 x w/4 -> conv2 ----------------------> cat h/4 x w/8 -> convT7 / 
                    \ h/8 x w/8 -> conv3 -----------> cat h/8 x w/8 -> convT6 /
                        \ h/16 x w/16 -> conv4  -> convT5 /
        """


        layer_0 = torch.max_pool2d(torch.arctan(self.conv0(x)), 2) #128
        layer_1 = torch.max_pool2d(torch.arctan(self.conv1(layer_0)), 2) #64
        layer_2 = torch.max_pool2d(torch.arctan(self.conv2(layer_1)), 2) #32
        layer_3 = torch.max_pool2d(torch.arctan(self.conv3(layer_2)), 2) #16
        layer_4 = torch.max_pool2d(torch.arctan(self.conv4(layer_3)), 2) #8

        layer_5 = torch.arctan(self.conv_t5(layer_4))

        
        layer_6a = torch.cat([layer_3, layer_5], 1) 
        layer_6b = torch.arctan(self.conv_t6(layer_6a))
        
        layer_7a = torch.cat([layer_2, layer_6b], 1) 
        layer_7b = torch.arctan(self.conv_t7(layer_7a))

        layer_8a = torch.cat([layer_1, layer_7b], 1) 
        layer_8b = torch.arctan(self.conv_t8(layer_8a))

        layer_9a = torch.cat([layer_0, layer_8b], 1) 
        layer_9b = torch.arctan(self.conv_t9(layer_9a))

        layer_10a = torch.cat([x, layer_9b], 1)
        layer_10b = torch.arctan(self.conv_t10(layer_10a))

        layer_11 = torch.max_pool2d(torch.arctan(self.conv11(layer_10b)), 2)
        layer_12 = torch.max_pool2d(torch.arctan(self.conv12(layer_11)), 2)
        layer_13 = torch.arctan(self.conv13(layer_12))

        layer_14a = torch.arctan(self.conv_t14(layer_13))
        layer_14b = torch.cat([layer_11, layer_14a], 1)

        layer_15 = torch.sigmoid(self.conv15(layer_14b))

        return layer_15, layer_10b


if __name__ == "__main__":

    x = torch.randn(1, 1, 128, 128)

    model = Deconn()

    y = model(x)

    xx = torch.randn(16, 1, 128, 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for step in range(100):

        pred, decon = model(xx)

        loss = torch.mean(torch.abs(pred-xx)**2) + torch.mean(torch.abs(decon))

        loss.backward()
        optimizer.step()

        print("loss at step {} = {:.3e}".format(step, loss))

 
        



