import torch
import torch.optim as optim
import os

import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchvision import datasets, transforms

from style_network import *
from loss_network import *
import numpy as np
import utilFunc as utils
from torch.utils.data import DataLoader
from torchvision import datasets

import cv2 

class Transfer:
    def __init__(self, epoch, data_path, style_path, batch, lr, spatial_a, spatial_b, img_size=256):
        self.epoch = epoch
        self.data_path = data_path
        self.style_path = style_path
        self.lr = lr
        self.batch = batch

        self.s_a = spatial_a
        self.s_b = spatial_b

        self.transform = transforms.Compose([transforms.Scale(img_size),
                                    transforms.CenterCrop(img_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])

        self.style_net = TransformerNet()
        self.loss_net = Vgg16()

        self.img_size = img_size                                                                  

    def train(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.style_net = self.style_net.to(torch.device("cuda"))
        self.loss_net = self.loss_net.to(torch.device("cuda"))
        train_dataset = datasets.ImageFolder(self.data_path, self.transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch)

        adam = optim.Adam(self.style_net.parameters(), lr=self.lr)
        mse_loss = torch.nn.MSELoss()

        style = utils.load_rgbimg(self.style_path, size=self.img_size)
        style = style.repeat(self.batch, 1, 1, 1)
        style = style.to(torch.device("cuda"))

        style_v = utils.normalize(style)
        style_vgg_loss = self.loss_net(style_v)
        grams_style = [utils.gram(y) for y in style_vgg_loss]

        print('Training Start!!')
        for count in range(self.epoch):
            self.style_net.train()
            # frames = videos[np.random.randint(0, self.batch)]
            for imid, (x, _) in enumerate(train_loader):
                adam.zero_grad()
                nbatch = len(x)
                
                x = x.to(torch.device("cuda"))
                y = self.style_net(x)


                y = utils.normalize(y)
                x = utils.normalize(x)

                x_vgg_loss = self.loss_net(x)
                y_vgg_loss = self.loss_net(y)

                content_loss = mse_loss(y_vgg_loss[1], x_vgg_loss[1])

                # print(content_loss);
                style_loss = 0
                for a in range(len(y_vgg_loss)):
                    gram_s = grams_style[a]
                    gram_y = utils.gram(y_vgg_loss[a])
                    style_loss += mse_loss(gram_y, gram_s[:nbatch, :, :])
                
                spatial_loss = self.s_a * content_loss + self.s_b * style_loss
                Loss = spatial_loss # + self.t_l * temporal_loss
                Loss.backward()
                adam.step()

                del x
                torch.cuda.empty_cache()
                if (imid % (len(train_loader) // 100) == 0):
                    self.predict("./test/land.jpg", count)
                    self.style_net.eval()
                    name = "./model/net_save_epochs_" + str(count) + ".pth"
                    torch.save(self.style_net.state_dict(), name)
                print(str(imid) + " Loss :" + str(Loss))
            print("epochs :" + str(count) + " loss :" + str(Loss))
            plt.close()
            

    def predict(self, img_path, step=0): # add a int to choose wich train  network, after testing part
        self.style_net.eval()
        content_image = utils.load_rgbimg(img_path, size=1000)
        print(content_image.shape)
        content_image = content_image.unsqueeze(0)
        content_image = content_image.to(torch.device("cuda"))

        output = self.style_net(content_image)
        utils.save_bgrimage(content_image.data[0], './test/preprocess_test.jpg')
        utils.save_bgrimage(output.data[0], '.'.join(img_path.split(".")[:-1])+"_stylized"+str(step)+'.'+img_path.split(".")[-1])
    
    def load_weight(self, weight_path):
        self.style_net = self.style_net.to(torch.device("cuda"))
        self.loss_net = self.loss_net.to(torch.device("cuda"))
        self.style_net.load_state_dict(torch.load(weight_path))
        self.style_net.eval()