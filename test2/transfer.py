import torch
import torch.optim as optim

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
    def __init__(self, epoch, data_path, style_path, lr, spatial_a, spatial_b, img_size=600):
        self.epoch = epoch
        self.data_path = data_path
        self.style_path = style_path
        self.lr = lr
        self.batch = 1

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
        self.style_net = self.style_net.to(torch.device("cuda"))
        self.loss_net = self.loss_net.to(torch.device("cuda"))
        
        train_dataset = datasets.ImageFolder(self.data_path, self.transform)
        kwargs = {'num_workers': 0, 'pin_memory': False}
        train_loader = DataLoader(train_dataset, batch_size=1, **kwargs)

        adam = optim.Adam(self.style_net.parameters(), lr=self.lr)
        mse_loss = torch.nn.MSELoss()

        style = utils.load_rgbimg(self.style_path, size=self.img_size)
        style = style.repeat(self.batch, 1, 1, 1)
        style = utils.preprocess(style)
        style = style.to(torch.device("cuda"))

        style_v = Variable(style, volatile=True)
        style_v = utils.subtract_imagenet_mean(style_v)
        style_vgg_loss = self.loss_net(style_v)
        grams_style = [utils.gram(y) for y in style_vgg_loss]

        print('Training Start!!')
        for count in range(self.epoch):
            self.style_net.train()
            # frames = videos[np.random.randint(0, self.batch)]
            for imid, (x, _) in enumerate(train_loader):
                adam.zero_grad()
                x = Variable(utils.preprocess(x)).to(torch.device("cuda"))

                y = self.style_net(x)

                y = utils.subtract_imagenet_mean(y)
                x = utils.subtract_imagenet_mean(x)

                x_vgg_loss = self.loss_net(x)

                y_vgg_loss = self.loss_net(y)

                content_loss = mse_loss(y_vgg_loss[1], x_vgg_loss[1])

                # print(content_loss);
                style_loss = 0
                for a in range(len(y_vgg_loss)):
                    gram_s = Variable(grams_style[a].data, requires_grad=False)
                    gram_y = utils.gram(y_vgg_loss[a])
                    style_loss += mse_loss(gram_y, gram_s)
                
                # print(style_loss)
                
                spatial_loss = self.s_a * content_loss + self.s_b * style_loss
                Loss = spatial_loss # + self.t_l * temporal_loss
                Loss.backward()
                adam.step()

                del x
                torch.cuda.empty_cache()
                if (imid % (len(train_loader) // 100) == 0):
                    self.predict("./test/land.jpg", count)
                print(str(imid) + " Loss :" + str(Loss))
            torch.save(self.style_net.state_dict(), "./model/style_net_save_epochs:_" + str(count) + "_Loss:_" + str(Loss) + ".pth")
            print("epochs :" + str(count) + " loss :" + str(Loss))
            plt.close()
            

    def predict(self, img_path, step=0): # add a int to choose wich train  network, after testing part
        content_image = utils.load_rgbimg(img_path, size=self.img_size)
        print(content_image.shape)
        content_image = content_image.repeat(1, 1, 1, 1)
        content_image = utils.preprocess(content_image)
        content_image = content_image.to(torch.device("cuda"))
        output = self.style_net(content_image)
        utils.save_bgrimage(content_image.data[0], './test/preprocess_test.jpg')
        utils.save_bgrimage(output.data[0], '.'.join(img_path.split(".")[:-1])+"_stylized"+str(step)+'.'+img_path.split(".")[-1])