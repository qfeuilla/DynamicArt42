import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import imutils
from loss_network import Vgg16
import os
import torchfile 

def load_rgbimg(filename, size=None, scale=None):
    img = Image.open(filename)
    img = np.array(img)
    if size is not None:
        img = imutils.resize(img, width=size, height=size)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def preprocess(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch

def subtract_imagenet_mean(batch):
    tensortype = type(batch.data)
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean).to(torch.device("cuda")))
    return batch

def gram(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def save_rgbimage(tensor, filename):
    img = tensor.clone().cpu().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)

def save_bgrimage(tensor, filename):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    save_rgbimage(tensor, filename)

def get_vgg_dict(vgg_path):
    load = torch.load(vgg_path)
    keys = list(load.keys())[26:]
    for k in keys:
        del load[k]
    keys = list(load.keys())
    new = ["conv1_1.weight", "conv1_1.bias", "conv1_2.weight", "conv1_2.bias", "conv2_1.weight", "conv2_1.bias", "conv2_2.weight", "conv2_2.bias", "conv3_1.weight", "conv3_1.bias", "conv3_2.weight", "conv3_2.bias", "conv3_3.weight", "conv3_3.bias", "conv4_1.weight", "conv4_1.bias", "conv4_2.weight", "conv4_2.bias", "conv4_3.weight", "conv4_3.bias", "conv5_1.weight", "conv5_1.bias", "conv5_2.weight", "conv5_2.bias", "conv5_3.weight", "conv5_3.bias"]
    for n, k in zip(new, keys):
        load[n] = load[k]
        del load[k]
    return load