from PIL import Image

import torch
import numpy as np
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from skimage.transform import resize

style_img = Image.open("starry_night.jpg").convert('RGB').resize((300, 300), Image.ANTIALIAS)
style_img = torch.tensor(np.transpose(np.array(style_img), (2, 0, 1))).unsqueeze(0)
import torchvision.models as models
cnn = models.vgg19(pretrained=True).features

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        self.weight = weight

    def forward(self, input):
        self.loss = F.mse_loss(input * self.weight, self.target)
        return input.clone()

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
    
class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight

    def forward(self, input):
        self.G = gram_matrix(input)
        self.G.mul_(self.weight)
        self.loss = F.mse_loss(self.G, self.target)
        return input.clone()

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class Predictor:
    def __init__(self, path='trained_model_dict'):
        self.content_losses = []
        self.style_losses = []
        self.model = nn.Sequential()
        i = 1
        for layer in list(cnn):
                if isinstance(layer, nn.Conv2d):
                    name = "conv_" + str(i)
                    self.model.add_module(name, layer)

                    if name in content_layers:
                        target = self.model(content_img).clone()
                        content_loss = ContentLoss(target, 1)
                        self.model.add_module("content_loss_" + str(i), content_loss)
                        self.content_losses.append(content_loss)

                    if name in style_layers:
                        # add style loss:
                        target_feature = self.model(style_img).clone()
                        target_feature_gram = gram_matrix(target_feature)
                        style_loss = StyleLoss(target_feature_gram, 1000)
                        self.model.add_module("style_loss_" + str(i), style_loss)
                        self.style_losses.append(style_loss)

                if isinstance(layer, nn.ReLU):
                    name = "relu_" + str(i)
                    self.model.add_module(name, layer)

                    if name in content_layers:
                        # add content loss:
                        target = self.model(content_img).clone()
                        content_loss = ContentLoss(target, 1)
                        self.model.add_module("content_loss_" + str(i), content_loss)
                        self.content_losses.append(content_loss)

                    if name in style_layers:
                        # add style loss:
                        target_feature = self.model(style_img).clone()
                        target_feature_gram = gram_matrix(target_feature)
                        style_loss = StyleLoss(target_feature_gram, 1000)
                        self.model.add_module("style_loss_" + str(i), style_loss)
                        self.style_losses.append(style_loss)

                    i += 1

                if isinstance(layer, nn.MaxPool2d):
                    name = "pool_" + str(i)
                    self.model.add_module(name, layer)
                    self.model.load_state_dict(torch.load(path))

    def get_image_predict(self, img_path='img_path.jpg'):
        optimizer = torch.optim.LBFGS([input_image])
        image = Image.open(img_path).convert('RGB').resize((300, 300), Image.ANTIALIAS)
        img_tensor = torch.tensor(np.transpose(np.array(image), (2, 0, 1))).unsqueeze(0)
        print('Before normalize', torch.max(img_tensor))
        input_image = img_tensor / 255.
        for i in range(300):
          # correct the values of updated input image
          input_image.data.clamp_(0, 1)
          self.model(input_image)
          style_score = 0
          content_score = 0
          for sl in self.style_losses:
              style_score += sl.backward()
          for cl in self.content_losses:
              content_score += cl.backward()

          loss = style_score + content_score
          
          optimizer.step(lambda:loss)
          optimizer.zero_grad()
        return input_image.cpu().data.numpy()[0].transpose(1, 2, 0)
