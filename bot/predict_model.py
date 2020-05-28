from PIL import Image

import torch
import numpy as np
import torchvision
import torch.nn as nn

style_img = image_loader("starry_night.jpg").type(dtype)

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
        self.model = self.model = nn.Sequential(conv_1= Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), 
                                                style_loss_1= StyleLoss(), 
                                                relu_1= ReLU(inplace=True), 
                                                conv_2= Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                style_loss_2= StyleLoss(),
                                                relu_2= ReLU(inplace=True),
                                                pool_3= MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                                                conv_3= Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                style_loss_3= StyleLoss(),
                                                relu_3= ReLU(inplace=True),
                                                conv_4= Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                content_loss_4= ContentLoss(),
                                                style_loss_4= StyleLoss(),
                                                relu_4= ReLU(inplace=True),
                                                pool_5= MaxPool2d(kernel_si,ze=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                                                conv_5= Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                style_loss_5= StyleLoss(),
                                                relu_5= ReLU(inplace=True),
                                                conv_6= Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_6= ReLU(inplace=True),
                                                conv_7= Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_7= ReLU(inplace=True),
                                                conv_8= Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_8= ReLU(inplace=True),
                                                pool_9= MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                                                conv_9= Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_9= ReLU(inplace=True),
                                                conv_10= Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_10= ReLU(inplace=True),
                                                conv_11= Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_11= ReLU(inplace=True),
                                                conv_12= Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_12= ReLU(inplace=True),
                                                pool_13= MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                                                conv_13= Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_13= ReLU(inplace=True),
                                                conv_14= Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_14= ReLU(inplace=True),
                                                conv_15= Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_15= ReLU(inplace=True),
                                                conv_16= Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                                relu_16= ReLU(inplace=True),
                                                pool_17= MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)))

        self.model.load_state_dict(torch.load(path))

    def get_image_predict(self, img_path='img_path.jpg'):
        style_losses = [StyleLoss(), StyleLoss(), StyleLoss(), StyleLoss(), StyleLoss()]
        optimizer = torch.optim.LBFGS([input_image])
        image = Image.open(img_path).convert('RGB').resize((300, 300), Image.ANTIALIAS)
        img_tensor = torch.tensor(np.transpose(np.array(image), (2, 0, 1))).unsqueeze(0)
        print('Before normalize', torch.max(img_tensor))
        input_image = img_tensor / 255.
        for i in range(300):
          # correct the values of updated input image
          input_image.data.clamp_(0, 1)
          model(input_image)
          style_score = 0
          content_score = 0
          for sl in style_losses:
              style_score += sl.backward()
          for cl in content_losses:
              content_score += cl.backward()

          loss = style_score + content_score
          
          optimizer.step(lambda:loss)
          optimizer.zero_grad()
        return input_image.cpu().data.numpy()[0].transpose(1, 2, 0)
