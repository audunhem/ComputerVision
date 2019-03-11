import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision
import torch
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy


################################################################################
#PLOTTING IMAGES

def plot_filters(filters,original_image):
    fig, axs = plt.subplots(6, 2,figsize=(20,10))
    weight_images = []

    #iterating through the rows and columns of the 5x4 figure
    for i in range(6):
        for j in range(2):
            if i == 5 and j == 1:
                weight_images.append(axs[i,j].imshow(original_image))
            else:
                weight_images.append(axs[i,j].imshow(filters[i*2+j].detach()))

    plt.tight_layout()
    plt.show()

def plot_weights(weights):
    num_filters = 6
    fig, axs = plt.subplots(num_filters, 3,figsize=(20,10))
    weight_images = []
    for i in range(num_filters):
        for j in range(3):
            weight_images.append(axs[i,j].imshow(weights[i][j].detach()))

    vmin = -1
    vmax = 1


    #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    #cbaxes = fig.add_axes([0.8,0.1,0.03,0.8])
    fig.colorbar(weight_images[num_filters*3-2], ax=axs, orientation='horizontal', fraction=.03,pad = 0.05)
    #plt.colorbar(axs[0],cax=cbaxes)

    for im in weight_images:
        im.set_cmap(weight_images[2].get_cmap())
        im.set_clim(weight_images[2].get_clim())
        #iterating through the rows and columns of the 5x4 figure to normalize values
    plt.show()

################################################################################
#MODEL DEFINITIONS

class Model_first(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained = True).conv1
    def forward ( self , x):
        x = self.model(x)
        return x

class Model_last(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(*list(torchvision.models.resnet18(pretrained = True).children())[:-2])
    def forward ( self , x):
        x = self.model(x)
        return x


if __name__ == "__main__":

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    image_path = "task_3_data/"

    #defining transforms to be performed on the image
    transform_elements = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ]
    transform = torchvision.transforms.Compose(transform_elements)

    #loading and transforming data
    dataset = torchvision.datasets.ImageFolder(image_path,transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset)
    image = next(iter(dataloader))[0][0]
    image = image.unsqueeze(0)
    image = nn.functional.interpolate(image,256)

    #choosing model and performing a forward pass
    model = Model_first() #uncomment to look at first convolutional layer activations
    #model = Model_last() #uncomment to look at last convolutional layer activations
    activation = model(image)

    #readying data for plotting
    filters_to_visualize = activation.view(activation.shape[1],*activation.shape[2:])[:11]
    original_image = plt.imread("task_3_data/pics/tree.jpeg")

    for m in model.modules():
        if type(m) == nn.Conv2d:
            weights_to_visualize = m.weight.data
            plot_weights(weights_to_visualize) #uncomment to plot weights
            break

    #plot_filters(filters_to_visualize,original_image) #uncomment to plot filters
