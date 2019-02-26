import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchvision
import torch
#import tqdm
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy


################################################################################
#PLOTTING WEIGHT IMAGES

def plot_filters(filters):
    fig, axs = plt.subplots(6, 2,figsize=(40,20))
    weight_images = []

    #iterating through the rows and columns of the 5x4 figure
    for i in range(6):
        for j in range(2):
            if i == 5 and j == 1:
                weight_images.append(axs[i,j].imshow(image))
            else:
                weight_images.append(axs[i,j].imshow(filters[i*2+j].detach()))

    plt.tight_layout()
    plt.show()

def plot_weights(weights):
    num_filters = 5
    fig, axs = plt.subplots(num_filters, 3,figsize=(70,20))
    weight_images = []
    for i in range(num_filters):
        for j in range(3):
            weight_images.append(axs[i,j].imshow(weights[i+6][j].detach()))

    vmin = -1
    vmax = 1


    #norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    #cbaxes = fig.add_axes([0.8,0.1,0.03,0.8])
    #fig.colorbar(weight_images[num_filters*3-2], ax=axs, orientation='horizontal', fraction=.03,pad = 0.05)
    #plt.colorbar(axs[0],cax=cbaxes)

    #for im in weight_images:
    #    im.set_cmap(weight_images[2].get_cmap())
    #    im.set_clim(weight_images[2].get_clim())
    #iterating through the rows and columns of the 5x4 figure

    #plt.tight_layout()
    plt.show()



class Model_first(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained = True).conv1


                               # layers
    def forward ( self , x):
        x = self.model(x)
        return x


class Model_last(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(*list(torchvision.models.resnet18(pretrained = True).children())[:-2])



                               # layers
    def forward ( self , x):
        x = self.model(x)
        return x



class Trainer:

    def __init__(self):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.early_stop_count = 4

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        self.model = Model()
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)



        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.learning_rate, (0.99,0.999), eps=1e-1,weight_decay=1e-4)
        #self.optimizer = torch.optim.ASGD(self.model.parameters(), self.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 2, gamma=0.1, last_epoch=-1)
        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)

        self.validation_check = len(self.dataloader_train) // 2

        # Tracking variables
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC.append(validation_acc)
        self.VALIDATION_LOSS.append(validation_loss)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC.append(test_acc)
        self.TEST_LOSS.append(test_loss)

        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        self.validation_epoch()
        for epoch in range(self.epochs):
            #updating learing rate
            #self.scheduler.step()
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)



                # Perform the forward pass
                predictions = self.model((X_batch))
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()
                # Reset all computed gradients to 0
                self.optimizer.zero_grad()

                 # Compute loss/accuracy for all three datasets.
                if batch_it % self.validation_check == 0 and epoch != 0:
                    self.validation_epoch()
                    # Check early stopping criteria.
                    if self.should_early_stop():
                        print("Early stopping.")
                        return
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

if __name__ == "__main__":

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    image_path = "task_3_data/"
    transform = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ]
    transform = torchvision.transforms.Compose(transform)

    dataset = torchvision.datasets.ImageFolder(image_path,transform = transform)
    dataloader = torch.utils.data.DataLoader(dataset)



    image = next(iter(dataloader))[0][0]


    image = image.unsqueeze(0)

    #model = Model_first()

    model = Model_last()
    image = nn.functional.interpolate(image,256)
    activation = model(image)
    filters_to_visualize = activation.view(activation.shape[1],*activation.shape[2:])[:11]
    image = image[0].numpy().transpose(1,2,0)

    for m in model.modules():
        if type(m) == nn.Conv2d:
            weights_to_visualize = m.weight.data
            plot_weights(weights_to_visualize)
            break
