import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from auto_encoder import VariationalAutoEncoder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import os

# initialize parser
parser = argparse.ArgumentParser(description="This parser contains the starting_epoch and learning rate of training")

# adding the arguments (python src/model/train.py --starting_epoch 0 --learning_rate 3e-4)
parser.add_argument('--starting_epoch', type=int, required=True, help='epoch we are starting training from,\
                    this is for naming the files that are going to be saved')
parser.add_argument('--learning_rate', type=float, required=True, help='the learning rate used for training the model')
parser.add_argument('--device', type=int, required=True, help='in which device the model is going to be trained on')
args = parser.parse_args()

# Initialize some constants
torch.cuda.set_device(args.device) # set the training to be done on device 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM_X = 272 # input image is going to be resized to this size
INPUT_DIM_Y = 160
INIT_DIM = 8
LATENT_DIMS = 6
NUM_EPOCHS = 500
BATCH_SIZE = 1
LR_RATE = args.learning_rate # original is 3e-4
KERNEL_SIZE = 4
# STARTING_EPOCH = args.starting_epoch # its from where you last stopped, just for naming the model files

# loading the dataset
data_path = 'ocean/data/dataset/psi2/train_set' # setting path
# sequence of transformations to be done
transform = transforms.Compose([transforms.Resize((INPUT_DIM_X, INPUT_DIM_Y)),   # sequence of transformations to be done
                                transforms.Grayscale(num_output_channels=1), # on each image (resize, greyscale,
                                transforms.ToTensor()])                      # convert to tensor)

dataset = datasets.ImageFolder(root=data_path, transform=transform) # read data from folder

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True) # create dataloader object

if type(LATENT_DIMS) != list:
    LATENT_DIMS = [LATENT_DIMS]
else:
    pass

# if starting epoch is not 0, load from last trained model
for LATENT_DIM in LATENT_DIMS:

    os.mkdir(f"ocean/models/VAE_{LATENT_DIM}")

    model = VariationalAutoEncoder(init_dim=INIT_DIM, latent_dim=LATENT_DIM, kernel_size=KERNEL_SIZE).to(DEVICE)


    # defining adam optimizer and mean squared error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE) # defining optimizer
    loss_fn = nn.BCELoss(reduction='sum') # define loss function


    print(f"latent dimension {LATENT_DIM}")
    # training
    avg_losses = torch.tensor(data=[]).to(DEVICE)
    for epoch in range(NUM_EPOCHS + 1):
        loop = tqdm(enumerate(train_loader))
        print(f'Epoch: {epoch}')
        losses = torch.tensor(data=[])
        losses = losses.to(DEVICE)
        for i, (x, _) in loop:
            x = x.to(DEVICE)

            y = model(x)[0]
            # print(y)
            # add loss to list of losses in this epoch to later calculate its mean
            loss = loss_fn(y, x)
            loss = loss.view((1)).to(DEVICE)
            losses = torch.cat((losses, loss), -1)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        # calculate the mean of the losses
        avg_loss = losses.mean().view((1)).to(DEVICE)
        avg_losses = torch.cat((avg_losses,avg_loss), -1)
        print(f'Loss: {avg_loss}')
        
        # from 50 to 50 epochs save the current model state
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f'ocean/models/VAE_{LATENT_DIM}/model_vae_ld_{LATENT_DIM}_{epoch}')
            torch.save(avg_losses, f'ocean/models/VAE_{LATENT_DIM}/loss_vae_ld_{LATENT_DIM}_{epoch}.pt')

    # save model at end of training
    torch.save(model.state_dict(), f'ocean/models/VAE_{LATENT_DIM}/model_vae_ld_{LATENT_DIM}_end')