import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from model import SuperResolution
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse

# initialize parser
parser = argparse.ArgumentParser(description="This parser contains the starting_epoch and learning rate of training")

# adding the arguments (python src/model/train.py --starting_epoch 0 --learning_rate 3e-4)
parser.add_argument('--starting_epoch', type=int, required=True, help='epoch we are starting training from,\
                    this is for naming the files that are going to be saved')
parser.add_argument('--learning_rate', type=float, required=True, help='the learning rate used for training the model')
args = parser.parse_args()

# Initialize some constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 1000 # input image is going to be resized to this size
DOWNGRADE_FACTOR = 10 # by how much the image resolution is going to be reduced (downgrade_factor = 10 ==> img.shape /= 10)
NUM_EPOCHS = 200
BATCH_SIZE = 1
LR_RATE = args.learning_rate # original is 3e-4
STARTING_EPOCH = args.starting_epoch # its from where you last stopped, just for naming the model files

# loading the dataset
data_path = 'data/dataset/train_set' # setting path
# sequence of transformations to be done
transform = transforms.Compose([transforms.Resize((INPUT_DIM, INPUT_DIM)),   # sequence of transformations to be done
                                transforms.Grayscale(num_output_channels=1), # on each image (resize, greyscale,
                                transforms.ToTensor()])                      # convert to tensor)

dataset = datasets.ImageFolder(root=data_path, transform=transform) # read data from folder

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True) # create dataloader object

# if starting epoch is not 0, load from last trained model
model = SuperResolution().to(DEVICE)
if STARTING_EPOCH != 0:
    model.load_state_dict(torch.load(f'models/model_{STARTING_EPOCH}'))

# defining adam optimizer and mean squared error loss
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE) # defining optimizer
loss_fn = nn.MSELoss(reduction='sum') # define loss function

# training
avg_losses = torch.tensor(data=[]).to(DEVICE)
for epoch in range(STARTING_EPOCH + 1, NUM_EPOCHS + 1):
    loop = tqdm(enumerate(train_loader))
    print(f'Epoch: {epoch}')
    losses = torch.tensor(data=[])
    losses = losses.to(DEVICE)
    for i, (x, _) in loop:
        x = x.to(DEVICE)
        # downgrade image (reshape to smaller shape then brings it back to original shape)
        lr_x = F.interpolate(x, size=(INPUT_DIM//DOWNGRADE_FACTOR, INPUT_DIM//DOWNGRADE_FACTOR), mode='bilinear', align_corners=False)
        lr_x = F.interpolate(lr_x, size=(INPUT_DIM, INPUT_DIM), mode='bilinear', align_corners=False)
        # forward pass
        y = model(lr_x)
        
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
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'models/model_{epoch}')
        torch.save(avg_losses, f'models/loss_{epoch}.pt')

# save model at end of training
torch.save(model.state_dict(), 'models/model')