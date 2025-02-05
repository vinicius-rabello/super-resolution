import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from model import SuperResolution
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description="This parser contains the starting_epoch and learning rate of training")

# Adding command-line arguments (example usage: python train.py --starting_epoch 0 --learning_rate 3e-4)
parser.add_argument('--starting_epoch', type=int, required=True, help='Epoch to start training from. Used for saving model checkpoints.')
parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate used for training the model.')
args = parser.parse_args()

# Set device to GPU if available, otherwise use CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
INPUT_DIM = 256  # Input image size (will be resized to this dimension)
DOWNGRADE_FACTOR = 4  # Factor by which the image resolution is reduced
NUM_EPOCHS = 200  # Total number of training epochs
BATCH_SIZE = 1  # Number of samples per batch
LR_RATE = args.learning_rate  # Learning rate (e.g., 3e-4)
STARTING_EPOCH = args.starting_epoch  # Starting epoch (used for checkpointing)

# Define dataset path
data_path = 'sandstone/data/synthetic/processed/train_set'

# Define a sequence of image transformations
transform = transforms.Compose([
    transforms.Resize((INPUT_DIM, INPUT_DIM)),  # Resize image to uniform dimensions
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.ToTensor()  # Convert image to PyTorch tensor
])

# Load dataset from the specified folder
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Create a DataLoader for batch processing
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model and move it to the appropriate device
model = SuperResolution().to(DEVICE)

# If resuming training from a previous epoch, load the saved model state
if STARTING_EPOCH != 0:
    model.load_state_dict(torch.load(f'models/model_fahad_test_{STARTING_EPOCH}'))

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)  # Adam optimizer
loss_fn = nn.MSELoss(reduction='sum')  # Mean Squared Error (MSE) loss

# Initialize tensor to store average losses
avg_losses = torch.tensor(data=[]).to(DEVICE)

# Training loop
for epoch in range(STARTING_EPOCH + 1, NUM_EPOCHS + 1):
    loop = tqdm(enumerate(train_loader))  # Progress bar for monitoring training
    print(f'Epoch: {epoch}')
    losses = torch.tensor(data=[]).to(DEVICE)  # Store losses for this epoch
    
    for i, (x, _) in loop:
        x = x.to(DEVICE)  # Move input batch to the correct device
        
        # Downgrade image resolution and then upscale it again
        lr_x = F.interpolate(x, size=(INPUT_DIM//DOWNGRADE_FACTOR, INPUT_DIM//DOWNGRADE_FACTOR), mode='bilinear', align_corners=False)
        lr_x = F.interpolate(lr_x, size=(INPUT_DIM, INPUT_DIM), mode='bilinear', align_corners=False)
        
        # Forward pass: get the model prediction
        y = model(lr_x)
        
        # Compute the loss between prediction and ground truth
        loss = loss_fn(y, x)
        loss = loss.view((1)).to(DEVICE)  # Ensure loss tensor is properly shaped
        losses = torch.cat((losses, loss), -1)  # Store loss for later analysis

        # Backpropagation
        optimizer.zero_grad()  # Reset gradients to zero
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        
        loop.set_postfix(loss=loss.item())  # Update progress bar with loss value

    # Compute average loss for this epoch
    avg_loss = losses.mean().view((1)).to(DEVICE)
    avg_losses = torch.cat((avg_losses, avg_loss), -1)
    print(f'Loss: {avg_loss}')
    
    # Save model checkpoint every 5 epochs
    if epoch % 5 == 0:
        torch.save(model.state_dict(), f'sandstone/models/model_{epoch}')
        torch.save(avg_losses, f'sandstone/models/loss_{epoch}.pt')

# Save final trained model
torch.save(model.state_dict(), 'sandstone/models/model')