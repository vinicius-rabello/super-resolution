import torch
from torch import nn

kernel_size = 4  # (4, 4) kernel
init_channels = 8  # initial number of filters
input_dim = (272, 160)  # input shape (input_dim x input_dim)

# input img -> hidden dim -> mean, std -> reparametrization trick -> decoder -> output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, init_dim=8, latent_dim=3, kernel_size=4):
        super().__init__()
        self.init_dim = init_dim
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        # Encoder
        self.enc1 = nn.Conv2d(
            in_channels=1, out_channels=self.init_dim, kernel_size=self.kernel_size, 
            stride=2, padding=1  # Output: (136, 80)
        )
        self.enc2 = nn.Conv2d(
            in_channels=self.init_dim, out_channels=self.init_dim * 2, kernel_size=self.kernel_size, 
            stride=2, padding=1  # Output: (68, 40)
        )
        self.enc3 = nn.Conv2d(
            in_channels=self.init_dim * 2, out_channels=self.init_dim * 4, kernel_size=self.kernel_size, 
            stride=2, padding=1  # Output: (34, 20)
        )
        self.enc4 = nn.Conv2d(
            in_channels=self.init_dim * 4, out_channels=self.init_dim * 8, kernel_size=self.kernel_size, 
            stride=2, padding=1  # Output: (17, 10)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features = self.init_dim * 8 * 17 * 10, out_features = self.init_dim * 16)
        self.fc_mu = nn.Linear(self.init_dim * 16, self.latent_dim)  # Dimensão latente parametrizada
        self.fc_log_var = nn.Linear(self.init_dim * 16, self.latent_dim)  # Dimensão latente parametrizada
        self.fc2 = nn.Linear(self.latent_dim, self.init_dim * 8 * 17 * 10)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=self.init_dim * 8, out_channels=self.init_dim * 4, kernel_size=self.kernel_size, 
            stride=2, padding=1  # Output: (34, 20)
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=self.init_dim * 4, out_channels=self.init_dim * 2, kernel_size=self.kernel_size, 
            stride=2, padding=1  # Output: (68, 40)
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=self.init_dim * 2, out_channels=self.init_dim, kernel_size=self.kernel_size, 
            stride=2, padding=1  # Output: (136, 80)
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=self.init_dim, out_channels=1, kernel_size=self.kernel_size, 
            stride=2, padding=1  # Output: (272, 160)
        )
        
        # Definindo ReLU e Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def encode(self, x):
        x = self.relu(self.enc1(x))
        # print("enc1", x.shape)
        x = self.relu(self.enc2(x))
        # print("enc2", x.shape)
        x = self.relu(self.enc3(x))
        # print("enc3", x.shape)
        x = self.relu(self.enc4(x))
        # print("enc4", x.shape)
        x = x.view(x.size(0), -1)
        # print("view", x.shape)
        h = self.fc1(x)
        # print("fc1", h.shape)
        mu, sigma = self.fc_mu(h), self.fc_log_var(h)
        return mu, sigma
    
    def decode(self, z):
        x = self.fc2(z)
        x = x.view(x.size(0), self.init_dim * 8, 17, 10)  # Ajuste a forma aqui
        x = self.relu(self.dec1(x))
        # print("dec1", x.shape)
        x = self.relu(self.dec2(x))
        # print("dec2", x.shape)
        x = self.relu(self.dec3(x))
        # print("dec3", x.shape)
        x = self.dec4(x)  # Última camada
        # print("dec4", x.shape)
        return torch.sigmoid(x)
    
    def forward(self, x):
        x = self.dropout(x)
        mu, sigma = self.encode(x)
        # print("encoded", mu, sigma)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        # print("reparametrizacao", z_reparametrized)
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 1, 272, 160).to(device)
    vae = VariationalAutoEncoder(latent_dim=5).to(device)
    enc = vae.encode(x)[0]
    print(enc)
    dec = vae.decode(enc)
    print(dec.shape)
