import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from tqdm import tqdm
import os
import shutil

class SinteticDataset(torch.utils.data.Dataset):
    def __init__(self, directory, subwindow=None,
                 transform=None, D=1, skip=1):
        # D - Number of datas to retrieve. 
        # skip - interval between observations
        # skip = 1 we get every observation
        # skip = 2 every other observation

        self.directory = directory
        self.subwindow = subwindow  # Proportion of subwindow
        self.files = [f for f in os.listdir(directory) if '.npy' in f] #All available files
        self.D = D
        self.skip = skip
        self.Nx = (40*2)*2
        self.Ny = (68*2)*2
        self.valid_index = self.calcular_indices_validos()


    def __len__(self):
        return len(self.valid_index)

    def __getitem__(self, idx, subwindow=None):
        idx = self.valid_index[idx]
        file = idx//150
        data = np.load(self.directory + "/" + self.files[file])
        if file>0:
            idx = idx-file*150
        psi1 = data[0:150]
        psi2 = data[150:]
        psi1 = data[idx: idx + self.D*self.skip: self.skip]
        psi2 = data[idx: idx + self.D*self.skip: self.skip]
        if subwindow is None:
            subwindow = self.subwindow

        Ny_mesh, Nx_mesh = torch.meshgrid(torch.arange(self.Ny),torch.arange(self.Nx))
        lat_idx, lon_idx = self.get_indices_from_proportion(Ny_mesh, Nx_mesh, subwindow)
        # if subwindow == None
        # len(Ny_mesh) == Ny
        # len(Nx_mesh) == Nx
        # lat_idx = [0, Ny]
        # lon_idx = [0, Nx]


        psi1 = torch.tensor(psi1.reshape(psi1.shape[0],
                                         self.Ny,self.Nx)[:,
                                                     lat_idx[0]:lat_idx[1], 
                                                     lon_idx[0]:lon_idx[1]])
        #if subwindow == None,
        #ps1 has size (ps1.shape[0], Ny, Nx)
        
        psi2 = torch.tensor(psi2.reshape(psi2.shape[0],
                                         self.Ny,self.Nx)[:,
                                                     lat_idx[0]:lat_idx[1],
                                                     lon_idx[0]:lon_idx[1]])
        # if subwindow == None 
        # ps2 has size (ps2.shape[0], Ny, Nx)

        return torch.permute(torch.stack([psi1,psi2]), (1,0,2,3))
        # [D, psis, NY, NX]
        return self.prepare_tensors(u_velocity, v_velocity, ssh, mask, sliced_latitudes, sliced_longitudes)
    

    def get_indices_from_proportion(self, latitudes, longitudes, subwindow):
        if subwindow:
            lat_range = [int(subwindow[0][0] * len(latitudes)), int(subwindow[0][1] * len(latitudes))]
            lon_range = [int(subwindow[1][0] * len(longitudes)), int(subwindow[1][1] * len(longitudes))]
            return lat_range, lon_range
        return [0, len(latitudes)], [0, len(longitudes)]
    
    """
    def prepare_tensors(self, u_velocity, v_velocity, ssh, mask, latitudes, longitudes):
        u_tensor = torch.tensor(u_velocity.filled(0.0), dtype=torch.float32)
        v_tensor = torch.tensor(v_velocity.filled(0.0), dtype=torch.float32)
        ssh_tensor = torch.tensor(ssh.filled(0.0), dtype=torch.float32)
        combined_tensor = torch.stack([u_tensor, v_tensor, ssh_tensor])
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # Creating tensors for the latitude and longitude slices
        lat_tensor = torch.tensor(latitudes, dtype=torch.float32)
        lon_tensor = torch.tensor(longitudes, dtype=torch.float32)

        return mask_tensor, combined_tensor, lat_tensor, lon_tensor
    """
    
    def calcular_indices_validos(self):
        block_size = 150
        total_size = block_size*len(self.files)
        
        # Lista para armazenar os índices válidos
        valid_indices = []
        
        # Percorre todos os índices possíveis
        for start in range(total_size):
            # Calcula os índices da sequência
            indices = [start + i * self.skip for i in range(self.D)]
            # print(indices)
            # Verifica se o último índice é válido dentro do tamanho total
            if indices[-1] >= total_size:
                continue

            # Determina o bloco do índice inicial e final
            start_block = start // block_size
            end_block = indices[-1] // block_size
            # print('startbloc = ', start_block)
            # print('end = ', end_block)
            # Verifica se o bloco inicial e final são iguais
            if start_block != end_block:
                continue
            
            # Adiciona o índice inicial à lista de válidos se todas as condições forem satisfeitas
            valid_indices.append(start)
        
        return valid_indices

# if dataset folder exists, delete it
if os.path.exists('ocean/data/processed'):
        shutil.rmtree('ocean/data/processed')
# create train and test set folders
os.makedirs('ocean/data/processed/psi1')
os.makedirs('ocean/data/processed/psi2')


# dataset = SinteticDataset(D = 150)
print(os.listdir('.'))
path = "ocean/data/raw/"
dataset = SinteticDataset(path, D = 150)

for i in tqdm(range(20)):
    for j in range(150):
        np.save(f"ocean/data/processed/psi1/{i}_{j}.npy", dataset[i][j][0])
        np.save(f"ocean/data/processed/psi2/{i}_{j}.npy", dataset[i][j][1])