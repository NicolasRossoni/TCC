#### Importando Bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import torch as tc

from time import perf_counter
from torch import nn
from torch.optim.lr_scheduler import StepLR
from scipy.interpolate import RegularGridInterpolator
import logging
import os

# Configurar logging
log_dir = 'BurgersEquation/Output/2_Train'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'Images'), exist_ok=True)
os.makedirs(os.path.join(log_dir, 'Data'), exist_ok=True)

log_file = os.path.join(log_dir, 'console.log')
if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'console.log')),
        logging.StreamHandler()  # Também mantém no console
    ]
)
logger = logging.getLogger(__name__)

# Garantir precisão float32 em todos os tensores PyTorch
tc.set_default_dtype(tc.float32)

# Detecção de GPU
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
logger.info(f'Usando dispositivo: {device}')

# Initialize CUDA context to avoid cuBLAS warning
if tc.cuda.is_available():
    with tc.no_grad():
        dummy_tensor = tc.tensor([1.0], device=device)
        _ = dummy_tensor * 2.0  # Simple operation to establish CUDA context

tc.manual_seed(21)

### ======================================== ###
###            Definindo PINN                ###
### ======================================== ###

class PINN(nn.Module):

    def __init__(self, structure=[1, 10, 10, 1], activation=nn.Tanh()):
        super(PINN, self).__init__()
        self.structure = structure
        self.activation = activation
        self.hidden_layers = nn.ModuleList()

        for i in range(len(structure)-1):
            self.hidden_layers.append(nn.Linear(structure[i], structure[i+1]))
            
    
    def forward(self, x):
        for layer in self.hidden_layers[:-1]:
            x = self.activation(layer(x))
        x = self.hidden_layers[-1](x) # Sem ativação na última camada
        return x

### ======================================== ###
###             Carregando Dados             ###
### ======================================== ###

metadata = tc.load('BurgersEquation/Output/1_PreProcessing/Data/metadata.pt')
n_ci = metadata['n_ci']

treino = tc.load('BurgersEquation/Output/1_PreProcessing/Data/treino.pt', map_location=device)
treino_ci_u = tc.load('BurgersEquation/Output/1_PreProcessing/Data/treino_ci_u.pt', map_location=device)
treino_u = tc.load('BurgersEquation/Output/1_PreProcessing/Data/treino_u.pt', map_location=device)

### ======================================== ###
###              Treinando PINN              ###
### ======================================== ###

# Hiperparâmetros
supervisionado = True
hidden_layers = 20
neurons_per_layer = 60
activation = nn.Tanh()
lr = 0.005
step_size = 500
gamma = 0.95
epochs = 1000
epochs_log = 250 # A cada quantos epochs o loss é logado

# PINN
f = PINN(structure=[2] + [neurons_per_layer] * hidden_layers + [1], activation=activation).to(device)
optimizer = tc.optim.Adam(f.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

loss_log = []
start_time = perf_counter()

# Loop de treino
for epoch in range(epochs+1):
    u = f(treino) # inferencia dos dados de treino

    # Derivadas Parciais
    grad_u = tc.autograd.grad(u, treino, grad_outputs=tc.ones_like(u), create_graph=True, retain_graph=True)[0]
    du_dx = grad_u[:, 0]
    du_dt = grad_u[:, 1]
    d2u_dx2 = tc.autograd.grad(du_dx, treino, grad_outputs=tc.ones_like(du_dx), create_graph=True, retain_graph=True)[0][:, 0]

    # Losses
    #loss_dados = tc.mean((u - treino_u)**2) if supervisionado else tc.tensor(0.0)   # Supervisionado
    #loss_EDP = tc.mean((du_dt + u*du_dx - tc.tensor(0.01/np.pi)*d2u_dx2)**2)        # EDP
    loss_ci1 = tc.mean((u[:n_ci] - treino_u[:n_ci])**2)                          # u(x,0)
    loss_ci2 = tc.mean((u[n_ci:2*n_ci] - treino_u[n_ci:2*n_ci])**2)              # u(1,t)
    loss_ci3 = tc.mean((u[2*n_ci:3*n_ci] - treino_u[2*n_ci:3*n_ci])**2)          # u(-1,t)
    loss_ci = loss_ci1 + loss_ci2 + loss_ci3
    loss_EDP = tc.tensor(0.0)
    #loss_ci = tc.tensor(0.0)
    loss_dados = tc.tensor(0.0)
    loss = loss_dados + loss_EDP + loss_ci
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Salvando loss e printando a cada 1000 epochs
    loss_log.append([loss.item(), loss_dados.item(), loss_EDP.item(), loss_ci.item()])
    if epoch % epochs_log == 0:
        end_time = perf_counter()
        logger.info(f"Epoch {epoch} - Loss: {loss.item():.2e} - Time: {end_time - start_time:.2f}s")
        start_time = perf_counter()
    
### ======================================== ###
###               Plotando Loss              ###
### ======================================== ###

loss_array = np.array(loss_log)

# Figura 1: Apenas Loss Total
plt.figure(figsize=(8,5))
plt.semilogy(loss_array[:, 0], 'r-', linewidth=2, label='Loss Total')
plt.ylabel("Loss (log scale)")
plt.xlabel("Epochs")
plt.legend()
plt.title("Evolução da Loss Total")
plt.tight_layout()
plt.savefig('BurgersEquation/Output/2_Train/Images/Loss.png')

# Figura 2: Todas as Losses
plt.figure(figsize=(10,6))
plt.semilogy(loss_array[:, 0], 'r-', linewidth=2, label='Loss Total')
plt.semilogy(loss_array[:, 1], 'b--', linewidth=1.5, label='Loss Dados')
plt.semilogy(loss_array[:, 2], 'g--', linewidth=1.5, label='Loss EDP')
plt.semilogy(loss_array[:, 3], 'k--', linewidth=1.5, label='Loss CI')
plt.ylabel("Loss (log scale)")
plt.xlabel("Epochs")
plt.legend()
plt.title("Evolução das Losses")
plt.tight_layout()
plt.savefig('BurgersEquation/Output/2_Train/Images/Loss_All.png')

### ======================================== ###
###               Salvando Dados             ###
### ======================================== ###

tc.save(f.state_dict(), 'BurgersEquation/Output/2_Train/Data/PINN_state.pt')

metadata = {
    'hidden_layers': hidden_layers,
    'neurons_per_layer': neurons_per_layer,
}
tc.save(metadata, 'BurgersEquation/Output/2_Train/Data/metadata.pt')
