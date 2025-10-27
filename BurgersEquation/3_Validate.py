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

### ======================================== ###
###            Definindo Comparação          ###
### ======================================== ###

comparison = 1  # validacao(1) ou teste(2)

### ======================================== ###

# Configurar logging
log_dir = 'BurgersEquation/Output/3_Validate'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'Images'), exist_ok=True)

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

# Garantir precisão float64 em todos os tensores PyTorch
tc.set_default_dtype(tc.float64)

# Detecção de GPU
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
logger.info(f'Usando dispositivo: {device}')

tc.manual_seed(21)

### ======================================== ###
###            Definindo PINN                ###
### ======================================== ###

class PINN(nn.Module):

    def __init__(self, structure=[1, 10, 10, 1], activation=nn.Tanh(), dropout_ratio=0.0):
        super(PINN, self).__init__()
        self.structure = structure
        self.activation = activation
        self.dropout_ratio = dropout_ratio
        self.hidden_layers = nn.ModuleList()

        for i in range(len(structure)-1):
            self.hidden_layers.append(nn.Linear(structure[i], structure[i+1]))
            
        # Inicialização Xavier/Glorot
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inicialização Xavier/Glorot para PINNs
        Mantém a variância das ativações durante forward e backward pass
        """
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                # Xavier/Glorot normal initialization para tanh
                nn.init.xavier_normal_(layer.weight)
                # Inicializar bias com zeros
                nn.init.zeros_(layer.bias)
            
    
    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            
            # Aplica dropout personalizado apenas nas camadas ocultas durante treinamento
            if self.training and self.dropout_ratio > 0.0:
                # Gera máscara aleatória: 1 para manter, 0 para desativar
                keep_prob = 1.0 - self.dropout_ratio
                mask = (tc.rand_like(x) < keep_prob).float()
                
                # Aplica máscara e escala pelo keep_prob para compensar neurônios desativados
                x = x * mask / keep_prob
                
        x = self.hidden_layers[-1](x) # Sem ativação e sem dropout na última camada
        return x

### ======================================== ###
###             Carregando Dados             ###
### ======================================== ###

validacao = tc.load('BurgersEquation/Output/1_PreProcessing/Data/validacao.pt', map_location=device)
teste = tc.load('BurgersEquation/Output/1_PreProcessing/Data/teste.pt', map_location=device)
validacao_u = tc.load('BurgersEquation/Output/1_PreProcessing/Data/validacao_u.pt', map_location=device)
teste_u = tc.load('BurgersEquation/Output/1_PreProcessing/Data/teste_u.pt', map_location=device)

pinn_state = tc.load('BurgersEquation/Output/2_Train/Data/PINN_state.pt', map_location=device)

metadata1 = tc.load('BurgersEquation/Output/1_PreProcessing/Data/metadata.pt')
x_min = metadata1['x_min']
x_max = metadata1['x_max']
t_min = metadata1['t_min']
t_max = metadata1['t_max']

metadata2 = tc.load('BurgersEquation/Output/2_Train/Data/metadata.pt')
hidden_layers = metadata2['hidden_layers']
neurons_per_layer = metadata2['neurons_per_layer']

f = PINN(structure=[2] + [neurons_per_layer] * hidden_layers + [1], dropout_ratio=0.0).to(device)
f.load_state_dict(pinn_state)

### ======================================== ###
###            Validação/Teste               ###
### ======================================== ###

# Seleção de dados
pts_comp = validacao if comparison == 1 else teste
u_comp_ref = validacao_u if comparison == 1 else teste_u
label_comp = 'Validação' if comparison == 1 else 'Teste'

# Predição da rede
f.eval()
with tc.no_grad():
    u_comp_pred = f(pts_comp)

# Erro médio
erro_medio = tc.mean((u_comp_pred - u_comp_ref)**2).item()

# Grid para interpolação das predições
nx_grid = 256
nt_grid = 256
x_grid_1d = np.linspace(x_min, x_max, nx_grid)
t_grid_1d = np.linspace(t_min, t_max, nt_grid)
t_grid, x_grid = np.meshgrid(t_grid_1d, x_grid_1d, indexing='ij')

# Criar tensor de pontos para predição
pts_grid = tc.tensor(np.stack([x_grid.flatten(), t_grid.flatten()], axis=1), dtype=tc.float64, device=device)
with tc.no_grad():
    u_pred_grid = f(pts_grid).cpu().numpy().reshape(nt_grid, nx_grid)

# Parâmetros da malha de referência
nx = 1024      # Aumentar para melhor resolução espacial
nt = 10001     # Aumentar para melhor estabilidade (dt menor)
x_ref = np.linspace(x_min, x_max, nx)
t_ref = np.linspace(t_min, t_max, nt)
dx = x_ref[1] - x_ref[0]
dt = t_ref[1] - t_ref[0]
nu = 0.01 / np.pi

# Inicialização da solução na malha regular
u_ref = np.zeros((nt, nx), dtype=np.float64)
u_ref[0, :] = -np.sin(np.pi * x_ref)

# Função RHS de Burgers (diferenças centrais)
def rhs(u):
    dudx = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    d2udx2 = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
    return -u * dudx + nu * d2udx2

# Integração temporal (RK4 explícito)
for n in range(nt - 1):
    u = u_ref[n]
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    u_next = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    u_next[0] = 0.0
    u_next[-1] = 0.0
    u_ref[n + 1] = u_next

# Interpolador para pontos arbitrários (x, t)
interp = RegularGridInterpolator((t_ref, x_ref), u_ref, bounds_error=False, fill_value=0.0)

# Interpolação da referência no grid
u_ref_grid = interp(np.stack([t_grid.flatten(), x_grid.flatten()], axis=1)).reshape(nt_grid, nx_grid)

# Tempos para perfis
t_perfis = [0.25, 0.5, 0.75]

# Plot
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.5)

fig.suptitle(f'Comparação: Referência vs. PINN - {label_comp}\nErro Médio Quadrático: {erro_medio:.2e}', fontsize=14, y=0.995)

# Colormap referência
ax1 = fig.add_subplot(gs[0, :])
im1 = ax1.contourf(t_grid, x_grid, u_ref_grid, levels=100, cmap='jet', vmin=-1, vmax=1)
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.set_title(f'Solução Referência - {label_comp}')
ax1.set_xlim(t_min, t_max)
ax1.set_ylim(x_min, x_max)
cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02, aspect=10, ticks=np.arange(-1, 1.25, 0.25))

# Colormap predição
ax2 = fig.add_subplot(gs[1, :])
im2 = ax2.contourf(t_grid, x_grid, u_pred_grid, levels=100, cmap='jet', vmin=-1, vmax=1)
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.set_title(f'Solução PINN - {label_comp}')
ax2.set_xlim(t_min, t_max)
ax2.set_ylim(x_min, x_max)
cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02, aspect=10, ticks=np.arange(-1, 1.25, 0.25))

# Perfis temporais
handles, labels = None, None
for i, t_val in enumerate(t_perfis):
    ax = fig.add_subplot(gs[2, i])
    idx_t = np.argmin(np.abs(t_grid_1d - t_val))
    l1, = ax.plot(x_grid_1d, u_ref_grid[idx_t, :], 'b-', linewidth=2, label='Referência')
    l2, = ax.plot(x_grid_1d, u_pred_grid[idx_t, :], 'r--', linewidth=2, label='PINN')
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_title(f't = {t_val:.2f}')
    ax.set_xlim(x_min, x_max)
    ax.grid(True, alpha=0.3)
    if i == 0:
        handles, labels = [l1, l2], ['Referência', 'PINN']

# Legenda centralizada abaixo
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02), frameon=True)

plt.savefig('BurgersEquation/Output/3_Validate/Images/Comparison.png')

logger.info('Process Finished.')