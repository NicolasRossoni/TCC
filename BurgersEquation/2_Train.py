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

# Garantir precisão float64 em todos os tensores PyTorch
tc.set_default_dtype(tc.float64)

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
###            Funções Auxiliares            ###
### ======================================== ###

def random_select(treino, treino_u, ratio, n_ci):
    """
    Seleciona uma proporção aleatória dos dados de treino preservando as condições iniciais
    
    Args:
        treino: tensor com dados de entrada
        treino_u: tensor com dados de saída correspondentes
        ratio: proporção dos dados após 3*n_ci para selecionar
        n_ci: número de pontos de condição inicial
    
    Returns:
        partial_treino, partial_treino_u: tensors reduzidos mantendo correlação
    """
    # Preserva os primeiros 3*n_ci elementos (condições iniciais)
    initial_treino = treino[:3*n_ci]
    initial_treino_u = treino_u[:3*n_ci]
    
    # Seleciona aleatoriamente da parte restante
    remaining_treino = treino[3*n_ci:]
    remaining_treino_u = treino_u[3*n_ci:]
    
    # Número de amostras a selecionar da parte restante
    n_samples = int(len(remaining_treino) * ratio)
    
    # Gera índices aleatórios para manter correlação entre treino e treino_u
    indices = tc.randperm(len(remaining_treino))[:n_samples]
    
    # Seleciona os dados correspondentes
    selected_treino = remaining_treino[indices]
    selected_treino_u = remaining_treino_u[indices]
    
    # Concatena condições iniciais com dados selecionados
    partial_treino = tc.cat([initial_treino, selected_treino], dim=0)
    partial_treino_u = tc.cat([initial_treino_u, selected_treino_u], dim=0)
    
    return partial_treino, partial_treino_u

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
supervisionado = False
hidden_layers = 4
neurons_per_layer = 128
dataset_ratio = 1.0     # Seleciona parte dos dados de treino aleatoriamente
dropout_ratio = 0.0     # Proporção de neurônios a desativar durante treinamento
lr = 0.01
step_size = 5000
gamma = 0.95
epochs = 2000
epochs_log = 100        # A cada quantos epochs o loss é logado

# PINN
f = PINN(structure=[2] + [neurons_per_layer] * hidden_layers + [1], dropout_ratio=dropout_ratio).to(device)
optimizer = tc.optim.Adam(f.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

loss_log = []
start_time = perf_counter()

# Loop de treino
for epoch in range(epochs+1):

    # Selecionando parcialmente e aleatoriamente os dados de treino preservando condições iniciais
    partial_treino, partial_treino_u = random_select(treino, treino_u, dataset_ratio, n_ci)

    u = f(partial_treino) # inferencia dos dados de treino

    # Derivadas Parciais
    grad_u = tc.autograd.grad(u, partial_treino, grad_outputs=tc.ones_like(u), create_graph=True, retain_graph=True)[0]
    du_dx = grad_u[:, 0]
    du_dt = grad_u[:, 1]
    d2u_dx2 = tc.autograd.grad(du_dx, partial_treino, grad_outputs=tc.ones_like(du_dx), create_graph=True, retain_graph=True)[0][:, 0]

    # Losses
    loss_EDP = tc.mean((du_dt + u*du_dx - tc.tensor(0.01/np.pi)*d2u_dx2)**2)        # EDP
    loss_dados = tc.mean((u - partial_treino_u)**2) if supervisionado else tc.tensor(0.0)   # Supervisionado
    loss_EDP = loss_EDP if not supervisionado else tc.tensor(0.0)                   # Supervisionado
    loss_ci1 = tc.mean((u[:n_ci] - partial_treino_u[:n_ci])**2)                             # u(x,0)
    loss_ci2 = tc.mean((u[n_ci:2*n_ci] - partial_treino_u[n_ci:2*n_ci])**2)                 # u(1,t)
    loss_ci3 = tc.mean((u[2*n_ci:3*n_ci] - partial_treino_u[2*n_ci:3*n_ci])**2)             # u(-1,t)
    loss_ci = loss_ci1 + loss_ci2 + loss_ci3
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

logger.info('Process Finished.')