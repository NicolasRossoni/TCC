#### Importando Bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import logging
import os
import shutil
import sys

# Adicionar caminho do MixFunn ao sys.path
sys.path.append('/home/exx/users/nicolas/TCC/MixFunn')
from mixfunn import Mixfun

### ======================================== ###
###            Configuração Inicial          ###
### ======================================== ###

# Configurar diretórios de saída
log_dir = 'MixFunn/DampedHO/Output/2_Train'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'Images'), exist_ok=True)
shutil.rmtree(os.path.join(log_dir, 'Data'), ignore_errors=True)
os.makedirs(os.path.join(log_dir, 'Data'), exist_ok=True)

# Configurar logging
log_file = os.path.join(log_dir, 'console.log')
if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurar PyTorch
tc.set_default_dtype(tc.float64)
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
logger.info(f'Usando dispositivo: {device}')

# Seed para reprodutibilidade
tc.manual_seed(42)
if tc.cuda.is_available():
    tc.cuda.manual_seed(42)

### ======================================== ###
###             Carregando Dados             ###
### ======================================== ###

metadata = tc.load('MixFunn/DampedHO/Output/1_PreProcessing/Data/metadata.pt')
n_ci = metadata['n_ci']
M = metadata['M']
B = metadata['B']
K = metadata['K']

# Parâmetros derivados (usados internamente)
gamma = B / M
omega_0 = tc.sqrt(tc.tensor(K / M, dtype=tc.float64))

# Carregar dados de treino
treino = tc.load('MixFunn/DampedHO/Output/1_PreProcessing/Data/treino.pt', map_location=device)
treino_x = tc.load('MixFunn/DampedHO/Output/1_PreProcessing/Data/treino_x.pt', map_location=device)

# Habilitar gradientes no tensor de entrada (necessário para calcular derivadas)
treino.requires_grad_(True)

logger.info(f'Dados carregados - Treino: {treino.shape}')
logger.info(f'Parâmetros físicos - M: {M:.2f} kg, B: {B:.2f} kg/s, K: {K:.2f} N/m')

### ======================================== ###
###      Hiperparâmetros da Rede Neural      ###
### ======================================== ###

# Arquitetura da rede (seguindo exemplo do GitHub: tiago939/MixFunn)
n_in = 1                    # Dimensão de entrada (tempo t)
n_out = 1                   # Dimensão de saída (posição x)

# Hiperparâmetros MixFun (configuração do exemplo de referência)
normalization_function = False
normalization_neuron = False
p_drop = False
second_order_input = False
second_order_function = True  # CRÍTICO: Usar funções de segunda ordem
temperature = 1.0

# Hiperparâmetros de treinamento (baseado no exemplo)
epochs = 10000              # Número de épocas
lr = 0.1                    # Taxa de aprendizado (mesmo do exemplo)
betas = (0.9, 0.9)          # Betas do Adam (mesmo do exemplo)
batch_size = 256            # Tamanho do batch (mesmo do exemplo)
epochs_log = 1000           # Imprimir log a cada N épocas

logger.info(f'Arquitetura: 1 camada Mixfun com second_order_function=True')
logger.info(f'Treinamento: {epochs} épocas, lr={lr}, batch_size={batch_size}')

### ======================================== ###
###           Instanciando o Modelo          ###
### ======================================== ###

# Usar diretamente a classe Mixfun (sem wrapper)
model = Mixfun(
    n_in=n_in,
    n_out=n_out,
    normalization_function=normalization_function,
    normalization_neuron=normalization_neuron,
    p_drop=p_drop,
    second_order_input=second_order_input,
    second_order_function=second_order_function,
    temperature=temperature
).to(device)

# Contar parâmetros
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f'Número de parâmetros treináveis: {n_params:,}')

### ======================================== ###
###              Treinando PINN              ###
### ======================================== ###

# Otimizador (configuração do exemplo de referência)
optimizer = tc.optim.Adam(model.parameters(), lr=lr, betas=betas)

# Listas para armazenar histórico de loss
loss_history = []

logger.info('Iniciando treinamento...')

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    x = model(treino)  # Forward pass
    
    # Calcular derivadas da rede
    dx_dt = tc.autograd.grad(x, treino, grad_outputs=tc.ones_like(x), create_graph=True, retain_graph=True)[0]
    d2x_dt2 = tc.autograd.grad(dx_dt, treino, grad_outputs=tc.ones_like(dx_dt), create_graph=True, retain_graph=True)[0]
    
    # Loss da EDP: x''(t) + (B/M)·x'(t) + (K/M)·x(t) = 0
    residual_edp = d2x_dt2 + gamma*dx_dt + omega_0**2*x
    loss_edp = tc.mean(residual_edp**2)
    loss_ci = tc.mean((x[:n_ci] - treino_x[:n_ci])**2)  # Loss CI: x(0) = x0
    loss = loss_edp + loss_ci  # Loss total
    
    loss.backward()  # Backward pass
    optimizer.step()
    
    loss_history.append([loss.item(), loss_edp.item(), loss_ci.item()])  # Armazenar histórico
    
    if (epoch + 1) % epochs_log == 0 or epoch == 0:  # Log
        logger.info(f'Epoch {epoch+1:6d}/{epochs} | Loss: {loss.item():.6e} | EDP: {loss_edp.item():.6e} | CI: {loss_ci.item():.6e}')

logger.info('Treinamento concluído!')

### ======================================== ###
###          Salvando Resultados             ###
### ======================================== ###

# Salvar histórico de loss
loss_history = np.array(loss_history)
np.save(os.path.join(log_dir, 'Data', 'loss_history.npy'), loss_history)

# Salvar hiperparâmetros da rede (necessários para recriar o modelo)
model_config = {
    'n_in': n_in,
    'n_out': n_out,
    'normalization_function': normalization_function,
    'normalization_neuron': normalization_neuron,
    'p_drop': p_drop,
    'second_order_input': second_order_input,
    'second_order_function': second_order_function,
    'temperature': temperature,
}
tc.save(model_config, os.path.join(log_dir, 'Data', 'model_config.pt'))

# Salvar pesos do modelo
tc.save(model.state_dict(), os.path.join(log_dir, 'Data', 'model_weights.pt'))

# Salvar metadados do treinamento
training_metadata = {
    'epochs': epochs,
    'lr': lr,
    'n_params': n_params,
    'final_loss': loss.item(),
    'final_loss_edp': loss_edp.item(),
    'final_loss_ci': loss_ci.item(),
}
tc.save(training_metadata, os.path.join(log_dir, 'Data', 'training_metadata.pt'))

logger.info(f'Modelo salvo em: {log_dir}/Data/')

### ======================================== ###
###          Plotando Curvas de Loss         ###
### ======================================== ###

fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogy(loss_history[:, 0], 'k-', linewidth=2, label='Loss Total')
ax.semilogy(loss_history[:, 1], 'r--', linewidth=1.5, label='Loss EDP')
ax.semilogy(loss_history[:, 2], 'b--', linewidth=1.5, label='Loss CI')

ax.set_xlabel('Época')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Histórico de Treinamento - Damped Harmonic Oscillator')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(log_dir, 'Images', 'loss_history.png'), dpi=150)

logger.info('Process Finished.')
