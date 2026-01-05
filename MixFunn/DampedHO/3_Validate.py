#### Importando Bibliotecas
import matplotlib.pyplot as plt
from matplotlib import gridspec
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
log_dir = 'MixFunn/DampedHO/Output/3_Validate'
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

### ======================================== ###
###     Solução Analítica do Oscilador      ###
### ======================================== ###

def solucao_analitica(t, x0, v0, omega_0, gamma):
    """
    Calcula a solução analítica do oscilador harmônico amortecido
    
    Equação: M*x''(t) + B*x'(t) + K*x(t) = 0
    Forma normalizada: x''(t) + (B/M)*x'(t) + (K/M)*x(t) = 0
    
    Args:
        t: tempo (array ou tensor) - shape: [N] ou [N, 1]
        x0: posição inicial
        v0: velocidade inicial
        omega_0: frequência natural = sqrt(K/M)
        gamma: coeficiente de amortecimento normalizado = B/M
    
    Returns:
        x(t): posição em função do tempo - shape: [N, 1]
        v(t): velocidade em função do tempo - shape: [N, 1]
    """
    # Converter para numpy se for tensor
    if isinstance(t, tc.Tensor):
        t_np = t.detach().cpu().numpy().flatten()
    else:
        t_np = np.array(t).flatten()
    
    # Discriminante para classificar o regime (forma do paper)
    delta = gamma**2 - 4*omega_0**2
    
    if delta < 0:  # Subamortecido (oscilação com decaimento)
        # Frequência amortecida
        omega_d = np.sqrt(omega_0**2 - (gamma/2)**2)
        
        # Constantes A e phi baseadas nas condições iniciais
        # x(t) = A * exp(-gamma*t/2) * cos(omega_d*t + phi)
        A = np.sqrt(x0**2 + ((v0 + (gamma/2)*x0)/omega_d)**2)
        phi = np.arctan2(-(v0 + (gamma/2)*x0)/omega_d, x0)
        
        x = A * np.exp(-(gamma/2) * t_np) * np.cos(omega_d * t_np + phi)
        v = A * np.exp(-(gamma/2) * t_np) * (-(gamma/2) * np.cos(omega_d * t_np + phi) - omega_d * np.sin(omega_d * t_np + phi))
        
    elif delta == 0:  # Criticamente amortecido
        # x(t) = (A + B*t) * exp(-gamma*t/2)
        A = x0
        B = v0 + (gamma/2)*x0
        
        x = (A + B * t_np) * np.exp(-(gamma/2) * t_np)
        v = B * np.exp(-(gamma/2) * t_np) - (gamma/2) * (A + B * t_np) * np.exp(-(gamma/2) * t_np)
        
    else:  # Superamortecido
        # x(t) = C1*exp(r1*t) + C2*exp(r2*t)
        r1 = (-gamma + np.sqrt(delta)) / 2
        r2 = (-gamma - np.sqrt(delta)) / 2
        
        C1 = (v0 - r2*x0) / (r1 - r2)
        C2 = (r1*x0 - v0) / (r1 - r2)
        
        x = C1 * np.exp(r1 * t_np) + C2 * np.exp(r2 * t_np)
        v = C1 * r1 * np.exp(r1 * t_np) + C2 * r2 * np.exp(r2 * t_np)
    
    return x.reshape(-1, 1), v.reshape(-1, 1)

### ======================================== ###
###             Carregando Dados             ###
### ======================================== ###

# Carregar metadados do pré-processamento
metadata = tc.load('MixFunn/DampedHO/Output/1_PreProcessing/Data/metadata.pt')
t_min = metadata['t_min']
t_max = metadata['t_max']
M = metadata['M']
B = metadata['B']
K = metadata['K']
x0 = metadata['x0']
v0 = metadata['v0']
regime = metadata['regime']

# Parâmetros derivados
gamma = B / M
omega_0 = np.sqrt(K / M)

logger.info(f'Parâmetros físicos - M: {M:.2f} kg, B: {B:.2f} kg/s, K: {K:.2f} N/m')
logger.info(f'Parâmetros derivados - gamma: {gamma:.4f} 1/s, omega_0: {omega_0:.4f} rad/s')
logger.info(f'Condições iniciais - x0: {x0:.4f}, v0: {v0:.4f}')
logger.info(f'Regime de amortecimento: {regime}')

# Carregar dados de validação e teste
validacao = tc.load('MixFunn/DampedHO/Output/1_PreProcessing/Data/validacao.pt', map_location=device)
teste = tc.load('MixFunn/DampedHO/Output/1_PreProcessing/Data/teste.pt', map_location=device)
validacao_x = tc.load('MixFunn/DampedHO/Output/1_PreProcessing/Data/validacao_x.pt', map_location=device)
teste_x = tc.load('MixFunn/DampedHO/Output/1_PreProcessing/Data/teste_x.pt', map_location=device)

# Carregar metadados do treinamento
training_metadata = tc.load('MixFunn/DampedHO/Output/2_Train/Data/training_metadata.pt')
logger.info(f'Treinamento - Épocas: {training_metadata["epochs"]}, LR: {training_metadata["lr"]}')
logger.info(f'Loss final - Total: {training_metadata["final_loss"]:.6e}')

### ======================================== ###
###          Recriando o Modelo              ###
### ======================================== ###

# Carregar hiperparâmetros da rede
model_config = tc.load('MixFunn/DampedHO/Output/2_Train/Data/model_config.pt')

n_in = model_config['n_in']
n_out = model_config['n_out']
normalization_function = model_config['normalization_function']
normalization_neuron = model_config['normalization_neuron']
p_drop = model_config['p_drop']
second_order_input = model_config['second_order_input']
second_order_function = model_config['second_order_function']
temperature = model_config['temperature']

logger.info(f'Arquitetura: 1 camada Mixfun com second_order_function={second_order_function}')

# Instanciar modelo usando diretamente Mixfun e carregar pesos
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

model.load_state_dict(tc.load('MixFunn/DampedHO/Output/2_Train/Data/model_weights.pt', map_location=device))
model.eval()

logger.info('Modelo carregado com sucesso!')

### ======================================== ###
###       Avaliando no Conjunto de Teste     ###
### ======================================== ###

# Criar grid de tempo para visualização (alta resolução)
t_grid = tc.linspace(t_min, t_max, 1000, device=device).reshape(-1, 1)
t_grid_np = t_grid.detach().cpu().numpy().flatten()

# Predição da rede
with tc.no_grad():
    x_pred_validacao = model(validacao).detach().cpu().numpy()
    x_pred_teste = model(teste).detach().cpu().numpy()
    x_pred_grid = model(t_grid).detach().cpu().numpy()

# Solução analítica (apenas posição)
x_ref_validacao, _ = solucao_analitica(validacao, x0, v0, omega_0, gamma)
x_ref_teste, _ = solucao_analitica(teste, x0, v0, omega_0, gamma)
x_ref_grid, _ = solucao_analitica(t_grid, x0, v0, omega_0, gamma)

# Calcular erros
erro_validacao = np.abs(x_pred_validacao - x_ref_validacao)
erro_teste = np.abs(x_pred_teste - x_ref_teste)

# Métricas
mae_validacao = np.mean(erro_validacao)
mae_teste = np.mean(erro_teste)
mse_validacao = np.mean(erro_validacao**2)
mse_teste = np.mean(erro_teste**2)

logger.info(f'Validação - MAE: {mae_validacao:.6e}, MSE: {mse_validacao:.6e}')
logger.info(f'Teste - MAE: {mae_teste:.6e}, MSE: {mse_teste:.6e}')


### ======================================== ###
###            Plotando Resultados           ###
### ======================================== ###

fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. Posição x(t)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t_grid_np, x_ref_grid, 'b-', linewidth=2, label='Solução Analítica', alpha=0.8)
ax1.plot(t_grid_np, x_pred_grid, 'r--', linewidth=2, label='PINN (MixFunn)', alpha=0.8)
ax1.scatter(validacao.cpu().numpy(), validacao_x.cpu().numpy(), c='green', s=20, alpha=0.5, label='Validação', zorder=5)
ax1.set_xlabel('t (s)')
ax1.set_ylabel('x(t) (m)')
ax1.set_title('Posição do Oscilador')
ax1.set_xlim(t_min, t_max)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)


# 2. Erro absoluto (log scale)
ax2 = fig.add_subplot(gs[0, 1])
erro_grid = np.abs(x_pred_grid - x_ref_grid)
ax2.semilogy(t_grid_np, erro_grid, 'purple', linewidth=2)
ax2.set_xlabel('t (s)')
ax2.set_ylabel('|Erro| (m)')
ax2.set_title('Erro Absoluto da Posição (log scale)')
ax2.set_xlim(t_min, t_max)
ax2.grid(True, alpha=0.3)


# 3. Histograma de erros
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(erro_teste.flatten(), bins=50, color='purple', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Erro Absoluto (m)')
ax3.set_ylabel('Frequência')
ax3.set_title(f'Distribuição de Erros no Teste\nMAE = {mae_teste:.6e}')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Curva de loss do treinamento
ax4 = fig.add_subplot(gs[1, 1])
loss_history = np.load('MixFunn/DampedHO/Output/2_Train/Data/loss_history.npy')
ax4.semilogy(loss_history[:, 0], 'k-', linewidth=2, label='Loss Total')
ax4.semilogy(loss_history[:, 1], 'r--', linewidth=1.5, label='Loss EDP')
ax4.semilogy(loss_history[:, 2], 'b--', linewidth=1.5, label='Loss CI')
ax4.set_xlabel('Época')
ax4.set_ylabel('Loss (log scale)')
ax4.set_title('Histórico de Treinamento')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

# Título geral
fig.suptitle(f'Damped Harmonic Oscillator - PINN Validation\nM={M:.2f} kg, B={B:.2f} kg/s, K={K:.2f} N/m, Regime: {regime}', 
             fontsize=14, fontweight='bold')

plt.savefig(os.path.join(log_dir, 'Images', 'validation_results.png'), dpi=150, bbox_inches='tight')

logger.info(f'Resultados salvos em: {log_dir}/Images/validation_results.png')
logger.info('Process Finished.')
