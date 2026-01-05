#### Importando Bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import torch as tc

from time import perf_counter
import logging
import os
import sys

# Adicionar caminho do MixFunn ao sys.path
sys.path.append('/home/exx/users/nicolas/TCC/MixFunn')

# Configurar logging
log_dir = 'MixFunn/DampedHO/Output/1_PreProcessing'
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

tc.manual_seed(21)

### ======================================== ###
###   Parâmetros do Oscilador Harmônico     ###
### ======================================== ###

# Parâmetros físicos do sistema
# Equação (paper de referência com F(t)=0): M*x''(t) + B*x'(t) + K*x(t) = 0
M = 1.0               # Massa (kg)
B = 0.6               # Coeficiente de amortecimento (kg/s)
K = (2.0 * np.pi)**2  # Constante da mola (N/m)

# Parâmetros derivados (usados internamente)
gamma = B / M         # Coeficiente de amortecimento normalizado (1/s)
omega_0 = np.sqrt(K / M)  # Frequência natural (rad/s)

logger.info(f'Parâmetros físicos - M: {M:.2f} kg, B: {B:.2f} kg/s, K: {K:.2f} N/m')
logger.info(f'Parâmetros derivados - gamma: {gamma:.4f} 1/s, omega_0: {omega_0:.4f} rad/s')

# Verificar regime de amortecimento
delta = gamma**2 - 4*omega_0**2
if delta < 0:
    regime = 'Subamortecido'
    omega_d = np.sqrt(omega_0**2 - (gamma/2)**2)  # Frequência amortecida
    logger.info(f'Regime: {regime} - omega_d = {omega_d:.4f} rad/s')
elif delta == 0:
    regime = 'Criticamente Amortecido'
    logger.info(f'Regime: {regime}')
else:
    regime = 'Superamortecido'
    r1 = (-gamma + np.sqrt(delta)) / 2
    r2 = (-gamma - np.sqrt(delta)) / 2
    logger.info(f'Regime: {regime} - r1 = {r1:.4f}, r2 = {r2:.4f}')

# Condições iniciais
x0 = 1.0      # Posição inicial (m)
v0 = 0.0      # Velocidade inicial (m/s)

# Definindo domínio temporal
t_min, t_max = 0.0, 10.0

# Número de pontos
n_ci = 50     # Pontos para condições iniciais (t=0)
n = 5000      # Pontos usados no treino, validação e teste

distribution = [0.6, 0.2, 0.2] # Distribuição entre treino, validação e teste

### ======================================== ###
###        Amostragem Latin Hypercube       ###
### ======================================== ###

def latin_hypercube_1d(n_points, t_min, t_max):
    """
    Amostragem Latin Hypercube em 1D para garantir cobertura uniforme do domínio temporal
    
    Args:
        n_points: número de pontos a amostrar
        t_min: tempo mínimo
        t_max: tempo máximo
    
    Returns:
        tensor shape [n_points, 1] com valores de t
    """
    intervals = tc.linspace(0, 1, n_points + 1)
    points = tc.zeros(n_points, 1)
    
    # Permutação aleatória dos intervalos
    perm = tc.randperm(n_points)
    points[:, 0] = intervals[:-1] + (intervals[1] - intervals[0]) * tc.rand(n_points)
    points[:, 0] = points[perm, 0]
    
    # Escalar para o domínio definido
    points[:, 0] = points[:, 0] * (t_max - t_min) + t_min  # t
    return points

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
        # x'(t) = A * exp(-gamma*t/2) * [-(gamma/2)*cos(omega_d*t + phi) - omega_d*sin(omega_d*t + phi)]
        
        # Em t=0: x0 = A*cos(phi), v0 = -A*(gamma/2)*cos(phi) - A*omega_d*sin(phi)
        A = np.sqrt(x0**2 + ((v0 + (gamma/2)*x0)/omega_d)**2)
        phi = np.arctan2(-(v0 + (gamma/2)*x0)/omega_d, x0)
        
        x = A * np.exp(-(gamma/2) * t_np) * np.cos(omega_d * t_np + phi)
        v = A * np.exp(-(gamma/2) * t_np) * (-(gamma/2) * np.cos(omega_d * t_np + phi) - omega_d * np.sin(omega_d * t_np + phi))
        
    elif delta == 0:  # Criticamente amortecido
        # x(t) = (A + B*t) * exp(-gamma*t/2)
        # x'(t) = B*exp(-gamma*t/2) - (gamma/2)*(A + B*t)*exp(-gamma*t/2)
        
        # Em t=0: x0 = A, v0 = B - (gamma/2)*A
        A = x0
        B = v0 + (gamma/2)*x0
        
        x = (A + B * t_np) * np.exp(-(gamma/2) * t_np)
        v = B * np.exp(-(gamma/2) * t_np) - (gamma/2) * (A + B * t_np) * np.exp(-(gamma/2) * t_np)
        
    else:  # Superamortecido
        # x(t) = C1*exp(r1*t) + C2*exp(r2*t)
        r1 = (-gamma + np.sqrt(delta)) / 2
        r2 = (-gamma - np.sqrt(delta)) / 2
        
        # Em t=0: x0 = C1 + C2, v0 = C1*r1 + C2*r2
        C1 = (v0 - r2*x0) / (r1 - r2)
        C2 = (r1*x0 - v0) / (r1 - r2)
        
        x = C1 * np.exp(r1 * t_np) + C2 * np.exp(r2 * t_np)
        v = C1 * r1 * np.exp(r1 * t_np) + C2 * r2 * np.exp(r2 * t_np)
    
    return x.reshape(-1, 1), v.reshape(-1, 1)

### ================================== ###
###    Condições Iniciais/Contorno     ###
### ================================== ###

# Pontos de condição inicial em t=0
# Queremos garantir que a rede aprenda x(0) = x0 e x'(0) = v0
treino_ci = tc.zeros(n_ci, 1)  # t = 0 -> shape: [n_ci, 1]

# Solução nas condições iniciais
x_ci, v_ci = solucao_analitica(treino_ci, x0, v0, omega_0, gamma)
treino_ci_x = tc.tensor(x_ci, dtype=tc.float64, device=device)  # x(0) = x0 -> shape: [n_ci, 1]
treino_ci_v = tc.tensor(v_ci, dtype=tc.float64, device=device)  # x'(0) = v0 -> shape: [n_ci, 1]

### ==================================== ###
###  Dados de treino, validação e teste  ###
### ==================================== ###

n_treino = int(n * distribution[0])
n_validacao = int(n * distribution[1])
n_teste = int(n * distribution[2])

# Pontos t - treino (incluindo condições iniciais)
treino_geral = latin_hypercube_1d(n_treino, t_min, t_max)       # t -> shape: [n_treino, 1]
treino = tc.cat([treino_ci, treino_geral]).to(device)           # t -> shape: [n_ci + n_treino, 1]
treino.requires_grad_(True)

# Pontos t - validação
validacao = latin_hypercube_1d(n_validacao, t_min, t_max).to(device)  # t -> shape: [n_validacao, 1]
validacao.requires_grad_(False)

# Pontos t - teste
teste = latin_hypercube_1d(n_teste, t_min, t_max).to(device)          # t -> shape: [n_teste, 1]
teste.requires_grad_(False)

### ========================================= ###
###   Solução Analítica (todos os conjuntos) ###
### ========================================= ###

# Treino (apenas posição)
x_treino, _ = solucao_analitica(treino, x0, v0, omega_0, gamma)
treino_x = tc.tensor(x_treino, dtype=tc.float64, device=device)  # x(t) -> shape: [n_ci + n_treino, 1]

# Validação (apenas posição)
x_validacao, _ = solucao_analitica(validacao, x0, v0, omega_0, gamma)
validacao_x = tc.tensor(x_validacao, dtype=tc.float64, device=device)  # x(t) -> shape: [n_validacao, 1]

# Teste (apenas posição)
x_teste, _ = solucao_analitica(teste, x0, v0, omega_0, gamma)
teste_x = tc.tensor(x_teste, dtype=tc.float64, device=device)  # x(t) -> shape: [n_teste, 1]

logger.info(f'Shapes - Treino: {treino.shape}, Validação: {validacao.shape}, Teste: {teste.shape}')

### ================================================ ###
###   Plotando Pontos de Treino, Teste e Validação   ###
### ================================================ ###

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Treino - Pontos temporais
ax = axes[0, 0]
ax.scatter(treino_geral.detach().cpu().numpy(), np.zeros_like(treino_geral.detach().cpu().numpy()), 
           c='blue', s=15, label='Interior', alpha=0.6)
ax.scatter(treino_ci.detach().cpu().numpy(), np.zeros_like(treino_ci.detach().cpu().numpy()), 
           c='red', s=30, label='Condição Inicial (t=0)', alpha=0.8, marker='x')
ax.set_xlabel('t (s)')
ax.set_ylabel('')
ax.set_title('Pontos de Treino - Domínio Temporal')
ax.set_xlim(t_min, t_max)
ax.set_ylim(-0.5, 0.5)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_yticks([])

# Validação - Pontos temporais
ax = axes[0, 1]
ax.scatter(validacao.detach().cpu().numpy(), np.zeros_like(validacao.detach().cpu().numpy()), 
           c='green', s=15, alpha=0.6)
ax.set_xlabel('t (s)')
ax.set_ylabel('')
ax.set_title('Pontos de Validação - Domínio Temporal')
ax.set_xlim(t_min, t_max)
ax.set_ylim(-0.5, 0.5)
ax.grid(True, alpha=0.3)
ax.set_yticks([])

# Teste - Pontos temporais
ax = axes[1, 0]
ax.scatter(teste.detach().cpu().numpy(), np.zeros_like(teste.detach().cpu().numpy()), 
           c='purple', s=15, alpha=0.6)
ax.set_xlabel('t (s)')
ax.set_ylabel('')
ax.set_title('Pontos de Teste - Domínio Temporal')
ax.set_xlim(t_min, t_max)
ax.set_ylim(-0.5, 0.5)
ax.grid(True, alpha=0.3)
ax.set_yticks([])

# Histograma da distribuição dos pontos
ax = axes[1, 1]
ax.hist(treino_geral.detach().cpu().numpy(), bins=30, alpha=0.5, label='Treino', color='blue')
ax.hist(validacao.detach().cpu().numpy(), bins=30, alpha=0.5, label='Validação', color='green')
ax.hist(teste.detach().cpu().numpy(), bins=30, alpha=0.5, label='Teste', color='purple')
ax.set_xlabel('t (s)')
ax.set_ylabel('Frequência')
ax.set_title('Distribuição dos Pontos')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('MixFunn/DampedHO/Output/1_PreProcessing/Images/InputData.png', dpi=150)

### ======================================== ###
###   Visualizando Solução de Referência     ###
### ======================================== ###

# Grid temporal fino para visualização
t_ref = np.linspace(t_min, t_max, 1000)
x_ref, _ = solucao_analitica(t_ref, x0, v0, omega_0, gamma)

# Plotar apenas posição x(t)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_ref, x_ref, 'b-', linewidth=2, label='Solução Analítica')
ax.scatter(treino.detach().cpu().numpy(), treino_x.detach().cpu().numpy(), 
           c='red', s=10, alpha=0.3, label='Pontos de Treino')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('t (s)')
ax.set_ylabel('x(t) (m)')
ax.set_title(f'Oscilador Harmônico Amortecido - Posição\n' + 
             f'M = {M:.2f} kg, B = {B:.2f} kg/s, K = {K:.2f} N/m, Regime: {regime}')
ax.set_xlim(t_min, t_max)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('MixFunn/DampedHO/Output/1_PreProcessing/Images/ReferenceSolution.png', dpi=150)

### ======================================== ###
###              Salvando Dados              ###
### ======================================== ###

# Tensores t (tempo)
tc.save(treino, 'MixFunn/DampedHO/Output/1_PreProcessing/Data/treino.pt')
tc.save(validacao, 'MixFunn/DampedHO/Output/1_PreProcessing/Data/validacao.pt')
tc.save(teste, 'MixFunn/DampedHO/Output/1_PreProcessing/Data/teste.pt')

# Tensores x(t) - posição
tc.save(treino_x, 'MixFunn/DampedHO/Output/1_PreProcessing/Data/treino_x.pt')
tc.save(validacao_x, 'MixFunn/DampedHO/Output/1_PreProcessing/Data/validacao_x.pt')
tc.save(teste_x, 'MixFunn/DampedHO/Output/1_PreProcessing/Data/teste_x.pt')

# Metadata
metadata = {
    't_min': t_min,
    't_max': t_max,
    'n': n,
    'n_ci': n_ci,
    'n_treino': n_treino,
    'n_validacao': n_validacao,
    'n_teste': n_teste,
    'M': M,
    'B': B,
    'K': K,
    'x0': x0,
    'v0': v0,
    'regime': regime,
}
tc.save(metadata, 'MixFunn/DampedHO/Output/1_PreProcessing/Data/metadata.pt')

logger.info('Process Finished.')
