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
log_dir = 'BurgersEquation/Output/1_PreProcessing'
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

# Definindo domínio
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0

# Número de pontos
n_ci = 200   # Pra cada condição de contorno(teste)
n = 10000    # Pontos usados no treino, validação e teste

distribution = [0.6, 0.2, 0.2] # Distribuição entre treino, validação e teste

# Hipercubo Latino 2D (para (x,t) ∈ [(x_min, x_max), (t_min, t_max)])
def latin_hypercube(n_points):
    intervals = tc.linspace(0, 1, n_points + 1)
    points = tc.zeros(n_points, 2)
    for i in range(2):
        perm = tc.randperm(n_points)
        points[:, i] = intervals[:-1] + (intervals[1] - intervals[0]) * tc.rand(n_points)
        points[:, i] = points[perm, i]
    
    # Escalar para o domínio definido
    points[:, 0] = points[:, 0] * (x_max - x_min) + x_min # x
    points[:, 1] = points[:, 1] * (t_max - t_min) + t_min # t
    return points

### ================================== ###
###    Condições Iniciais/Contorno     ###
### ================================== ###

# Pontos (x,t)
ci1 = latin_hypercube(n_ci)     # (x,t) = (:,0)
ci2 = latin_hypercube(n_ci)     # (x,t) = (1,:)
ci3 = latin_hypercube(n_ci)     # (x,t) = (-1,:)

ci1[:,1] = tc.zeros(n_ci)           # t = 0
ci2[:,0] = tc.full((n_ci,), 1.0)    # x = 1
ci3[:,0] = tc.full((n_ci,), -1.0)   # x = -1

treino_ci = tc.concatenate([ci1, ci2, ci3]) # (x,t) -> shape: [3*n_ci, 2]

# Soluções u(x,t)
ci1_u = -tc.sin(tc.pi*ci1[:,0])     # u(x,0) -> shape: [n_ci]
ci2_u = tc.zeros(n_ci)              # u(1,t) -> shape: [n_ci]
ci3_u = tc.zeros(n_ci)              # u(-1,t) -> shape: [n_ci]

treino_ci_u = tc.concatenate([ci1_u, ci2_u, ci3_u]).unsqueeze(1).to(device) # (x,t) -> shape: [3*n_ci, 1]

### ==================================== ###
###  Dados de treino, validação e teste  ###
### ==================================== ###

n_treino = int(n*distribution[0])
n_validacao = int(n*distribution[1])
n_teste = int(n*distribution[2])

# Pontos (x,t) - treino
treino_geral = latin_hypercube(n_treino)               # (x,t) -> shape: [n_treino, 2]
treino = tc.cat([treino_ci, treino_geral]).to(device)  # (x,t) -> shape: [3*n_ci + n_treino, 2]
treino.requires_grad_(True)

# Pontos (x,t) - validação
validacao = latin_hypercube(n_validacao).to(device)     # (x,t) -> shape: [n_validacao, 2]
validacao.requires_grad_(False)

# Pontos (x,t) - teste
teste = latin_hypercube(n_teste).to(device)             # (x,t) -> shape: [n_teste, 2]
teste.requires_grad_(False)

### ========================================= ###
###   Solução Referencia(validação e teste)   ###
### ========================================= ###

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

def get_u_tensor(points):
    # points: tensor shape [N, 2] com (x, t)
    pts_np = points.detach().cpu().numpy()
    # scipy espera (t, x)
    pts_np = np.stack([pts_np[:,1], pts_np[:,0]], axis=1)
    u_np = interp(pts_np)
    return tc.tensor(u_np, dtype=tc.float64, device=points.device).unsqueeze(1)

treino_u = get_u_tensor(treino)             # u(x,t) -> shape: [3*n_ci + n_treino, 1]
validacao_u = get_u_tensor(validacao)       # u(x,t) -> shape: [n_validacao, 1]
teste_u = get_u_tensor(teste)               # u(x,t) -> shape: [n_teste, 1]

### ================================================ ###
###   Plotando Pontos de Treino, Teste e Validação   ###
### ================================================ ###
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# Treino (ocupa primeira linha, 2 colunas)
ax_treino = fig.add_subplot(gs[0, :])
ax_treino.scatter(treino_geral[:,1].detach().numpy(), treino_geral[:,0].detach().numpy(), c='blue', s=15, label='Interior', alpha=0.6)
ax_treino.scatter(ci1[:,1].detach().numpy(), ci1[:,0].detach().numpy(), c='red', label='Contorno', s=15, alpha=0.6)
ax_treino.scatter(ci2[:,1].detach().numpy(), ci2[:,0].detach().numpy(), c='red', s=15, alpha=0.6)
ax_treino.scatter(ci3[:,1].detach().numpy(), ci3[:,0].detach().numpy(), c='red', s=15, alpha=0.6)
ax_treino.set_xlabel('t')
ax_treino.set_ylabel('x')
ax_treino.set_title('Pontos de Treino')
ax_treino.set_xlim(t_min, t_max)
ax_treino.set_ylim(x_min, x_max)
ax_treino.legend(loc='upper right')
ax_treino.grid(True, alpha=0.3)

# Validação (segunda linha, coluna 1)
ax_val = fig.add_subplot(gs[1, 0])
ax_val.scatter(validacao[:,1].detach().cpu().numpy(), validacao[:,0].detach().cpu().numpy(), c='green', s=15, alpha=0.6)
ax_val.set_xlabel('t')
ax_val.set_ylabel('x')
ax_val.set_title('Pontos de Validação')
ax_val.set_xlim(t_min, t_max)
ax_val.set_ylim(x_min, x_max)
ax_val.grid(True, alpha=0.3)

# Teste (segunda linha, coluna 2)
ax_teste = fig.add_subplot(gs[1, 1])
ax_teste.scatter(teste[:,1].detach().cpu().numpy(), teste[:,0].detach().cpu().numpy(), c='purple', s=15, alpha=0.6)
ax_teste.set_xlabel('t')
ax_teste.set_ylabel('x')
ax_teste.set_title('Pontos de Teste')
ax_teste.set_xlim(t_min, t_max)
ax_teste.set_ylim(x_min, x_max)
ax_teste.grid(True, alpha=0.3)

plt.savefig('BurgersEquation/Output/1_PreProcessing/Images/InputData.png')

### ======================================== ###
###   Visualizando Solução de Referência     ###
### ======================================== ###

# Pontos treino + validação
pts_tv = tc.cat([treino_geral.cpu(), validacao.cpu()]).detach().cpu()
u_tv = get_u_tensor(pts_tv).detach().cpu().numpy().flatten()

# Grid para contorno
t_grid, x_grid = np.meshgrid(t_ref, x_ref, indexing='ij')

# Tempos para perfis
t_perfis = [0.25, 0.5, 0.75]

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.5)

# Mapa de calor principal
ax_main = fig.add_subplot(gs[0, :])
im = ax_main.contourf(t_grid, x_grid, u_ref, levels=100, cmap='jet', vmin=-1, vmax=1)
ax_main.set_xlabel('t')
ax_main.set_ylabel('x')
ax_main.set_title('Solução de Referência u(t,x) - com RK4')
ax_main.set_xlim(t_min, t_max)
ax_main.set_ylim(x_min, x_max)
cbar = plt.colorbar(im, ax=ax_main, pad=0.02, aspect=10, ticks=np.arange(-1, 1.25, 0.25))

# Perfis temporais
for i, t_val in enumerate(t_perfis):
    ax = fig.add_subplot(gs[1, i])
    idx_t = np.argmin(np.abs(t_ref - t_val))
    ax.plot(x_ref, u_ref[idx_t, :], 'b-', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_title(f't = {t_val:.2f}')
    ax.set_xlim(x_min, x_max)
    ax.grid(True, alpha=0.3)

plt.savefig('BurgersEquation/Output/1_PreProcessing/Images/ReferenceSolution.png')

### ======================================== ###
###              Salvando Dados              ###
### ======================================== ###

# Tensores (x,t)
tc.save(treino, 'BurgersEquation/Output/1_PreProcessing/Data/treino.pt')
tc.save(validacao, 'BurgersEquation/Output/1_PreProcessing/Data/validacao.pt')
tc.save(teste, 'BurgersEquation/Output/1_PreProcessing/Data/teste.pt')

# Tensores u(x,t)
tc.save(treino_ci_u, 'BurgersEquation/Output/1_PreProcessing/Data/treino_ci_u.pt')
tc.save(treino_u, 'BurgersEquation/Output/1_PreProcessing/Data/treino_u.pt') # Para o caso supervisionado
tc.save(validacao_u, 'BurgersEquation/Output/1_PreProcessing/Data/validacao_u.pt')
tc.save(teste_u, 'BurgersEquation/Output/1_PreProcessing/Data/teste_u.pt')

# Metadata
metadata = {
    'x_min': x_min,
    'x_max': x_max,
    't_min': t_min,
    't_max': t_max,
    'n': n,
    'n_ci': n_ci,
    'n_treino': n_treino,
    'n_validacao': n_validacao,
    'n_teste': n_teste,
}
tc.save(metadata, 'BurgersEquation/Output/1_PreProcessing/Data/metadata.pt')

logger.info('Process Finished.')