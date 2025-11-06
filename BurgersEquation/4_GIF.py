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
import glob
import shutil
from PIL import Image

### ======================================== ###
###          Configuração Inicial            ###
### ======================================== ###

# Escolher dados: validacao(1) ou teste(2)
DATASET = 2  # 1 = validação, 2 = teste

# Configurações do GIF
FIRST_SECTION_DURATION = 2.0  # Primeiros 2 segundos (epochs até ~5000)
REMAINING_DURATION = 13.0      # Próximos 13 segundos
LAST_FRAME_DURATION = 4000     # Duração do último frame em milissegundos (4 segundos)
GIF_LOOP = 0  # 0 = loop infinito, 1 = uma vez, etc.
THRESHOLD_EPOCH = 5000  # Epoch que separa primeira seção da segunda

# Configurações de seleção de modelos
SELECTION_RATIO = 0.25  # ~25% dos modelos serão selecionados
EARLY_CONCENTRATION = 0.70  # 70% da probabilidade concentrada nos primeiros 10%

### ======================================== ###

# Configurar logging
log_dir = 'BurgersEquation/Output/4_GIF'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.join(log_dir, 'Images'), exist_ok=True)
temp_dir = os.path.join(log_dir, 'Temp')

# Criar pasta temporária se não existir (não limpar para preservar imagens)
os.makedirs(temp_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'console.log')
if os.path.exists(log_file):
    os.remove(log_file)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
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

logger.info("Carregando dados de pré-processamento...")

validacao = tc.load('BurgersEquation/Output/1_PreProcessing/Data/validacao.pt', map_location=device)
teste = tc.load('BurgersEquation/Output/1_PreProcessing/Data/teste.pt', map_location=device)
validacao_u = tc.load('BurgersEquation/Output/1_PreProcessing/Data/validacao_u.pt', map_location=device)
teste_u = tc.load('BurgersEquation/Output/1_PreProcessing/Data/teste_u.pt', map_location=device)

metadata1 = tc.load('BurgersEquation/Output/1_PreProcessing/Data/metadata.pt')
x_min = metadata1['x_min']
x_max = metadata1['x_max']
t_min = metadata1['t_min']
t_max = metadata1['t_max']

metadata2 = tc.load('BurgersEquation/Output/2_Train/Data/metadata.pt')
hidden_layers = metadata2['hidden_layers']
neurons_per_layer = metadata2['neurons_per_layer']

# Seleção de dados
pts_comp = validacao if DATASET == 1 else teste
u_comp_ref = validacao_u if DATASET == 1 else teste_u
label_comp = 'Validação' if DATASET == 1 else 'Teste'

logger.info(f"Dataset selecionado: {label_comp}")

### ======================================== ###
###          Encontrando Modelos             ###
### ======================================== ###

logger.info("Buscando modelos salvos...")

# Encontrar todos os arquivos de modelo
model_files = glob.glob('BurgersEquation/Output/2_Train/Data/PINN_state_*.pt')

# Filtrar apenas os que têm número (excluir final, metadata, etc)
model_epochs = []
for file in model_files:
    basename = os.path.basename(file)
    if basename.startswith('PINN_state_') and basename.endswith('.pt'):
        epoch_str = basename.replace('PINN_state_', '').replace('.pt', '')
        if epoch_str.isdigit():
            model_epochs.append(int(epoch_str))

# Ordenar por epoch
model_epochs.sort()

logger.info(f"Encontrados {len(model_epochs)} modelos (epochs: {model_epochs[0]} a {model_epochs[-1]})")

### ======================================== ###
###     Seleção com Distribuição Exponencial ###
### ======================================== ###

logger.info(f"Aplicando distribuição exponencial para selecionar ~{SELECTION_RATIO*100:.0f}% dos modelos...")
logger.info(f"Concentração: {EARLY_CONCENTRATION*100:.0f}% nos primeiros 10% dos epochs")

# Normalizar epochs para [0, 1]
min_epoch = model_epochs[0]
max_epoch = model_epochs[-1]
epoch_range = max_epoch - min_epoch

# Calcular probabilidade de seleção para cada modelo
# Queremos 70% da probabilidade nos primeiros 10% (x < 0.1)
# Usar função: p(x) = a * exp(-k*x) para x < 0.1, depois continua exponencial

selected_epochs = []
for epoch in model_epochs:
    # Normalizar posição [0, 1]
    x = (epoch - min_epoch) / epoch_range if epoch_range > 0 else 0
    
    # Probabilidade de seleção com concentração nos primeiros 10%
    if x < 0.1:
        # Alta probabilidade nos primeiros 10% (ajustado para 70% total)
        prob = 0.95  # ~95% de chance de pegar nos primeiros 10%
    else:
        # Decai exponencialmente após 10%
        # Ajustar k para que o resto dos 30% seja distribuído nos 90% restantes
        k = 3.5  # Fator de decaimento
        prob = 0.95 * np.exp(-k * (x - 0.1))
    
    # Ajustar probabilidade global para atingir ~25% de seleção
    prob = prob * (SELECTION_RATIO / 0.35)  # Fator de ajuste
    
    # Decidir se seleciona este modelo
    if np.random.random() < prob:
        selected_epochs.append(epoch)

# Sempre incluir o último modelo
if model_epochs[-1] not in selected_epochs:
    selected_epochs.append(model_epochs[-1])

selected_epochs.sort()

logger.info(f"Selecionados {len(selected_epochs)} modelos ({len(selected_epochs)/len(model_epochs)*100:.1f}%)")
logger.info(f"Epochs selecionados: {selected_epochs[0]} a {selected_epochs[-1]}")

# Calcular quantos estão nos primeiros 10%
first_10_percent_threshold = min_epoch + 0.1 * epoch_range
early_selected = sum(1 for e in selected_epochs if e <= first_10_percent_threshold)
logger.info(f"Modelos nos primeiros 10%: {early_selected}/{len(selected_epochs)} ({early_selected/len(selected_epochs)*100:.1f}%)")

# Encontrar índice onde epoch passa de THRESHOLD_EPOCH
split_idx = 0
for i, epoch in enumerate(selected_epochs):
    if epoch > THRESHOLD_EPOCH:
        split_idx = i
        break

logger.info(f"Primeira seção: frames 0-{split_idx} (epochs 0-{THRESHOLD_EPOCH})")
logger.info(f"Segunda seção: frames {split_idx}-{len(selected_epochs)-1} (epochs {THRESHOLD_EPOCH}-{selected_epochs[-1]})")

# Calcular durações por seção
first_section_frames = split_idx
second_section_frames = len(selected_epochs) - split_idx - 1  # -1 porque último frame tem duração especial

# Duração por frame na primeira seção (ms)
if first_section_frames > 0:
    first_duration = int((FIRST_SECTION_DURATION * 1000) / first_section_frames)
else:
    first_duration = 50  # fallback

# Duração por frame na segunda seção (ms)
if second_section_frames > 0:
    second_duration = int((REMAINING_DURATION * 1000) / second_section_frames)
else:
    second_duration = 50  # fallback

logger.info(f"Primeira seção: {first_section_frames} frames × {first_duration}ms = {first_section_frames * first_duration / 1000:.1f}s")
logger.info(f"Segunda seção: {second_section_frames} frames × {second_duration}ms = {second_section_frames * second_duration / 1000:.1f}s")
logger.info(f"Último frame: {LAST_FRAME_DURATION}ms = {LAST_FRAME_DURATION / 1000:.1f}s")
total_estimated = (first_section_frames * first_duration + second_section_frames * second_duration + LAST_FRAME_DURATION) / 1000
logger.info(f"Duração total estimada: {total_estimated:.1f}s")

### ======================================== ###
###      Calculando Solução Referência       ###
### ======================================== ###

logger.info("Calculando solução de referência...")

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

# Grid para interpolação das predições
nx_grid = 256
nt_grid = 256
x_grid_1d = np.linspace(x_min, x_max, nx_grid)
t_grid_1d = np.linspace(t_min, t_max, nt_grid)
t_grid, x_grid = np.meshgrid(t_grid_1d, x_grid_1d, indexing='ij')

# Interpolação da referência no grid
u_ref_grid = interp(np.stack([t_grid.flatten(), x_grid.flatten()], axis=1)).reshape(nt_grid, nx_grid)

# Tempos para perfis
t_perfis = [0.25, 0.5, 0.75]

### ======================================== ###
###         Gerando Imagens do GIF           ###
### ======================================== ###

logger.info(f"Gerando {len(selected_epochs)} imagens...")
start_time = perf_counter()
total_images = len(selected_epochs)
log_interval = max(1, total_images // 100)  # Log a cada 1%

for idx, epoch in enumerate(selected_epochs):
    # Carregar modelo
    model_file = f'BurgersEquation/Output/2_Train/Data/PINN_state_{epoch}.pt'
    pinn_state = tc.load(model_file, map_location=device)
    
    # Criar e carregar PINN
    f = PINN(structure=[2] + [neurons_per_layer] * hidden_layers + [1], dropout_ratio=0.0).to(device)
    f.load_state_dict(pinn_state)
    f.eval()
    
    # Predição da rede
    with tc.no_grad():
        u_comp_pred = f(pts_comp)
        
        # Criar tensor de pontos para predição no grid
        pts_grid = tc.tensor(np.stack([x_grid.flatten(), t_grid.flatten()], axis=1), dtype=tc.float64, device=device)
        u_pred_grid = f(pts_grid).cpu().numpy().reshape(nt_grid, nx_grid)
    
    # Calcular erro médio
    erro_medio = tc.mean((u_comp_pred - u_comp_ref)**2).item()
    
    # Plot
    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.5)
    
    # Título com 3 linhas (ordem: Comparação, Epoch, Erro)
    fig.text(0.5, 0.98, f'Comparação: Referência vs. PINN - {label_comp}', 
             ha='center', va='top', fontsize=14, weight='normal')
    fig.text(0.5, 0.96, f'Epoch: {epoch}', 
             ha='center', va='top', fontsize=13, weight='normal')
    fig.text(0.5, 0.94, f'Erro Médio Quadrático: {erro_medio:.2e}', 
             ha='center', va='top', fontsize=12, weight='normal')
    
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
    
    # Salvar imagem temporária com nome padronizado (importante para ordenação)
    temp_file = os.path.join(temp_dir, f'frame_{idx:05d}.png')
    plt.savefig(temp_file, bbox_inches='tight')
    plt.close(fig)
    
    # Limpar memória GPU e objetos Python
    del f, pinn_state, u_comp_pred, u_pred_grid, pts_grid, fig
    tc.cuda.empty_cache()
    
    # Log de progresso a cada 1%
    if (idx + 1) % log_interval == 0 or idx == 0 or idx == total_images - 1:
        elapsed = perf_counter() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (total_images - idx - 1)
        total_time = elapsed + remaining
        logger.info(f"{idx+1}/{total_images} imagens - Tempo: {int(elapsed)}s/{int(total_time)}s")

end_time = perf_counter()
logger.info(f"Todas as imagens geradas em {(end_time - start_time)/60:.2f} minutos")

### ======================================== ###
###              Criando GIF                 ###
### ======================================== ###

logger.info("Criando GIF a partir das imagens...")

# Coletar todos os frames
frame_files = sorted(glob.glob(os.path.join(temp_dir, 'frame_*.png')))

# Carregar todas as imagens
frames = []
for frame_file in frame_files:
    frames.append(Image.open(frame_file))

# Preparar durações dos frames (primeira seção + segunda seção + último frame)
durations = []
durations.extend([first_duration] * first_section_frames)
durations.extend([second_duration] * second_section_frames)
durations.append(LAST_FRAME_DURATION)

# Salvar como GIF
gif_file = os.path.join(log_dir, 'Images', f'PINN_Evolution_{label_comp}.gif')
frames[0].save(
    gif_file,
    save_all=True,
    append_images=frames[1:],
    duration=durations,
    loop=GIF_LOOP,
    optimize=False  # Não otimizar para manter qualidade
)

logger.info(f"GIF salvo em: {gif_file}")
logger.info(f"Total de frames: {len(frames)}")
total_duration = sum(durations) / 1000
logger.info(f"Duração total do GIF: {total_duration:.1f} segundos")

### ======================================== ###
###        Limpando Pasta Temporária         ###
### ======================================== ###

logger.info("Limpando pasta temporária...")
shutil.rmtree(temp_dir)
logger.info(f"Pasta temporária removida: {temp_dir}")

logger.info('Process Finished.')
