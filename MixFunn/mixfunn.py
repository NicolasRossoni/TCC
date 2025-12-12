#### Importando Bibliotecas
import torch
import torch.nn as nn
import torch.nn.functional as F

### ======================================== ###
###       Funções de Ativação Básicas       ###
### ======================================== ###

class Sin(torch.nn.Module):
    def forward(self, x):
        f = torch.sin(x)
        return f


class Cos(torch.nn.Module):
    def forward(self, x):
        f = torch.cos(x)
        return f


class Exp(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(x)
        return f


class ExpN(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(-x)
        return f


class ExpAbs(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(-0.01 * abs(x))
        return f


class ExpAbsP(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(0.01 * abs(x))
        return f


class Sqrt(torch.nn.Module):
    """Aproximação da raiz quadrada para evitar instabilidade numérica"""
    def __init__(self):
        super(Sqrt, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        f = (0.01 + self.relu(x)) ** 0.5
        return f


class Log(torch.nn.Module):
    """Aproximação do logaritmo para evitar instabilidade numérica"""
    def __init__(self):
        super(Log, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        f = torch.log(0.1 + self.relu(x))
        return f


class Id(torch.nn.Module):
    def forward(self, x):
        f = x
        return f


class Tanh(torch.nn.Module):
    def forward(self, x):
        f = torch.tanh(x)
        return f

# Lista de funções disponíveis para os neurônios de função mista
functions = [Sin(), Cos(), ExpAbs(), ExpAbsP(), Sqrt(), Log(), Id()]
L = len(functions)


### ======================================== ###
###       Neurônios de Segunda Ordem        ###
### ======================================== ###

class Quad(nn.Module):
    """
    Neurônios quadráticos que capturam interações entre variáveis de entrada
    
    Args:
        n_in: dimensão de entrada
        n_out: dimensão de saída
        second_order: se True, adiciona termos quadráticos; se False, usa apenas transformação linear
    """
    def __init__(self, n_in, n_out, second_order=True):
        super(Quad, self).__init__()

        self.second_order = second_order

        if not second_order:
            # Neurônios de primeira ordem (apenas transformação linear)
            self.linear = nn.Linear(n_in, n_out)
        else:
            L = int(n_in * (n_in - 1) / 2)
            self.linear = nn.Linear(L + n_in, n_out) # Linear + Termos quadráticos
            self.ids = torch.triu_indices(n_in, n_in, 1) # Indices da diagonal superior

    def forward(self, x):
        if self.second_order:
            # Calcula termos quadráticos: produtos entre todas as variáveis de entrada
            x2 = x[:, :, None] @ x[:, None, :]
            x2 = x2[:, self.ids[0], self.ids[1]] # Apenas diagonal superior
            x = torch.cat((x, x2), axis=1)

        x = self.linear(x)
        return x


### ======================================== ###
###     Neurônios de Função Mista (MixFun)  ###
### ======================================== ###

class Mixfun(nn.Module):
    """
    Neurônios que combinam múltiplas funções não-lineares parametrizadas
    para melhorar a flexibilidade representacional
    
    Args:
        n_in: dimensão de entrada
        n_out: dimensão de saída
        normalization_function: força cada neurônio a escolher uma única função
        normalization_neuron: força cada neurônio a ter uma função diferente dos outros
        p_drop: taxa de dropout (False = sem dropout)
        second_order_input: se True, usa neurônios quadráticos na entrada
        second_order_function: se True, usa produtos de segunda ordem entre funções
        temperature: parâmetro de temperatura para normalização softmax
    """
    def __init__(
        self,
        n_in,
        n_out,
        normalization_function=False,
        normalization_neuron=False,
        p_drop=False,
        second_order_input=False,
        second_order_function=False,
        temperature=1.0,
    ):
        super(Mixfun, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.p_drop = p_drop

        if p_drop:
            self.dropout = nn.Dropout(p=p_drop)
        self.second_order_function = second_order_function
        self.temperature = temperature

        # Projeção de primeira ordem
        self.project1 = Quad(n_in, L * n_out, second_order=second_order_input)

        # Calcula número total de funções (primeira ordem + segunda ordem se aplicável)
        self.length = int(L * (L + 1) / 2) * int(second_order_function)
        self.F = L + self.length

        if second_order_function:
            # Projeções de segunda ordem para capturar interações entre funções
            self.project21 = Quad(
                n_in, L * n_out, second_order=second_order_function
            )
            self.project22 = Quad(
                n_in, L * n_out, second_order=second_order_function
            )
            self.ids = torch.triu_indices(L, L, 0)

        # Parâmetros de normalização
        self.normalization_function = normalization_function
        self.normalization_neuron = normalization_neuron

        # Inicialização dos pesos conforme tipo de normalização
        if not (normalization_function or normalization_neuron):
            self.p_raw = nn.Parameter(torch.randn(n_out, self.F)) # Inicializa aleatoriamente

        if normalization_function:
            self.p_fun = nn.Parameter(torch.ones(n_out, self.F)) # Normaliza funções
        else:
            self.p_fun = None

        if normalization_neuron:
            self.p_neuron = nn.Parameter(torch.ones(n_out, self.F)) # Normaliza neurônios
        else:
            self.p_neuron = None

        if normalization_function or normalization_neuron:
            self.amplitude = nn.Parameter(torch.randn(n_out)) # Não deixa o softmax limitar os valores

    def __project_and_stack(self, x, projection):
        """
        Aplica projeção e empilha resultados de todas as funções de ativação
        
        Args:
            x: tensor de entrada
            projection: camada de projeção (Quad)
        
        Returns:
            tensor com saídas de todas as funções empilhadas
        """
        y = projection(x).reshape((x.shape[0], self.n_out, L)) # Reshape [batch, neurônios, funções]
        
        # Aplica cada função de ativação
        y = [fun(y[:, :, i]) for i, fun in enumerate(functions)]
        
        y = torch.stack(y, dim=1).reshape((x.shape[0], self.n_out, L))
        return y

    def forward(self, x):
        """
        Propagação forward da camada MixFun
        
        Args:
            x: tensor de entrada
        
        Returns:
            tensor de saída combinando múltiplas funções de ativação
        """
        if self.second_order_function:
            # Funções de primeira ordem
            x1 = self.__project_and_stack(x, self.project1)

            # Funções de segunda ordem (produtos entre funções)
            x2_1 = self.__project_and_stack(x, self.project21)
            x2_2 = self.__project_and_stack(x, self.project22)
            x2 = x2_1[:, :, :, None] @ x2_2[:, :, None, :]
            x2 = x2[:, :, self.ids[0], self.ids[1]]

            # Concatena funções de primeira e segunda ordem
            x = torch.cat((x1, x2), axis=2)
        else:
            # Apenas funções de primeira ordem
            x = self.__project_and_stack(x, self.project1)

        # Aplica dropout durante treinamento
        if self.p_drop and self.training:
            x = self.dropout(x)

        # Combinação das funções conforme tipo de normalização
        if not (self.normalization_function and self.normalization_neuron):
            return torch.sum(self.p_raw * x, axis=2)

        # Normalização por função (softmax sobre funções)
        if self.normalization_function:
            p_fun = F.softmax(-self.p_fun / self.temperature, dim=1)
        else:
            p_fun = 1.0

        # Normalização por neurônio (softmax sobre neurônios)
        if self.normalization_neuron:
            p_neuron = F.softmax(-self.p_neuron / self.temperature, dim=0)
        else:
            p_neuron = 1.0

        x = self.amplitude * torch.sum(p_neuron * p_fun * x, axis=2)
        return x