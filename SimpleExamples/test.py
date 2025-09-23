import matplotlib.pyplot as plt
import numpy as np
import torch as tc

from torch import nn
from torch.optim.lr_scheduler import StepLR


        


# Definindo a rede neural 
net = GravityNet(neuronio=1,M=1,activation=nn.Tanh())

# Loss e otimizador
optimizer = tc.optim.Adam(net.parameters(), lr=0.001) # definindo o otimizador 

# Criando um scheduler para diminuir o learning rate a cada 'step_size' épocas
scheduler = StepLR(optimizer, step_size=10000, gamma=0.9)


# loss
LOSS = []

# Treinamento
for epoch in range(5000):
    omega = net(t_train)
    # Se observamos a variavel "gravidade" é um vetor, com tamanho igual a t_train
    # para utilizar como parametro ele deve ser apenas um numero
    # Portando iremos tomar a media desse vetor para usar como gravidade.
    w = tc.mean(omega)

    #theta_redeneural= x_0*tc.cos(tc.sqrt(g/comprimento_pendulo)*t_train)
    theta_NN = x_0*tc.cos(w*t_train)  
    loss1 = tc.mean((theta_train-theta_NN)**2)

    # podemos impor que, independentemente do input, o parametro deve ser igual
    loss2 = tc.mean(abs(omega-w))
    loss = loss1 + loss2

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    LOSS.append(loss.item())

gravidade = net(t_train) # Rede neural
freq = gravidade.mean().detach().numpy()
print("Valor da aceleração da gravidade",freq**2*comprimento_pendulo)