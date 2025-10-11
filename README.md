# TCC

Rodar tudo com sbatch run.sh {FILEPATH}

## Problmeas resolvidos

### Equação de Burgers

$$\left\{\begin{array}{l}
u_t + u u_x - (0.01/\pi) u_{xx} = 0, \ x \in [-1,1], \ t \in [0,1] \\
u(x,0) = -\sin(\pi x) \\
u(-1,t) = u(1,t) = 0
\end{array}\right.$$

#### Ref: https://arxiv.org/pdf/1711.10561


### Oscilador Harmônico

$$\left\{\begin{array}{l}
\ddot{x} + \frac{k}{m}x = 0 \\
x(0) = x_0 \\
\dot{x}(0) = v_0
\end{array}\right.$$

Com solução analitica:

$$x(t) = x_0 \cos\left(\sqrt{\frac{k}{m}} t\right) + v_0 \sqrt{\frac{m}{k}} \sin\left(\sqrt{\frac{k}{m}} t\right)$$

#### Ref: https://github.com/Coffee4MePlz/Notebooks_NN_Physics/blob/Ver%C3%A3o-PT-br/05-Exemplo%202%20OHS.ipynb

#### Ref: https://github.com/Coffee4MePlz/Notebooks_NN_Physics/blob/Ver%C3%A3o-PT-br/04-Exemplo%201%20Pendulo.ipynb