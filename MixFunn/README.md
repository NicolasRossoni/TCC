# MixFunn - Problemas Resolvidos

Implementação de Physics-Informed Neural Networks (PINNs) usando a biblioteca MixFunn para resolver equações diferenciais.

## Referências MixFunn

### Paper: https://arxiv.org/pdf/2503.22528

### Repositório: https://github.com/tiago939/MixFunn/

---

## Problemas Resolvidos

### Oscilador Harmônico Amortecido (Damped Harmonic Oscillator)

$$\left\{\begin{array}{l}
M\ddot{x}(t) + B\dot{x}(t) + Kx(t) = 0, \quad t \in [0, T] \\
x(0) = x_0 \\
\dot{x}(0) = v_0
\end{array}\right.$$

Onde:
- $M$ é a massa (kg)
- $B$ é o coeficiente de amortecimento (kg/s)
- $K$ é a constante da mola (N/m)
- $x_0$ é a posição inicial (m)
- $v_0$ é a velocidade inicial (m/s)

#### Ref: https://arxiv.org/pdf/2503.22528 (Seção IV.A - Harmonic Oscillator)

#### Ref: https://github.com/tiago939/MixFunn/tree/main/examples/damped%20harmonic%20oscillator

---

## Estrutura dos Códigos

Cada problema segue a mesma estrutura:

- `1_PreProcessing.py` - Geração de dados de treino/validação/teste e solução de referência
- `2_Train.py` - Treinamento da rede PINN com MixFunn
- `3_Validate.py` - Validação e comparação com solução analítica/numérica
