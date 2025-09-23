# TCC

## Setup Jupyter no Windsurf

### Tutorial: Instalar extensão Jupyter compatível com Windsurf

#### Pré-requisitos
- Windsurf instalado
- Python com pyenv configurado
- Ambiente virtual criado

#### Passo 1: Configurar ambiente Python
```bash
# Ativar ambiente virtual
pyenv activate venv

# Instalar Jupyter
pip install jupyter ipykernel

# Registrar kernel do ambiente virtual
/home/nicobico/.pyenv/versions/3.13.3/envs/venv/bin/python -m ipykernel install --user --name=venv-tcc --display-name="Python (venv-tcc)"
```

#### Passo 2: Baixar e modificar extensão Jupyter

```bash
# Baixar extensão original
curl -L "https://marketplace.visualstudio.com/_apis/public/gallery/publishers/ms-toolsai/vsextensions/jupyter/latest/vspackage" -o jupyter.vsix

# Descompactar (pode estar em gzip)
mv jupyter.vsix jupyter.vsix.gz
gunzip jupyter.vsix.gz

# Extrair conteúdo
mkdir jupyter-ext && cd jupyter-ext && unzip ../jupyter.vsix

# Modificar compatibilidade no package.json
python3 -c "
import json
with open('extension/package.json', 'r') as f:
    data = json.load(f)
data['engines']['vscode'] = '^1.74.0'
with open('extension/package.json', 'w') as f:
    json.dump(data, f, indent=2)
print('Modificado engines.vscode para: ^1.74.0')
"

# Reempacotar extensão
zip -r ../jupyter-modified.vsix .

# Voltar ao diretório anterior
cd ..
```

#### Passo 3: Instalar extensão modificada
```bash
# Instalar no Windsurf
windsurf --install-extension jupyter-modified.vsix
```

#### Passo 4: Limpeza
```bash
# Remover arquivos temporários
rm -rf jupyter-ext jupyter.vsix jupyter-modified.vsix
```

#### Verificação
```bash
# Verificar extensões instaladas
windsurf --list-extensions | grep jupyter
```

Deve mostrar:
- ms-toolsai.jupyter
- ms-toolsai.jupyter-keymap
- ms-toolsai.vscode-jupyter-cell-tags
- ms-toolsai.vscode-jupyter-slideshow

#### Como usar
1. Abra qualquer arquivo `.ipynb` no Windsurf
2. Selecione o kernel "Python (venv-tcc)"
3. Execute células com Shift+Enter

#### Solução de problemas

**Erro de arquivo corrompido:**
```bash
# Verificar se o arquivo foi baixado corretamente
file jupyter.vsix

# Se necessário, usar wget ao invés de curl
wget "https://marketplace.visualstudio.com/_apis/public/gallery/publishers/ms-toolsai/vsextensions/jupyter/latest/vspackage" -O jupyter.vsix
```

**Extensão não compatível:**
- Certifique-se de que modificou o `engines.vscode` no package.json
- Verifique se reempacotou corretamente com zip

**Kernel não aparece:**
- Execute o comando de registro do kernel novamente
- Reinicie o Windsurf após instalar a extensão