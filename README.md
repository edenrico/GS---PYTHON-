# GS---PYTHON-

EDUARDO ENRICO RM553179
MAX RM552645

A proposta é desenvolver uma Inteligência Artificial (IA) dedicada ao diagnóstico de tumores cerebrais por meio de imagens. O código disponibilizado inclui um banco de dados contendo imagens de pacientes com 
suspeita de câncer cerebral, apresentando tanto as imagens de pacientes quanto uma imagem de rótulo correspondente. Este conjunto de dados visa aprimorar a precisão da IA no diagnóstico de tumores cerebrais,
evidenciando o propósito e a intenção por trás do código.

Este é um projeto de grande relevância, considerando os avanços significativos que as tecnologias de IA têm proporcionado na área médica, especialmente no diagnóstico por imagem. Estamos disponíveis para 
auxiliar e fornecer orientações conforme necessário. Caso haja mais detalhes específicos ou requisitos adicionais, por favor, sinta-se à vontade para compartilhar. Estamos aqui para apoiar o 
desenvolvimento dessa iniciativa dedicada à melhoria do diagnóstico médico.




PASSO 1-

import requests: Esta linha importa a biblioteca requests, que é uma biblioteca popular em Python para fazer solicitações HTTP. Frequentemente, é utilizada para enviar solicitações HTTP para servidores web e receber respostas.

import torch: Esta linha importa a biblioteca PyTorch. PyTorch é uma biblioteca de código aberto para aprendizado de máquina, utilizada para tarefas como aprendizado profundo e redes neurais.

import torch.nn as nn: Esta linha importa especificamente o módulo de redes neurais (nn) do PyTorch. O módulo nn do PyTorch fornece ferramentas para construir e treinar redes neurais.

from torchvision import transforms: Aqui, o código importa o módulo transforms da biblioteca torchvision. torchvision é uma biblioteca do PyTorch para tarefas de visão computacional, e transforms fornece transformações comuns de imagens.

from PIL import Image: Esta linha importa o módulo Image da Biblioteca de Imagens Python (PIL) ou sua versão fork, Pillow. É comumente usada para abrir, manipular e salvar vários formatos de arquivo de imagem.

from io import BytesIO: Esta linha importa a classe BytesIO do módulo io. BytesIO é uma classe que fornece um buffer binário usando uma interface semelhante a bytes. Neste contexto, pode ser usado para manipular dados de imagem em memória.

import matplotlib.pyplot as plt: Esta linha importa o módulo pyplot da biblioteca matplotlib. matplotlib é uma biblioteca de plotagem em Python, e pyplot fornece uma interface conveniente para criar visualizações.

import numpy as np: Esta linha importa a biblioteca numpy e a renomeia como np. numpy é uma poderosa biblioteca para operações numéricas em Python, sendo frequentemente utilizada em computação científica e análise de dados.



PASSO 2-


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
Este código está implementando uma versão simplificada de uma rede neural do tipo U-Net para a detecção de tumores cerebrais em imagens médicas. 

A classe UNet herda da classe nn.Module do PyTorch, indicando que é um modelo PyTorch.
No método __init__, são definidas duas camadas convolucionais (conv1 e conv2).



PASSO 3-



def carregar_imagem(url):
    resposta = requests.get(url)
    imagem = Image.open(BytesIO(resposta.content))
    transformacao = transforms.Compose([transforms.ToTensor()])
    return transformacao(imagem)

A função carregar_imagem recebe uma URL, faz uma solicitação HTTP para obter a imagem, a converte para um objeto Image usando o PIL, e em seguida, aplica transformações para convertê-la em um tensor usando transforms.ToTensor().


PASSO 4-


def visualizar_resultados(imagem, mascara_verdadeira, predicao):
    # ... (código para plotar as imagens)

A função visualizar_resultados recebe uma imagem, uma máscara verdadeira e uma predição. Ela usa o matplotlib para exibir as três imagens em uma única figura.


PASSO 5-


def verificar_tumor(predicao, limiar=0.5):
    probabilidade_tumor = torch.sigmoid(predicao.mean()).item()
    return probabilidade_tumor > limiar

A função verificar_tumor recebe uma predição do modelo e um limiar (por padrão, 0.5). Ela calcula a probabilidade média da presença de um tumor usando a função sigmoid e retorna True se a probabilidade for maior que o limiar.


PASSO 6-


imagem_rotulo = carregar_imagem(url_rotulo)
imagem_paciente = carregar_imagem(url_paciente)

imagem_rotulo = nn.functional.interpolate(imagem_rotulo.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0).float()
imagem_paciente = nn.functional.interpolate(imagem_paciente.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0).float()

As imagens são carregadas usando as URLs fornecidas e, em seguida, são redimensionadas para o tamanho desejado (256x256 pixels).



PASSO 7-



modelo = UNet()
modelo.eval()

with torch.no_grad():
    predicao_rotulo = modelo(imagem_rotulo.unsqueeze(0))
    predicao_paciente = modelo(imagem_paciente.unsqueeze(0))

Uma instância do modelo U-Net é criada, definindo-o em modo de avaliação (eval). As predições são feitas para as imagens de rótulo e de paciente.



PASSO 8-



visualizar_resultados(imagem_paciente, np.zeros((256, 256)), predicao_paciente)
visualizar_resultados(imagem_rotulo, np.ones((256, 256)), predicao_rotulo)

tem_tumor_rotulo = verificar_tumor(predicao_rotulo)
tem_tumor_paciente = verificar_tumor(predicao_paciente)

print(f"Rótulo: {tem_tumor_rotulo}")
print(f"O paciente tem tumor?: {tem_tumor_paciente}")


Os resultados são visualizados usando a função visualizar_resultados para ambas as imagens. Em seguida, a presença de tumores é verificada usando a função verificar_tumor e os resultados são impressos.





