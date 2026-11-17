
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """Imagem (1, 28, 28) -> vetor de dimensão `emb_dim`, com um MLP."""

    def __init__(self, emb_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            #(batch, 1, 28, 28) -> (batch, 784): "estica" a imagem em um vetor.
            #o MLP perde a noção de vizinhança entre pixels.
            nn.Flatten(),
            #camada oculta: combina os 784 pixels em `hidden` (128) características.
            nn.Linear(28 * 28, hidden),
            #não-linearidade; sem ela, as duas camadas lineares equivaleriam a uma só.
            nn.ReLU(),
            #projeta no espaço de embedding: saída (batch, emb_dim).
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNEncoder(nn.Module):
    """Imagem (1, 28, 28) -> vetor de dimensão `emb_dim`, com uma CNN."""

    def __init__(self, emb_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            #32 filtros 3x3 deslizam pela imagem detectando padrões locais
            #(bordas, curvas). padding=1 mantém o tamanho 28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            #Max pooling 2x2: reduz a resolução pela metade (28x28 -> 14x14),
            #guardando o maior valor de cada bloco
            nn.MaxPool2d(2),
            #64 filtros combinam os padrões simples em padrões mais complexos
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            #nova redução: 14x14 -> 7x7
            nn.MaxPool2d(2),
        )
        #após as convoluções temos 64 mapas de 7x7 = 3136 valores por imagem
        #esta camada linear os comprime no vetor de embedding
        self.proj = nn.Linear(64 * 7 * 7, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #flatten(1) mantém a dimensão do batch e achata o restante
        return self.proj(self.conv(x).flatten(1))


class EmbeddingClassifier(nn.Module):
    """Encoder + tabela `nn.Embedding` de classes, no mesmo espaço vetorial."""

    def __init__(self, encoder: nn.Module, num_classes: int = 10, emb_dim: int = 32, scale: float = 10.0):
        super().__init__()
        self.encoder = encoder
        #tabela de embeddings: uma linha (vetor aprendido) para cada dígito
        self.class_embedding = nn.Embedding(num_classes, emb_dim)
        #fator que amplia as similaridades (que ficam entre -1 e 1). Sem ele, os logits
        #seriam muito próximos e o softmax/cross-entropy aprenderia devagar
        self.scale = scale

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding da imagem, normalizado para ter comprimento 1."""
        return F.normalize(self.encoder(x), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #embeddings das 10 classes, também normalizados: formato (10, emb_dim)
        classes = F.normalize(self.class_embedding.weight, dim=-1)
        #com vetores de comprimento 1, o produto escalar é a similaridade de cosseno
        #(batch, emb_dim) @ (emb_dim, 10) -> (batch, 10): um "score" por dígito
        #a classe prevista é a de maior score (argmax), e o treino usa estes logits
        return self.scale * self.embed(x) @ classes.T


def criar_modelo(abordagem: str, emb_dim: int = 32, hidden: int = 128) -> EmbeddingClassifier:
    """Monta o modelo completo para a abordagem escolhida ("mlp" ou "cnn")."""
    if abordagem == "mlp":
        encoder = MLPEncoder(emb_dim, hidden)
    elif abordagem == "cnn":
        encoder = CNNEncoder(emb_dim)
    else:
        raise ValueError(f"abordagem desconhecida: {abordagem}")
    return EmbeddingClassifier(encoder, emb_dim=emb_dim)
