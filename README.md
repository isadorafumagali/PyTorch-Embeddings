# Reconhecimento de dígitos (MNIST) com PyTorch e Embeddings: MLP vs. CNN

Duas abordagens em PyTorch para o mesmo problema, classificar dígitos manuscritos do MNIST,
treinadas e comparadas lado a lado. Ambas usam **embeddings**.

## Como funciona

- Um **encoder** transforma cada imagem 28×28 em um vetor de 32 dimensões (embedding da imagem).
- `nn.Embedding(10, 32)` guarda um vetor aprendido para cada dígito (0–9).
- O dígito previsto é o da classe com maior **similaridade de cosseno** com o embedding da imagem.

## As duas abordagens

A única diferença é o encoder:

| | Abordagem 1: `mlp` | Abordagem 2: `cnn` |
|---|---|---|
| Encoder | `Flatten → Linear(784,128) → ReLU → Linear(128,32)` | `Conv(1→32) → Pool → Conv(32→64) → Pool → Linear(3136,32)` |

## Autora 

Isadora Fumagali — @isadorafumagali