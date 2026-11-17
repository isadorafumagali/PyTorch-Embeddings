
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from model import criar_modelo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--approach", choices=["mlp", "cnn"], default="cnn")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--n", type=int, default=2000, help="número de imagens a projetar")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    #recria o modelo com os mesmos hiperparâmetros do treino e carrega os pesos salvos
    #por train.py 
    ckpt = torch.load(Path(args.out_dir) / f"{args.approach}.pt", map_location="cpu")
    modelo = criar_modelo(args.approach, ckpt["emb_dim"], ckpt["hidden"])
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()

    #pega as primeiras `n` imagens do conjunto de teste e seus dígitos reais
    teste = datasets.MNIST(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    X = torch.stack([teste[i][0] for i in range(args.n)])
    y = torch.tensor([teste[i][1] for i in range(args.n)])

    #embedding de cada imagem: um vetor de `emb_dim` dimensões (ex.: 32)
    with torch.no_grad():
        emb = modelo.embed(X).numpy()

    #t-SNE reduz de 32 dimensões para 2, tentando manter próximos os pontos que
    #eram próximos no espaço original... assim dá para desenhar os embeddings
    pontos = TSNE(n_components=2, random_state=0).fit_transform(emb)


    plt.figure(figsize=(7, 6))
    sc = plt.scatter(pontos[:, 0], pontos[:, 1], c=y, cmap="tab10", s=8)
    plt.legend(*sc.legend_elements(), title="dígito", loc="best", fontsize=8)
    plt.title(f"Embeddings das imagens: {args.approach.upper()} (t-SNE)")
    plt.tight_layout()

    out = Path(args.out_dir) / f"embeddings_{args.approach}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"figura salva em {out}")


if __name__ == "__main__":
    main()
