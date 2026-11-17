
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import criar_modelo

ABORDAGENS = ("mlp", "cnn")


def acuracia(modelo, loader, device):
    """Fração de imagens do `loader` classificadas corretamente."""
    modelo.eval()  #modo de avaliação (sem dropout)
    acertos = 0
    with torch.no_grad():  #sem gradiente
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            #argmax: índice do maior logit = dígito previsto; compara com o real
            acertos += (modelo(xb).argmax(dim=1) == yb).sum().item()
    return acertos / len(loader.dataset)


def treinar(abordagem, args, loaders, device):
    """Treina um modelo e devolve (acurácia de validação, acurácia de teste)."""
    train_loader, val_loader, test_loader = loaders
    torch.manual_seed(args.seed)  # mesma semente para as duas abordagens
    modelo = criar_modelo(abordagem, args.emb_dim, args.hidden).to(device)
    #adam ajusta todos os parâmetros: encoder e tabela de embeddings das classes
    otimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)

    print(f"\n=== abordagem: {abordagem} ===")
    for epoca in range(args.epochs):  
        modelo.train()
        for xb, yb in train_loader:  
            xb, yb = xb.to(device), yb.to(device)
            otimizador.zero_grad()  #zera os gradientes do lote anterior
            #cross-entropy compara os logits com o dígito real
            #backward() calcula os gradientes; step() atualiza os parâmetros
            F.cross_entropy(modelo(xb), yb).backward()
            otimizador.step()
        #ao fim de cada época, mede a acurácia em dados que o modelo não treinou
        print(f"época {epoca + 1}: acurácia de validação {acuracia(modelo, val_loader, device):.4f}")

    acc_val = acuracia(modelo, val_loader, device)
    acc_teste = acuracia(modelo, test_loader, device)  

    #salva os pesos e os hiperparâmetros necessários para recriar o modelo depois
    out = Path(args.out_dir) / f"{abordagem}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": modelo.state_dict(), "hidden": args.hidden, "emb_dim": args.emb_dim}, out
    )
    return acc_val, acc_teste


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--approach", choices=[*ABORDAGENS, "all"], default="all")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128, help="camada oculta do MLP")
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    #usa GPU se houver, se não, CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #ToTensor converte a imagem em tensor float com pixels em [0, 1] e formato (1, 28, 28)
    tf = transforms.ToTensor()
    #o MNIST é baixado automaticamente para `data_dir` na primeira execução
    treino_full = datasets.MNIST(args.data_dir, train=True, download=True, transform=tf)
    teste = datasets.MNIST(args.data_dir, train=False, download=True, transform=tf)
    #separa 50 mil imagens para treinar e 10 mil para validar
    treino, val = random_split(
        treino_full, [50_000, 10_000], generator=torch.Generator().manual_seed(args.seed)
    )
    loaders = (
        DataLoader(treino, batch_size=args.batch_size, shuffle=True),  #embaralha a cada época
        DataLoader(val, batch_size=1000),
        DataLoader(teste, batch_size=1000),
    )

    #treina uma ou as duas abordagens e guarda os resultados para a tabela final
    escolhidas = ABORDAGENS if args.approach == "all" else (args.approach,)
    resultados = {a: treinar(a, args, loaders, device) for a in escolhidas}

    print("\nResumo (acurácia):")
    print(f"{'abordagem':<12}{'validação':>10}{'teste':>10}")
    for a, (v, t) in resultados.items():
        print(f"{a:<12}{v:>10.4f}{t:>10.4f}")


if __name__ == "__main__":
    main()
