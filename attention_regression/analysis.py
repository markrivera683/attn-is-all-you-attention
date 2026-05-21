import torch
from metrics import matrix_cosine_similarity, mse
from models import DotProdAttention, WhitenedDotProdAttention, LinearRegression, LearnBilinear, Attention
from data import SynDataset, synthetic_data
from train import train_Bilinear, train_Attn


def sample_memory_query_batch(X: torch.Tensor,
                              Y: torch.Tensor,
                              memory_size: int=128,
                              query_size: int=64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = X.shape[0]
    midx = torch.randint(0, N, (memory_size,))
    qidx = torch.randint(0, N, (query_size,))

    Xm, Ym = X[midx], Y[midx]
    Xq, Yq = X[qidx], Y[qidx]

    return Xm, Ym, Xq, Yq


def evaluate_model(model, X, Y, num_episodes=100, memory_size=128,query_size=64):
    losses = []

    model.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            Xm, Ym, Xq, Yq = sample_memory_query_batch(X, Y, memory_size, query_size)
            Y_pred = model(Xq, Xm, Ym)
            loss = torch.nn.functional.mse_loss(Y_pred, Yq)
            losses.append(loss)

    return sum(losses) / len(losses)



X, Y = synthetic_data()

dot = DotProdAttention()
white = WhitenedDotProdAttention(20, 1e-3)
linr = LinearRegression(20, 1e-3)

bilinear = LearnBilinear(20)
attention = Attention(20)

trained_bilinear, loss_log_bi = train_Bilinear(X, Y, 20, 128, 64, 1000, 1, 1e-3, 20)
trained_atten, loss_log_atten = train_Attn(X, Y, 20, 128, 64, 1000, 10, 1e-3, 20)

print("dot: ", evaluate_model(dot, X, Y, 100))
print("white: ", evaluate_model(white, X, Y, 100))
print("linr: ", evaluate_model(linr, X, Y, 100))
print("bilinear: ", evaluate_model(bilinear, X, Y, 100))
print("attention: ", evaluate_model(attention, X, Y, 100))

print("\n\n")

print(loss_log_bi)

print("\n\n")

print(loss_log_atten)