import torch
import wandb
from torch.nn import functional as F

from metrics import matrix_cosine_similarity, condition_number
from models import (
    DotProdAttention,
    WhitenedDotProdAttention,
    LinearRegression,
    LearnBilinear,
    Attention,
)
from data import synthetic_data
from train import train_Bilinear, train_Attn


CONFIG = {
    "seed": 42,
    "dim": 20,
    "lambda_reg": 1e-3,
    "learning_rate": 1e-3,
    "train_steps": 1000,
    "eval_episodes": 100,
    "memory_size": 128,
    "query_size": 64,
    "log_interval": 20,
    "device": "cpu",

    "use_wandb": false,
    "wandb_entity": "markrivera683-beijing-university-of-posts-and-communications",
    "wandb_project": "attention mechanism",
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_memory_query_batch(
    X: torch.Tensor,
    Y: torch.Tensor,
    memory_size: int,
    query_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    N = X.shape[0]

    idx_m = torch.randint(0, N, (memory_size,), device=X.device)
    idx_q = torch.randint(0, N, (query_size,), device=X.device)

    Xm, Ym = X[idx_m], Y[idx_m]
    Xq, Yq = X[idx_q], Y[idx_q]

    return Xm, Ym, Xq, Yq


def make_eval_batches(
    X: torch.Tensor,
    Y: torch.Tensor,
    num_episodes: int,
    memory_size: int,
    query_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [
        sample_memory_query_batch(X, Y, memory_size, query_size)
        for _ in range(num_episodes)
    ]


def evaluate_model(
    model: torch.nn.Module,
    model_name: str,
    eval_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    use_wandb: bool = False,
) -> dict[str, float]:
    model.eval()
    losses = []

    run = None
    if use_wandb:
        run = wandb.init(
            entity=CONFIG["wandb_entity"],
            project=CONFIG["wandb_project"],
            name=model_name,
            config=CONFIG,
            reinit=True,
        )

    with torch.no_grad():
        for epi, (Xm, Ym, Xq, Yq) in enumerate(eval_batches):
            Y_pred = model(Xq, Xm, Ym)
            loss = F.mse_loss(Y_pred, Yq).item()
            losses.append(loss)

            if run is not None:
                run.log({
                    "episode": epi,
                    "loss": loss,
                })

    losses_tensor = torch.tensor(losses)

    avg_loss = float(losses_tensor.mean().item())
    std_loss = float(losses_tensor.std(unbiased=False).item())
    final_loss = float(losses[-1])

    result = {
        "avg_loss": avg_loss,
        "std_loss": std_loss,
        "final_loss": final_loss,
    }

    if run is not None:
        run.summary["avg_loss"] = avg_loss
        run.summary["std_loss"] = std_loss
        run.summary["final_loss"] = final_loss
        run.finish()

    return result


def probe_model(
    model: torch.nn.Module,
    Xm: torch.Tensor,
    Ym: torch.Tensor,
    Xq: torch.Tensor,
) -> None:
    model.eval()
    with torch.no_grad():
        _ = model(Xq, Xm, Ym)


def row_center(A: torch.Tensor) -> torch.Tensor:
    return A - A.mean(dim=-1, keepdim=True)


def row_standardize(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (A - A.mean(dim=-1, keepdim=True)) / (
        A.std(dim=-1, keepdim=True, unbiased=False) + eps
    )


def get_internal_state(
    model: torch.nn.Module,
    name: str,
) -> torch.Tensor | None:
    if name == "G":
        return model.G

    if name == "Attn":
        return model.Attn

    if name == "softmax_G":
        if model.G is None:
            return None
        return F.softmax(model.G, dim=-1)

    if name == "M":
        return model.M

    raise ValueError(f"Unknown internal state: {name}")


def normalize_matrix(
    A: torch.Tensor,
    normalize: str,
) -> torch.Tensor:
    if normalize == "none":
        return A

    if normalize == "row_center":
        return row_center(A)

    if normalize == "row_standardize":
        return row_standardize(A)

    raise ValueError(f"Unknown normalize mode: {normalize}")


def compare_model_internal_states(
    models: dict[str, torch.nn.Module],
    pairs: list[tuple[str, str]],
    state_name: str = "G",
    normalize: str = "row_center",
) -> dict[str, float | None]:
    results = {}

    for a_name, b_name in pairs:
        A = get_internal_state(models[a_name], state_name)
        B = get_internal_state(models[b_name], state_name)

        key = f"cos({a_name}.{state_name}, {b_name}.{state_name})"

        if A is None or B is None:
            results[key] = None
            continue

        if A.shape != B.shape:
            results[key] = None
            continue

        A = normalize_matrix(A, normalize)
        B = normalize_matrix(B, normalize)

        results[key] = matrix_cosine_similarity(A, B)

    return results


def summarize_M(model: torch.nn.Module) -> dict[str, float | None]:
    M = model.M

    if M is None:
        return {
            "M_norm": None,
            "M_condition": None,
        }

    output = {
        "M_norm": float(torch.linalg.norm(M).item()),
        "M_condition": None,
    }

    if M.ndim == 2 and M.shape[0] == M.shape[1]:
        try:
            output["M_condition"] = float(condition_number(M))
        except Exception:
            output["M_condition"] = None

    return output


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_results(results: dict[str, float | None]) -> None:
    for key, value in results.items():
        if value is None:
            print(f"{key}: None")
        else:
            print(f"{key}: {value:.6f}")


def main() -> None:
    set_seed(CONFIG["seed"])

    device = torch.device(CONFIG["device"])

    X, Y = synthetic_data()
    X = X.to(device).float()
    Y = Y.to(device).float()

    dim = CONFIG["dim"]
    lambda_reg = CONFIG["lambda_reg"]

    print_section("Training learnable models")

    trained_bilinear, loss_log_bi = train_Bilinear(
        X=X,
        Y=Y,
        dim=dim,
        memory_size=CONFIG["memory_size"],
        query_size=CONFIG["query_size"],
        steps=CONFIG["train_steps"],
        lr=CONFIG["learning_rate"],
        log_interval=CONFIG["log_interval"],
    )

    trained_attention, loss_log_attn = train_Attn(
        X=X,
        Y=Y,
        dim=dim,
        memory_size=CONFIG["memory_size"],
        query_size=CONFIG["query_size"],
        steps=CONFIG["train_steps"],
        lr=CONFIG["learning_rate"],
        log_interval=CONFIG["log_interval"],
    )

    models = {
        "dot": DotProdAttention(),
        "white": WhitenedDotProdAttention(dim, lambda_reg),
        "linr": LinearRegression(dim, lambda_reg),
        "bilinear": trained_bilinear,
        "attention": trained_attention,
    }

    print_section("Building shared evaluation episodes")

    eval_batches = make_eval_batches(
        X=X,
        Y=Y,
        num_episodes=CONFIG["eval_episodes"],
        memory_size=CONFIG["memory_size"],
        query_size=CONFIG["query_size"],
    )

    print_section("Evaluation")

    eval_results = {}

    for name, model in models.items():
        result = evaluate_model(
            model=model,
            model_name=name,
            eval_batches=eval_batches,
            use_wandb=CONFIG["use_wandb"],
        )

        eval_results[name] = result

        print(
            f"{name:10s} | "
            f"avg_loss={result['avg_loss']:.6f} | "
            f"std_loss={result['std_loss']:.6f} | "
            f"final_loss={result['final_loss']:.6f}"
        )

    print_section("Probe all models on the same batch")

    Xm_probe, Ym_probe, Xq_probe, Yq_probe = eval_batches[0]

    for name, model in models.items():
        probe_model(model, Xm_probe, Ym_probe, Xq_probe)

    g_pairs = [
        ("dot", "linr"),
        ("white", "linr"),
        ("bilinear", "linr"),
        ("attention", "linr"),
        ("bilinear", "attention"),
        ("white", "bilinear"),
        ("white", "attention"),
    ]

    m_pairs = [
        ("white", "linr"),
        ("bilinear", "linr"),
        ("attention", "linr"),
        ("bilinear", "attention"),
        ("white", "bilinear"),
        ("white", "attention"),
    ]

    softmax_pairs = [
        ("dot", "linr"),
        ("white", "linr"),
        ("bilinear", "linr"),
        ("attention", "linr"),
        ("bilinear", "attention"),
        ("white", "bilinear"),
        ("white", "attention"),
    ]

    print_section("G similarity: row-centered")

    G_centered_comparisons = compare_model_internal_states(
        models=models,
        pairs=g_pairs,
        state_name="G",
        normalize="row_center",
    )
    print_results(G_centered_comparisons)

    print_section("G similarity: row-standardized")

    G_standardized_comparisons = compare_model_internal_states(
        models=models,
        pairs=g_pairs,
        state_name="G",
        normalize="row_standardize",
    )
    print_results(G_standardized_comparisons)

    print_section("softmax(G) similarity")

    softmax_G_comparisons = compare_model_internal_states(
        models=models,
        pairs=softmax_pairs,
        state_name="softmax_G",
        normalize="none",
    )
    print_results(softmax_G_comparisons)

    print_section("Raw Attn similarity")

    attn_pairs = [
        ("dot", "white"),
        ("dot", "bilinear"),
        ("dot", "attention"),
        ("white", "bilinear"),
        ("white", "attention"),
        ("bilinear", "attention"),
    ]

    Attn_comparisons = compare_model_internal_states(
        models=models,
        pairs=attn_pairs,
        state_name="Attn",
        normalize="none",
    )
    print_results(Attn_comparisons)

    print_section("M statistics after same probe batch")

    for name, model in models.items():
        stats = summarize_M(model)
        print(f"{name:10s} | {stats}")

    print_section("M cosine similarity")

    M_comparisons = compare_model_internal_states(
        models=models,
        pairs=m_pairs,
        state_name="M",
        normalize="none",
    )
    print_results(M_comparisons)

    print_section("Training loss summary")

    print(f"bilinear first loss:  {loss_log_bi[0]:.6f}")
    print(f"bilinear final loss:  {loss_log_bi[-1]:.6f}")
    print(f"attention first loss: {loss_log_attn[0]:.6f}")
    print(f"attention final loss: {loss_log_attn[-1]:.6f}")


if __name__ == "__main__":
    main()