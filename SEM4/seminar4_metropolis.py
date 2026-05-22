"""
Семінарське заняття №4 — Імовірнісне машинне навчання
Тема: алгоритм Метрополіса–Гастінгса для цільового розподілу
      pi(x) ∝ exp(-(x^4 - x^2)) = exp(x^2 - x^4)

Скрипт:
1) реалізує MCMC для нормального та лапласівського proposal;
2) будує діагностичні графіки;
3) порівнює вибірки з еталонною rejection-sampling вибіркою через KS-тест;
4) оцінює acceptance rate, середнє, стандартне відхилення, ESS;
5) демонструє вплив параметра кроку на якість вибірки.
"""

from __future__ import annotations

import os
import math
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import quad

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 1. Налаштування
# ------------------------------------------------------------

OUT_DIR = Path("seminar4_outputs")
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_SAMPLES = 35_000
BURN_IN = 3_000

SIGMA = 1.0
LAMBDA = 1.0


# ------------------------------------------------------------
# 2. Цільовий розподіл
# ------------------------------------------------------------

def log_target(x):
    """
    log pi(x) = x^2 - x^4.
    Логарифмічна форма потрібна для числової стабільності.
    """
    x = np.asarray(x)
    return x ** 2 - x ** 4


def target_unnormalized(x):
    """Ненормована щільність pi(x) ∝ exp(x^2 - x^4)."""
    return np.exp(log_target(x))


Z, _ = quad(lambda t: float(target_unnormalized(t)), -np.inf, np.inf)


def target_pdf(x):
    """Нормована щільність для графіків."""
    return target_unnormalized(x) / Z


def theoretical_mean():
    num, _ = quad(lambda t: t * float(target_unnormalized(t)), -np.inf, np.inf)
    return num / Z


def theoretical_std():
    m = theoretical_mean()
    second, _ = quad(lambda t: (t - m) ** 2 * float(target_unnormalized(t)), -np.inf, np.inf)
    return math.sqrt(second / Z)


# ------------------------------------------------------------
# 3. Алгоритм Метрополіса
# ------------------------------------------------------------

def metropolis(
    n_samples: int,
    proposal: str,
    sigma: float = 1.0,
    lam: float = 1.0,
    x0: float = 0.0,
    seed: int = 42
):
    """
    Реалізація алгоритму Метрополіса для симетричних proposal-розподілів.

    proposal = "normal":
        x' ~ N(x, sigma^2)

    proposal = "laplace":
        x' ~ Laplace(x, scale=1/lambda)

    Оскільки обидва proposal симетричні, коефіцієнт прийняття:
        alpha = min(1, pi(x') / pi(x))
    """
    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples)
    x_current = x0
    accepted = 0

    for i in range(n_samples):
        if proposal == "normal":
            x_candidate = rng.normal(loc=x_current, scale=sigma)
        elif proposal == "laplace":
            x_candidate = rng.laplace(loc=x_current, scale=1.0 / lam)
        else:
            raise ValueError("proposal має бути 'normal' або 'laplace'")

        log_alpha = log_target(x_candidate) - log_target(x_current)

        if np.log(rng.uniform()) < log_alpha:
            x_current = x_candidate
            accepted += 1

        samples[i] = x_current

    return samples, accepted / n_samples


# ------------------------------------------------------------
# 4. Діагностичні метрики
# ------------------------------------------------------------

def autocorrelation(x: np.ndarray, max_lag: int = 150) -> np.ndarray:
    """ACF для лагів 0..max_lag."""
    x = np.asarray(x)
    x = x - x.mean()
    denom = np.dot(x, x)

    if denom == 0:
        return np.ones(max_lag + 1)

    acf = np.array([
        np.dot(x[:len(x) - lag], x[lag:]) / denom
        for lag in range(max_lag + 1)
    ])
    return acf


def effective_sample_size(x: np.ndarray, max_lag: int = 120) -> float:
    """
    Наближена ESS-оцінка:
        ESS = n / (1 + 2 * sum rho_k)
    де підсумовуємо тільки додатну частину ACF.
    """
    acf = autocorrelation(x, max_lag=max_lag)
    positive = acf[1:][acf[1:] > 0]
    tau = 1 + 2 * positive.sum()
    return len(x) / tau


def rejection_sampling(n: int, seed: int = 123) -> np.ndarray:
    """
    Еталонна вибірка з pi(x) через rejection sampling на відрізку [-3, 3].
    У межах цього інтервалу практично вся маса розподілу.
    """
    rng = np.random.default_rng(seed)
    max_val = math.exp(0.25)  # максимум exp(x^2 - x^4), x = ±sqrt(1/2)

    result = []
    batch = max(5000, n // 4)

    while len(result) < n:
        x = rng.uniform(-3, 3, size=batch)
        y = rng.uniform(0, max_val, size=batch)
        accepted = x[y <= target_unnormalized(x)]
        result.extend(accepted.tolist())

    return np.array(result[:n])


def summarize_chain(name: str, samples: np.ndarray, acc_rate: float) -> dict:
    ess = effective_sample_size(samples)
    return {
        "name": name,
        "acceptance_rate": acc_rate,
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)),
        "median": float(np.median(samples)),
        "ess": float(ess),
        "ess_percent": float(100 * ess / len(samples)),
    }


# ------------------------------------------------------------
# 5. Графіки
# ------------------------------------------------------------

def save_target_plot(out_dir: Path):
    x = np.linspace(-2.4, 2.4, 1000)
    y = target_pdf(x)
    modes = [-math.sqrt(0.5), math.sqrt(0.5)]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, y, linewidth=2.5, label="Нормована цільова щільність")
    ax.axvline(modes[0], linestyle="--", linewidth=1.4, label="Моди ±√0.5")
    ax.axvline(modes[1], linestyle="--", linewidth=1.4)
    ax.fill_between(x, y, alpha=0.18)
    ax.set_title("Цільовий розподіл π(x) ∝ exp(x² − x⁴)")
    ax.set_xlabel("x")
    ax.set_ylabel("Щільність")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "fig1_target_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_diagnostics(
    samples_full: np.ndarray,
    samples: np.ndarray,
    acc_rate: float,
    proposal_label: str,
    out_path: Path,
    max_lag: int = 120
):
    x_grid = np.linspace(-2.4, 2.4, 1000)
    true_pdf = target_pdf(x_grid)
    true_m = theoretical_mean()
    acf = autocorrelation(samples, max_lag=max_lag)
    cumulative_mean = np.cumsum(samples) / (np.arange(len(samples)) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))

    ax = axes[0, 0]
    ax.plot(samples_full[:3000], linewidth=0.55, alpha=0.8)
    if BURN_IN < 3000:
        ax.axvline(BURN_IN, linestyle="--", linewidth=1.2, label="burn-in")
    ax.set_title(f"Trace plot: {proposal_label}")
    ax.set_xlabel("Ітерація")
    ax.set_ylabel("x")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    ax.hist(samples, bins=100, density=True, alpha=0.65, label=f"MCMC, acc={acc_rate:.1%}")
    ax.plot(x_grid, true_pdf, linewidth=2.0, label="Теоретична π(x)")
    ax.set_title("Гістограма після burn-in")
    ax.set_xlabel("x")
    ax.set_ylabel("Щільність")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(np.arange(max_lag + 1), acf, width=1.0, alpha=0.75)
    ax.axhline(0, linewidth=1.0)
    ax.set_title("Автокореляційна функція")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(cumulative_mean, linewidth=1.1, label="Кумулятивне середнє")
    ax.axhline(true_m, linestyle="--", linewidth=1.4, label=f"Теор. E[X]≈{true_m:.3f}")
    ax.set_title("Збіжність середнього")
    ax.set_xlabel("Кількість зразків")
    ax.set_ylabel("Середнє")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(f"Діагностика алгоритму Метрополіса: {proposal_label}", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_comparison_plot(sn, sl, acc_n, acc_l, out_path: Path):
    x_grid = np.linspace(-2.4, 2.4, 1000)
    true_pdf = target_pdf(x_grid)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)

    axes[0].hist(sn, bins=100, density=True, alpha=0.65, label=f"Normal, acc={acc_n:.1%}")
    axes[0].plot(x_grid, true_pdf, linewidth=2, label="Теоретична π(x)")
    axes[0].set_title("Нормальний proposal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Щільність")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].hist(sl, bins=100, density=True, alpha=0.65, label=f"Laplace, acc={acc_l:.1%}")
    axes[1].plot(x_grid, true_pdf, linewidth=2, label="Теоретична π(x)")
    axes[1].set_title("Лапласівський proposal")
    axes[1].set_xlabel("x")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("Порівняння відтворення цільового розподілу", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_parameter_sensitivity(out_path: Path):
    settings = [0.15, 1.0, 3.0]
    labels = ["замалий крок σ=0.15", "збалансований σ=1.0", "завеликий крок σ=3.0"]
    x_grid = np.linspace(-2.4, 2.4, 1000)
    true_pdf = target_pdf(x_grid)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for ax, sigma, label, seed in zip(axes, settings, labels, [101, 102, 103]):
        chain, acc = metropolis(18_000, proposal="normal", sigma=sigma, seed=seed)
        s = chain[2_000:]
        ax.hist(s, bins=80, density=True, alpha=0.65)
        ax.plot(x_grid, true_pdf, linewidth=2)
        ax.set_title(f"{label}\nAcceptance={acc:.1%}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Щільність")
    fig.suptitle("Вплив масштабу proposal на якість MCMC-вибірки", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------
# 6. Основний експеримент
# ------------------------------------------------------------

def run_experiment():
    print("Запуск MCMC...")

    samples_normal_full, acc_normal = metropolis(
        N_SAMPLES, proposal="normal", sigma=SIGMA, seed=SEED
    )
    samples_laplace_full, acc_laplace = metropolis(
        N_SAMPLES, proposal="laplace", lam=LAMBDA, seed=SEED + 1
    )

    sn = samples_normal_full[BURN_IN:]
    sl = samples_laplace_full[BURN_IN:]

    print(f"Нормальний proposal: acceptance rate = {acc_normal:.4f}")
    print(f"Лапласівський proposal: acceptance rate = {acc_laplace:.4f}")

    ref = rejection_sampling(20_000, seed=777)
    ks_normal = stats.ks_2samp(sn, ref)
    ks_laplace = stats.ks_2samp(sl, ref)

    summary_normal = summarize_chain("Нормальний proposal", sn, acc_normal)
    summary_laplace = summarize_chain("Лапласівський proposal", sl, acc_laplace)

    for summary, ks in [(summary_normal, ks_normal), (summary_laplace, ks_laplace)]:
        summary["ks_stat"] = float(ks.statistic)
        summary["ks_pvalue"] = float(ks.pvalue)

    print("\nKS-тест:")
    print(f"Нормальний: D={ks_normal.statistic:.4f}, p={ks_normal.pvalue:.4f}")
    print(f"Лапласівський: D={ks_laplace.statistic:.4f}, p={ks_laplace.pvalue:.4f}")

    print("\nОписова статистика:")
    for s in [summary_normal, summary_laplace]:
        print(
            f"{s['name']}: mean={s['mean']:.5f}, std={s['std']:.5f}, "
            f"ESS={s['ess']:.0f}, ESS%={s['ess_percent']:.1f}%"
        )

    save_target_plot(OUT_DIR)
    save_diagnostics(
        samples_normal_full,
        sn,
        acc_normal,
        f"нормальний proposal, σ={SIGMA}",
        OUT_DIR / "fig2_normal_diagnostics.png"
    )
    save_diagnostics(
        samples_laplace_full,
        sl,
        acc_laplace,
        f"лапласівський proposal, λ={LAMBDA}",
        OUT_DIR / "fig3_laplace_diagnostics.png"
    )
    save_comparison_plot(
        sn, sl, acc_normal, acc_laplace,
        OUT_DIR / "fig4_comparison_histograms.png"
    )
    save_parameter_sensitivity(OUT_DIR / "fig5_parameter_sensitivity.png")

    return summary_normal, summary_laplace


if __name__ == "__main__":
    run_experiment()
