
# Код для семінарського заняття №2 з дисципліни "Імовірнісне машинне навчання"
# Автор звіту: Шайда Валерій Віталійович
#
# Що робить скрипт:
# 1) генерує 2D гауссівські класи та будує лінійний байєсівський класифікатор;
# 2) досліджує 2D нормальні розподіли з різними кореляціями;
# 3) генерує 3D гауссівські вибірки з R = I;
# 4) аналізує 3D нормальні розподіли з різними кореляційними матрицями;
# 5) перевіряє нормальність за Shapiro-Wilk та Kolmogorov-Smirnov;
# 6) генерує 5D набір даних із заданою структурою кореляцій.
#
# Перед запуском:
# pip install numpy pandas matplotlib scipy scikit-learn

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2, norm, shapiro, kstest, probplot
from sklearn.metrics import confusion_matrix, accuracy_score


# ------------------------------------------------------------
# Загальні налаштування
# ------------------------------------------------------------

np.random.seed(42)

OUT_DIR = "seminar2_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10


# ------------------------------------------------------------
# Допоміжні функції
# ------------------------------------------------------------

def bayes_score(X, mu, R, prior=0.5):
    """
    Дискримінантна функція для багатовимірного нормального розподілу:
    g_i(x) = -1/2 ln|R_i| - 1/2 (x-mu_i)^T R_i^-1 (x-mu_i) + ln P(w_i)
    """
    X = np.atleast_2d(X)
    mu = np.asarray(mu)
    R = np.asarray(R)
    inv_R = np.linalg.inv(R)
    det_R = np.linalg.det(R)
    diff = X - mu
    quad = np.sum(diff @ inv_R * diff, axis=1)
    return -0.5 * np.log(det_R) - 0.5 * quad + np.log(prior)


def predict_bayes(X, mus, covs, priors):
    scores = np.column_stack([
        bayes_score(X, mus[i], covs[i], priors[i])
        for i in range(len(mus))
    ])
    return np.argmax(scores, axis=1)


def confidence_ellipse(ax, mean, cov, color="crimson", label=None, level=0.95):
    """
    Малює 95% довірчий еліпс для двовимірного нормального розподілу.
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    radius = math.sqrt(chi2.ppf(level, df=2))

    t = np.linspace(0, 2 * np.pi, 300)
    ellipse = np.array([np.cos(t), np.sin(t)])
    scale = np.diag(radius * np.sqrt(vals))
    points = vecs @ scale @ ellipse
    points[0, :] += mean[0]
    points[1, :] += mean[1]

    ax.plot(points[0], points[1], color=color, linewidth=2, label=label)


def draw_decision_regions_2d(ax, mus, covs, priors, xlim, ylim, alpha=0.18):
    """
    Малює області рішень для двокласового байєсівського класифікатора.
    """
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 300),
        np.linspace(ylim[0], ylim[1], 300)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = predict_bayes(grid, mus, covs, priors).reshape(xx.shape)

    ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5], alpha=alpha)
    ax.contour(xx, yy, pred, levels=[0.5], colors="black", linestyles="--", linewidths=1.5)


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Збережено: {path}")


# ------------------------------------------------------------
# Завдання 1. Лінійний байєсівський класифікатор у 2D
# ------------------------------------------------------------

def task1_linear_bayes():
    n = 500
    mu0 = np.array([-1.0, -1.0])
    mu1 = np.array([1.0, 1.0])
    R = np.eye(2)

    X0 = np.random.multivariate_normal(mu0, R, n)
    X1 = np.random.multivariate_normal(mu1, R, n)

    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)

    mus = [mu0, mu1]
    covs = [R, R]
    priors = [0.5, 0.5]

    y_pred = predict_bayes(X, mus, covs, priors)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    d2 = (mu1 - mu0).T @ np.linalg.inv(R) @ (mu1 - mu0)
    d = math.sqrt(d2)
    theoretical_error = norm.cdf(-d / 2)

    print("\nЗавдання 1")
    print("Accuracy:", round(acc, 4))
    print("Confusion matrix:\n", cm)
    print("Mahalanobis distance:", round(d, 4))
    print("Theoretical error:", round(theoretical_error, 4))

    fig, ax = plt.subplots(figsize=(7, 6))

    xlim = (-5, 5)
    ylim = (-5, 5)
    draw_decision_regions_2d(ax, mus, covs, priors, xlim, ylim)

    ax.scatter(X0[:, 0], X0[:, 1], s=16, alpha=0.55, label="Клас 0")
    ax.scatter(X1[:, 0], X1[:, 1], s=16, alpha=0.55, label="Клас 1")

    confidence_ellipse(ax, mu0, R, color="crimson", label="95% еліпс")
    confidence_ellipse(ax, mu1, R, color="crimson")

    ax.scatter(mu0[0], mu0[1], marker="*", s=180, color="red")
    ax.scatter(mu1[0], mu1[1], marker="*", s=180, color="red")

    ax.set_title(f"Завдання 1. Лінійний байєсівський класифікатор\nAccuracy = {acc:.3f}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    ax.legend()
    savefig("task1_linear_bayes.png")

    return X0, X1, cm, acc


# ------------------------------------------------------------
# Завдання 2. 2D розподіли з різними кореляційними матрицями
# ------------------------------------------------------------

def task2_correlations_2d():
    n = 700
    mu = np.array([0.0, 1.0])

    matrices = [
        np.array([[1.0, 0.8], [0.8, 1.0]]),
        np.array([[1.0, -0.5], [-0.5, 1.0]]),
        np.array([[1.0, 0.6], [0.6, 1.0]]),
        np.array([[1.0, -0.7], [-0.7, 1.0]])
    ]

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()

    for i, R in enumerate(matrices, start=1):
        X = np.random.multivariate_normal(mu, R, n)
        R_emp = np.corrcoef(X.T)
        eig = np.linalg.eigvalsh(R)

        rows.append({
            "Матриця": f"R{i}",
            "Теоретична rho": R[0, 1],
            "Вибіркова rho": R_emp[0, 1],
            "Власне число 1": eig[0],
            "Власне число 2": eig[1]
        })

        ax = axes[i - 1]
        ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.45)
        confidence_ellipse(ax, mu, R, color="crimson")
        ax.scatter(mu[0], mu[1], marker="*", s=130, color="orange")
        ax.set_title(f"R{i}: rho={R[0,1]}, вибірк. rho={R_emp[0,1]:.3f}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    savefig("task2_2d_correlations.png")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "task2_correlation_summary.csv"), index=False)
    print("\nЗавдання 2")
    print(df.round(4))
    return df


# ------------------------------------------------------------
# Завдання 3. 3D гауссівські вибірки при R = I
# ------------------------------------------------------------

def task3_3d_identity():
    n = 450
    R = np.eye(3)

    mus = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]

    fig = plt.figure(figsize=(10, 8))

    for i, mu in enumerate(mus, start=1):
        X = np.random.multivariate_normal(mu, R, n)

        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=7, alpha=0.35)
        ax.scatter(mu[0], mu[1], mu[2], marker="*", s=150, color="orange")

        ax.set_title(f"mu = {tuple(mu.astype(int))}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.set_xlim(-4, 5)
        ax.set_ylim(-4, 5)
        ax.set_zlim(-4, 5)

    savefig("task3_3d_identity.png")
    print("\nЗавдання 3: збережено 3D-візуалізацію.")


# ------------------------------------------------------------
# Завдання 4-6. 3D матриці кореляції
# ------------------------------------------------------------

def task4_3d_correlations():
    n = 700
    mu = np.array([1.0, 1.0, 1.0])

    matrices = [
        np.array([[1.0, 0.8, 0.6],
                  [0.8, 1.0, 0.7],
                  [0.6, 0.7, 1.0]]),

        np.array([[1.0, -0.8, -0.6],
                  [-0.8, 1.0, 0.6],
                  [-0.6, 0.6, 1.0]]),

        np.array([[1.0, 0.8, -0.6],
                  [0.8, 1.0, -0.4],
                  [-0.6, -0.4, 1.0]]),

        np.array([[1.0, -0.8, -0.7],
                  [-0.8, 1.0, 0.8],
                  [-0.7, 0.8, 1.0]])
    ]

    eig_rows = []

    for idx, R in enumerate(matrices, start=1):
        eig = np.linalg.eigvalsh(R)
        eig_rows.append({
            "Матриця": f"R{idx}",
            "lambda_min": eig[0],
            "lambda_mid": eig[1],
            "lambda_max": eig[2],
            "Позитивно визначена": bool(np.all(eig > 0))
        })

        X = np.random.multivariate_normal(mu, R, n)
        R_emp = np.corrcoef(X.T)

        fig = plt.figure(figsize=(10, 8))

        ax3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax3d.scatter(X[:, 0], X[:, 1], X[:, 2], s=7, alpha=0.35)
        ax3d.scatter(mu[0], mu[1], mu[2], marker="*", s=150, color="orange")
        ax3d.set_title(f"R{idx}: 3D хмара")
        ax3d.set_xlabel("x1")
        ax3d.set_ylabel("x2")
        ax3d.set_zlabel("x3")

        pairs = [(0, 1), (0, 2), (1, 2)]
        titles = ["Проекція x1-x2", "Проекція x1-x3", "Проекція x2-x3"]

        for j, (a, b) in enumerate(pairs, start=2):
            ax = fig.add_subplot(2, 2, j)
            ax.scatter(X[:, a], X[:, b], s=10, alpha=0.4)
            ax.scatter(mu[a], mu[b], marker="*", s=120, color="orange")
            ax.set_title(titles[j - 2])
            ax.set_xlabel(f"x{a + 1}")
            ax.set_ylabel(f"x{b + 1}")
            ax.grid(alpha=0.3)

        savefig(f"task4_R{idx}_3d_and_projections.png")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        im0 = axes[0].imshow(R, vmin=-1, vmax=1)
        axes[0].set_title(f"R{idx}: теоретична R")
        axes[0].set_xticks([0, 1, 2], ["x1", "x2", "x3"])
        axes[0].set_yticks([0, 1, 2], ["x1", "x2", "x3"])

        im1 = axes[1].imshow(R_emp, vmin=-1, vmax=1)
        axes[1].set_title(f"R{idx}: вибіркова R")
        axes[1].set_xticks([0, 1, 2], ["x1", "x2", "x3"])
        axes[1].set_yticks([0, 1, 2], ["x1", "x2", "x3"])

        for ax, matrix in zip(axes, [R, R_emp]):
            for r in range(3):
                for c in range(3):
                    ax.text(c, r, f"{matrix[r, c]:.2f}", ha="center", va="center")

        fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.75)
        savefig(f"task4_R{idx}_heatmaps.png")

    eig_df = pd.DataFrame(eig_rows)
    eig_df.to_csv(os.path.join(OUT_DIR, "task4_eigenvalues.csv"), index=False)

    print("\nЗавдання 4-6")
    print(eig_df.round(4))
    return eig_df


# ------------------------------------------------------------
# Завдання 8. Перевірка гіпотез про нормальність
# ------------------------------------------------------------

def task8_normality_tests(X0, X1):
    """
    Перевіряємо кілька маргінальних вимірів на нормальність.
    """
    samples = {
        "class0_x1": X0[:, 0],
        "class0_x2": X0[:, 1],
        "class1_x1": X1[:, 0],
        "class1_x2": X1[:, 1],
    }

    rows = []

    for name, sample in samples.items():
        sample = np.asarray(sample)
        mu_hat = sample.mean()
        sigma_hat = sample.std(ddof=1)

        W, p_shapiro = shapiro(sample)
        D, p_ks = kstest(sample, "norm", args=(mu_hat, sigma_hat))

        rows.append({
            "Вибірка": name,
            "mean": mu_hat,
            "std": sigma_hat,
            "W Shapiro": W,
            "p Shapiro": p_shapiro,
            "D KS": D,
            "p KS": p_ks,
            "Висновок alpha=0.05": "не відхиляємо H0" if p_shapiro > 0.05 and p_ks > 0.05 else "є підстави відхилити H0"
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "task8_normality_tests.csv"), index=False)

    print("\nЗавдання 8")
    print(df.round(4))

    # Q-Q графіки
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()

    for ax, (name, sample) in zip(axes, samples.items()):
        probplot(sample, dist="norm", plot=ax)
        ax.set_title(f"Q-Q графік: {name}")
        ax.grid(alpha=0.3)

    savefig("task8_qq_plots.png")

    # Гістограми з накладеною нормальною густиною
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()

    for ax, (name, sample) in zip(axes, samples.items()):
        mu_hat = sample.mean()
        sigma_hat = sample.std(ddof=1)

        xs = np.linspace(sample.min(), sample.max(), 250)
        ax.hist(sample, bins=25, density=True, alpha=0.6)
        ax.plot(xs, norm.pdf(xs, mu_hat, sigma_hat), linewidth=2)
        ax.set_title(f"Гістограма: {name}")
        ax.grid(alpha=0.3)

    savefig("task8_histograms.png")
    return df


# ------------------------------------------------------------
# Завдання 9. 5D набір із заданою структурою кореляції
# ------------------------------------------------------------

def task9_five_dimensional_dataset():
    n = 600

    mu = np.array([2.0, 2.5, 0.0, -1.0, 1.5])

    R = np.array([
        [1.0, 0.85, 0.0, 0.0, 0.0],
        [0.85, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.30],
        [0.0, 0.0, 0.0, 0.30, 1.0],
    ])

    X = np.random.multivariate_normal(mu, R, n)
    columns = ["x1", "x2", "x3", "x4", "x5"]
    df = pd.DataFrame(X, columns=columns)

    df.to_csv(os.path.join(OUT_DIR, "task9_generated_5d_dataset.csv"), index=False)

    R_emp = df.corr().values

    # Теплові карти теоретичної та вибіркової кореляції
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(R, vmin=-1, vmax=1)
    axes[0].set_title("Теоретична кореляційна матриця")

    im = axes[1].imshow(R_emp, vmin=-1, vmax=1)
    axes[1].set_title("Вибіркова кореляційна матриця")

    for ax, matrix in zip(axes, [R, R_emp]):
        ax.set_xticks(range(5), columns)
        ax.set_yticks(range(5), columns)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    savefig("task9_correlation_heatmaps.png")

    # Порівняння сильної, слабкої та майже нульової кореляції
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    pairs = [
        ("x1", "x2", "Сильна кореляція x1-x2"),
        ("x1", "x3", "Майже нульова кореляція x1-x3"),
        ("x4", "x5", "Слабка кореляція x4-x5"),
    ]

    for ax, (a, b, title) in zip(axes, pairs):
        ax.scatter(df[a], df[b], s=12, alpha=0.45)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    savefig("task9_pairwise_comparison.png")

    print("\nЗавдання 9")
    print("Перші 10 рядків 5D набору:")
    print(df.head(10).round(4))
    print("\nВибіркова кореляційна матриця:")
    print(df.corr().round(3))

    return df


# ------------------------------------------------------------
# Головний запуск
# ------------------------------------------------------------

def main():
    X0, X1, cm, acc = task1_linear_bayes()
    task2_correlations_2d()
    task3_3d_identity()
    task4_3d_correlations()
    task8_normality_tests(X0, X1)
    task9_five_dimensional_dataset()

    print("\nГотово.")
    print(f"Усі графіки, таблиці та CSV-файли збережено в папці: {OUT_DIR}")


if __name__ == "__main__":
    main()
