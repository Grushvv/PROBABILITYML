import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = "plots"
RANDOM_SEED = 42
N_SAMPLES_CLASS = 200
TEST_SIZE = 1000


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def console_print(msg: str):
    print(msg)


def log_gaussian_pdf(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(X)
    d = X.shape[1]

    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")

    Sigma_inv = np.linalg.inv(Sigma)
    diff = X - mu
    quad = np.einsum("ni,ij,nj->n", diff, Sigma_inv, diff)

    logp = -0.5 * quad - 0.5 * logdet - 0.5 * d * np.log(2 * np.pi)
    return logp


def discriminant_g(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, prior: float) -> np.ndarray:
    return log_gaussian_pdf(X, mu, Sigma) + np.log(prior)


def predict_class(X: np.ndarray,
                  mu1: np.ndarray, S1: np.ndarray, p1: float,
                  mu2: np.ndarray, S2: np.ndarray, p2: float) -> np.ndarray:
    g1 = discriminant_g(X, mu1, S1, p1)
    g2 = discriminant_g(X, mu2, S2, p2)
    return np.where(g1 > g2, 1, 2)


def posterior_prob_w1(X: np.ndarray,
                      mu1: np.ndarray, S1: np.ndarray, p1: float,
                      mu2: np.ndarray, S2: np.ndarray, p2: float) -> np.ndarray:
    g1 = discriminant_g(X, mu1, S1, p1)
    g2 = discriminant_g(X, mu2, S2, p2)

    m = np.maximum(g1, g2)
    num = np.exp(g1 - m)
    den = np.exp(g1 - m) + np.exp(g2 - m)
    return num / den


def sample_gaussian(rng: np.random.Generator,
                    mu: np.ndarray, Sigma: np.ndarray, n: int) -> np.ndarray:
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=n)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t - 1, p - 1] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def make_grid_2d(X_all: np.ndarray, pad: float = 1.0, step: float = 0.03):
    x_min = X_all[:, 0].min() - pad
    x_max = X_all[:, 0].max() + pad
    y_min = X_all[:, 1].min() - pad
    y_max = X_all[:, 1].max() + pad

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_2d_points_and_boundary(X1, X2, mu1, S1, p1, mu2, S2, p2,
                                title: str, fig_name: str):
    X_all = np.vstack([X1, X2])
    xx, yy, grid = make_grid_2d(X_all)

    z = discriminant_g(grid, mu1, S1, p1) - discriminant_g(grid, mu2, S2, p2)
    zz = z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.scatter(X1[:, 0], X1[:, 1], s=14, alpha=0.7, label="Class 1")
    plt.scatter(X2[:, 0], X2[:, 1], s=14, alpha=0.7, label="Class 2")
    plt.scatter([mu1[0]], [mu1[1]], marker="x", s=140, linewidths=3, label="μ1")
    plt.scatter([mu2[0]], [mu2[1]], marker="x", s=140, linewidths=3, label="μ2")

    plt.contour(xx, yy, zz, levels=[0.0], linewidths=2)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.25)
    plt.legend()

    out_path = os.path.join(OUTPUT_DIR, fig_name)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.show()


def plot_3d_points(X1, X2, mu1, mu2, title: str, fig_name: str):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], s=10, alpha=0.6, label="Class 1")
    ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], s=10, alpha=0.6, label="Class 2")
    ax.scatter([mu1[0]], [mu1[1]], [mu1[2]], marker="x", s=160, linewidths=3, label="μ1")
    ax.scatter([mu2[0]], [mu2[1]], [mu2[2]], marker="x", s=160, linewidths=3, label="μ2")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend()

    out_path = os.path.join(OUTPUT_DIR, fig_name)
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.show()


def experiment_2d(mu1, S1, mu2, S2, p, n, seed,
                  title: str, fig_name: str):
    rng = np.random.default_rng(seed)

    X1 = sample_gaussian(rng, mu1, S1, n)
    X2 = sample_gaussian(rng, mu2, S2, n)

    plot_2d_points_and_boundary(
        X1, X2, mu1, S1, p, mu2, S2, 1 - p,
        title=title, fig_name=fig_name
    )

    m = TEST_SIZE
    y_test = rng.choice([1, 2], size=m, p=[p, 1 - p])
    X_test = np.zeros((m, 2))

    idx1 = (y_test == 1)
    idx2 = ~idx1
    X_test[idx1] = sample_gaussian(rng, mu1, S1, idx1.sum())
    X_test[idx2] = sample_gaussian(rng, mu2, S2, idx2.sum())

    y_pred = predict_class(X_test, mu1, S1, p, mu2, S2, 1 - p)

    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    console_print("====================================")
    console_print(title)
    console_print(f"Prior probabilities: p1={p:.2f}, p2={1 - p:.2f}")
    console_print(f"Test size: {m}")
    console_print(f"Accuracy: {acc:.4f}")
    console_print("Confusion matrix (rows=true, cols=pred):")
    console_print(str(cm))

    post = posterior_prob_w1(X_test[:5], mu1, S1, p, mu2, S2, 1 - p)
    console_print("Example posterior P(w1|x) for first 5 test samples:")
    console_print(str(np.round(post, 4)))
    console_print("====================================\n")

    return acc, cm


def experiment_3d(mu1, S1, mu2, S2, p, n, seed,
                  title: str, fig_name: str):
    rng = np.random.default_rng(seed)

    X1 = sample_gaussian(rng, mu1, S1, n)
    X2 = sample_gaussian(rng, mu2, S2, n)

    plot_3d_points(X1, X2, mu1, mu2, title=title, fig_name=fig_name)

    m = max(TEST_SIZE, 2000)
    y_test = rng.choice([1, 2], size=m, p=[p, 1 - p])
    X_test = np.zeros((m, 3))

    idx1 = (y_test == 1)
    idx2 = ~idx1
    X_test[idx1] = sample_gaussian(rng, mu1, S1, idx1.sum())
    X_test[idx2] = sample_gaussian(rng, mu2, S2, idx2.sum())

    y_pred = predict_class(X_test, mu1, S1, p, mu2, S2, 1 - p)

    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    console_print("====================================")
    console_print(title)
    console_print(f"Prior probabilities: p1={p:.2f}, p2={1 - p:.2f}")
    console_print(f"Test size: {m}")
    console_print(f"Accuracy: {acc:.4f}")
    console_print("Confusion matrix (rows=true, cols=pred):")
    console_print(str(cm))
    console_print("====================================\n")

    return acc, cm


def main(p: float = 0.5):
    ensure_output_dir()

    mu1 = np.array([0.0, 1.0])
    mu2 = np.array([2.0, 0.0])
    R1 = np.array([[1.0, 0.0],
                   [0.0, 2.0]])
    R2 = np.array([[2.0, 0.0],
                   [0.0, 1.0]])

    experiment_2d(
        mu1, R1, mu2, R2, p=p, n=N_SAMPLES_CLASS, seed=RANDOM_SEED,
        title="Task 1 μ1=(0,1), μ2=(2,0), R1=(1,2), R2=(2,1)",
        fig_name="fig1_task1_2d_boundary.png"
    )

    mu1_b = np.array([1.0, 1.0])
    mu2_b = np.array([-1.0, -1.0])
    I2 = np.eye(2)

    experiment_2d(
        mu1_b, I2, mu2_b, I2, p=p, n=N_SAMPLES_CLASS, seed=RANDOM_SEED + 1,
        title="Task 4 (2D): μ1=(1,1), μ2=(-1,-1), R1=I, R2=I (LDA linear boundary)",
        fig_name="fig2_task4_2d_boundary.png"
    )

    mu1_3 = np.array([0.0, 0.0, 0.0])
    mu2_3 = np.array([1.0, 1.0, 1.0])
    R1_3 = np.array([[1.0, 0.0, 0.0],
                     [0.0, 2.0, 0.0],
                     [0.0, 0.0, 1.0]])
    R2_3 = np.array([[2.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 2.0]])

    experiment_3d(
        mu1_3, R1_3, mu2_3, R2_3, p=p, n=max(300, N_SAMPLES_CLASS), seed=RANDOM_SEED + 2,
        title="Task 5 (3D): μ1=(0,0,0), μ2=(1,1,1), R1=(1,2,1), R2=(2,1,2)",
        fig_name="fig3_task5_3d_scatter.png"
    )

    experiment_2d(
        mu1, R1, mu2, R2, p=p, n=N_SAMPLES_CLASS, seed=RANDOM_SEED + 3,
        title="Task 6A (2D): original covariances (QDA)",
        fig_name="fig4_task6A_original_cov.png"
    )

    experiment_2d(
        mu1, I2, mu2, I2, p=p, n=N_SAMPLES_CLASS, seed=RANDOM_SEED + 3,
        title="Task 6B (2D): identity covariances (LDA)",
        fig_name="fig5_task6B_identity_cov.png"
    )

    console_print("All figures saved to folder: plots/")


if __name__ == "__main__":
    main(p=0.5)