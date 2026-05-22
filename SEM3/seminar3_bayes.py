# seminar3_bayes.py
# Семінарське заняття №3 з дисципліни "Імовірнісне машинне навчання"
# Тема: теорема Байєса, base rate fallacy та послідовне оновлення

import numpy as np
import matplotlib.pyplot as plt


def bayes_update(prior: float, sensitivity: float, fpr: float):
    """Повертає апостеріорну ймовірність P(C|+) та P(+)."""
    p_positive = sensitivity * prior + fpr * (1 - prior)
    posterior = sensitivity * prior / p_positive
    return posterior, p_positive


def main():
    prior = 0.01
    sensitivity_1 = 0.80
    fpr_1 = 0.096
    sensitivity_2 = 0.90
    fpr_2 = 0.056

    posterior_1, p_positive_1 = bayes_update(prior, sensitivity_1, fpr_1)
    posterior_2, p_positive_2 = bayes_update(posterior_1, sensitivity_2, fpr_2)

    print("=" * 60)
    print("ЗАВДАННЯ 1. Один позитивний тест")
    print("=" * 60)
    print(f"P(C) = {prior:.5f}")
    print(f"P(+|C) = {sensitivity_1:.5f}")
    print(f"P(+|not C) = {fpr_1:.5f}")
    print(f"P(+) = {sensitivity_1}*{prior} + {fpr_1}*{1-prior} = {p_positive_1:.5f}")
    print(f"P(C|+1) = {posterior_1:.5f} = {posterior_1*100:.2f}%")

    print("\n" + "=" * 60)
    print("ЗАВДАННЯ 2. Другий незалежний позитивний тест")
    print("=" * 60)
    print(f"Новий prior = P(C|+1) = {posterior_1:.5f}")
    print(f"P(+2) = {sensitivity_2}*{posterior_1:.5f} + {fpr_2}*{1-posterior_1:.5f} = {p_positive_2:.6f}")
    print(f"P(C|+1,+2) = {posterior_2:.5f} = {posterior_2*100:.2f}%")

    # Інтерпретація на 1000 осіб
    N = 1000
    sick = N * prior
    healthy = N - sick
    tp1 = sick * sensitivity_1
    fp1 = healthy * fpr_1
    print("\nІнтерпретація першого тесту на 1000 осіб:")
    print(f"Справді хворих із позитивним тестом: {tp1:.2f}")
    print(f"Здорових із хибнопозитивним тестом: {fp1:.2f}")
    print(f"PPV = {tp1 / (tp1 + fp1):.5f}")

    stages = ["Prior", "After test 1", "After test 2"]
    values = [prior * 100, posterior_1 * 100, posterior_2 * 100]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(stages, values)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, value + 1, f"{value:.2f}%", ha="center")
    plt.ylabel("Probability, %")
    plt.title("Bayesian updating after positive tests")
    plt.ylim(0, 70)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("seminar3_bayes_update.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
