from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# 1) Get bet data from open sources
DATA_URL = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
LOCAL_CSV = "E0_2324.csv"


def download_dataset(url: str = DATA_URL, out_path: str = LOCAL_CSV) -> pd.DataFrame:
    """
    Завантажуємо конкретний датасет EPL 2023/24 (E0.csv) з football-data.co.uk.
    """
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.exceptions.SSLError:
        r = requests.get(url, timeout=30, verify=False)
        r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    df.to_csv(out_path, index=False)
    return df


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Перевіряємо, що є потрібні колонки:
      HomeTeam, AwayTeam, FTR, B365H, B365D, B365A
    """
    required = ["HomeTeam", "AwayTeam", "FTR", "B365H", "B365D", "B365A"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"У датасеті відсутні потрібні колонки: {missing}")

    # чистимо пропуски
    df = df.dropna(subset=required).copy()

    # залишаємо лише H/D/A у FTR (іноді можуть бути інші маркери)
    df["FTR"] = df["FTR"].astype(str).str.strip()
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()

    return df


# ------------------------------------------------------------
# 2) Visualize the distribution of sample data
# ------------------------------------------------------------
def plot_distributions(df: pd.DataFrame) -> None:
    """Гістограми odds і розподіл класів (H/D/A)."""
    plt.figure()
    plt.hist(df["B365H"], bins=40, alpha=0.7, label="B365H")
    plt.hist(df["B365D"], bins=40, alpha=0.7, label="B365D")
    plt.hist(df["B365A"], bins=40, alpha=0.7, label="B365A")
    plt.title("Розподіл букмекерських коефіцієнтів (odds)")
    plt.xlabel("odds")
    plt.ylabel("кількість")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    counts = df["FTR"].value_counts().reindex(["H", "D", "A"])
    plt.bar(counts.index, counts.values)
    plt.title("Розподіл результатів (FTR)")
    plt.xlabel("Клас")
    plt.ylabel("Кількість")
    plt.grid(True, axis="y")
    plt.show()


# ------------------------------------------------------------
# 3) Estimate accuracy of strategies:
#    random, deterministic (host, visitor, challenger) and Bayesian
# ------------------------------------------------------------
def strategy_random(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(["H", "D", "A"], size=len(df))


def strategy_host(df: pd.DataFrame) -> np.ndarray:
    return np.array(["H"] * len(df))


def strategy_visitor(df: pd.DataFrame) -> np.ndarray:
    return np.array(["A"] * len(df))


def strategy_challenger_favorite_by_odds(df: pd.DataFrame) -> np.ndarray:
    """
    Challenger (детермінована, але "розумна"):
    вибираємо результат з мінімальним odds (фаворит за коефіцієнтами).
    """
    odds = df[["B365H", "B365D", "B365A"]].to_numpy()
    idx = odds.argmin(axis=1)
    mapping = np.array(["H", "D", "A"])
    return mapping[idx]


@dataclass(frozen=True)
class BetaPosterior:
    alpha: float
    beta: float

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def fit_beta_posteriors(
    df_train: pd.DataFrame,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> Tuple[Dict[str, BetaPosterior], Dict[str, BetaPosterior]]:
    """
    Bayesian: для кожної команди оцінюємо:
      - P(HomeWin | team plays at home)
      - P(AwayWin | team plays away)

    Beta-Binomial:
      p ~ Beta(alpha0, beta0)
      posterior = Beta(alpha0 + successes, beta0 + failures)
    """
    teams = pd.unique(pd.concat([df_train["HomeTeam"], df_train["AwayTeam"]]))

    home_post: Dict[str, BetaPosterior] = {}
    away_post: Dict[str, BetaPosterior] = {}

    for t in teams:
        home_games = df_train[df_train["HomeTeam"] == t]
        s = int((home_games["FTR"] == "H").sum())
        f = int((home_games["FTR"] != "H").sum())
        home_post[t] = BetaPosterior(alpha0 + s, beta0 + f)

        away_games = df_train[df_train["AwayTeam"] == t]
        s = int((away_games["FTR"] == "A").sum())
        f = int((away_games["FTR"] != "A").sum())
        away_post[t] = BetaPosterior(alpha0 + s, beta0 + f)

    return home_post, away_post


def bayesian_predict_probs_and_decision(
    df_test: pd.DataFrame,
    home_post: Dict[str, BetaPosterior],
    away_post: Dict[str, BetaPosterior],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Повертає:
      - probs (n,3): [P(H), P(D), P(A)]
      - pred (n,): прогноз H/D/A

    Рішення робимо через ШТРАФНУ ФУНКЦІЮ (penalty):
      Expected profit (ставка 1) = p * odds - 1
      Penalty = -Expected profit = 1 - p * odds
    Обираємо клас з мінімальним penalty.
    """
    n = len(df_test)
    pH = np.zeros(n, dtype=float)
    pA = np.zeros(n, dtype=float)

    for i, row in enumerate(df_test.itertuples(index=False)):
        ht = row.HomeTeam
        at = row.AwayTeam

        ph = home_post.get(ht, BetaPosterior(1.0, 1.0)).mean()
        pa = away_post.get(at, BetaPosterior(1.0, 1.0)).mean()

        pH[i] = ph
        pA[i] = pa

    # Для простоти: pD = залишок, далі нормалізація
    pD = np.clip(1.0 - (pH + pA), 0.0, 1.0)

    probs = np.vstack([pH, pD, pA]).T
    probs = probs / probs.sum(axis=1, keepdims=True)

    odds = df_test[["B365H", "B365D", "B365A"]].to_numpy()
    penalty = 1.0 - probs * odds  # Penalty(a) = 1 - p_a * odds_a

    idx = penalty.argmin(axis=1)
    mapping = np.array(["H", "D", "A"])
    pred = mapping[idx]

    return probs, pred


def evaluate_accuracy(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["H", "D", "A"])
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (рядки=true H/D/A, стовпці=pred H/D/A):")
    print(cm)


# 4) Build ROC curves and calculate AUC
def roc_auc_homewin(y_true_ftr: np.ndarray, score_home: np.ndarray, title: str) -> float:
    """
    ROC/AUC у класичному вигляді — для бінарної задачі.
    Тому беремо HomeWin як:
      y = 1 якщо FTR=='H', інакше 0
    score_home = "скор" або "ймовірність" HomeWin.
    """
    y = (y_true_ftr == "H").astype(int)
    fpr, tpr, _ = roc_curve(y, score_home)
    auc = roc_auc_score(y, score_home)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{title} | AUC={auc:.3f}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid(True)
    plt.show()

    return auc


# 5) Engineer predictive features and estimate predictive power
def make_predictive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - implied probabilities з odds:
        qH = (1/B365H) / sum(1/odds)
      аналогічно qD, qA
    """
    out = df.copy()
    invH = 1.0 / out["B365H"].to_numpy()
    invD = 1.0 / out["B365D"].to_numpy()
    invA = 1.0 / out["B365A"].to_numpy()
    s = invH + invD + invA

    out["qH"] = invH / s
    out["qD"] = invD / s
    out["qA"] = invA / s
    return out


def estimate_predictive_power_auc(df: pd.DataFrame) -> None:
    """
    Оцінюємо predictive power (мінімально і зрозуміло) через AUC для HomeWin.
    Беремо фічі qH, qD, qA і дивимось їх AUC як "наскільки вони передбачають HomeWin".
    """
    y = (df["FTR"].to_numpy() == "H").astype(int)

    features = {
        "qH (implied P(Home))": df["qH"].to_numpy(),
        "qD (implied P(Draw))": df["qD"].to_numpy(),
        "qA (implied P(Away))": df["qA"].to_numpy(),
    }

    print("\n--- Predictive power (AUC) для ознак відносно HomeWin ---")
    for name, score in features.items():
        auc = roc_auc_score(y, score)
        print(f"{name}: AUC={auc:.3f}")


# MAIN: виконує всі 5 пунктів
def main() -> None:
    # (1) Завантажили конкретний датасет з open source
    print("Завантаження датасету з:", DATA_URL)
    df = download_dataset()
    df = validate_columns(df)
    print(f"OK. Локальний файл: {LOCAL_CSV}. Рядків після чистки: {len(df)}")
    print(df[["HomeTeam", "AwayTeam", "B365H", "B365D", "B365A", "FTR"]].head())

    # (2) Візуалізація розподілу
    plot_distributions(df)

    # Простий split train/test
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    cut = int(0.8 * len(df))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()

    y_test = test["FTR"].to_numpy()

    # (3) Стратегії + accuracy
    pred_random = strategy_random(test, seed=1)
    pred_host = strategy_host(test)
    pred_visitor = strategy_visitor(test)
    pred_challenger = strategy_challenger_favorite_by_odds(test)

    home_post, away_post = fit_beta_posteriors(train, alpha0=1.0, beta0=1.0)
    probs_bayes, pred_bayes = bayesian_predict_probs_and_decision(test, home_post, away_post)

    evaluate_accuracy("Random strategy", y_test, pred_random)
    evaluate_accuracy("Deterministic: Host", y_test, pred_host)
    evaluate_accuracy("Deterministic: Visitor", y_test, pred_visitor)
    evaluate_accuracy("Deterministic: Challenger (favorite by odds)", y_test, pred_challenger)
    evaluate_accuracy("Bayesian strategy (Beta posterior + penalty decision)", y_test, pred_bayes)

    # (4) ROC/AUC (HomeWin)
    score_challenger_home = (pred_challenger == "H").astype(float)  # 0/1 скор
    score_bayes_home = probs_bayes[:, 0]  # P(H) від Байєса

    roc_auc_homewin(y_test, score_challenger_home, "ROC: Challenger (HomeWin)")
    roc_auc_homewin(y_test, score_bayes_home, "ROC: Bayesian (HomeWin)")

    # (5) Predictive features + їх predictive power
    test_feat = make_predictive_features(test)
    estimate_predictive_power_auc(test_feat)


if __name__ == "__main__":
    main()
