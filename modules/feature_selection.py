"""Feature-selection methods, including a self-contained GA selector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score


@dataclass
class GAResult:
    selected_descriptors: list[str]
    best_score: float
    history: pd.DataFrame


def _scoring_name(metric: str) -> str:
    if metric in {"R2", "Q2 / CV R2", "External validation R2", "Combined score"}:
        return "r2"
    if metric == "RMSE":
        return "neg_root_mean_squared_error"
    if metric == "MAE":
        return "neg_mean_absolute_error"
    return "r2"


def _repair_mask(mask: np.ndarray, rng: np.random.Generator, min_features: int, max_features: int) -> np.ndarray:
    repaired = mask.astype(bool).copy()
    n_features = len(repaired)
    min_features = max(1, min(min_features, n_features))
    max_features = max(min_features, min(max_features, n_features))
    selected = np.flatnonzero(repaired)
    if len(selected) < min_features:
        missing = np.setdiff1d(np.arange(n_features), selected)
        repaired[rng.choice(missing, size=min_features - len(selected), replace=False)] = True
    elif len(selected) > max_features:
        repaired[rng.choice(selected, size=len(selected) - max_features, replace=False)] = False
    return repaired


def run_genetic_algorithm_selection(
    X: pd.DataFrame,
    y: pd.Series,
    estimator_factory: Callable[[], object],
    population_size: int = 30,
    generations: int = 20,
    crossover_probability: float = 0.8,
    mutation_probability: float = 0.05,
    tournament_size: int = 3,
    min_descriptors: int = 1,
    max_descriptors: int | None = None,
    scoring_metric: str = "Q2 / CV R2",
    cv_folds: int = 5,
    random_seed: int = 42,
) -> GAResult:
    """Run a compact GA over descriptor bitmasks using cross-validation fitness."""

    if X.empty:
        raise ValueError("GA feature selection requires at least one descriptor.")
    rng = np.random.default_rng(random_seed)
    n_features = X.shape[1]
    max_descriptors = max_descriptors or n_features
    max_descriptors = min(max_descriptors, n_features)
    min_descriptors = min(max(1, min_descriptors), max_descriptors)
    population_size = max(4, int(population_size))
    tournament_size = max(2, min(int(tournament_size), population_size))
    cv_folds = max(2, min(int(cv_folds), len(y)))
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    scoring = _scoring_name(scoring_metric)
    cache: dict[tuple[bool, ...], float] = {}

    def random_mask() -> np.ndarray:
        n_selected = rng.integers(min_descriptors, max_descriptors + 1)
        mask = np.zeros(n_features, dtype=bool)
        mask[rng.choice(np.arange(n_features), size=n_selected, replace=False)] = True
        return mask

    def fitness(mask: np.ndarray) -> float:
        mask = _repair_mask(mask, rng, min_descriptors, max_descriptors)
        key = tuple(mask.tolist())
        if key in cache:
            return cache[key]
        X_subset = X.loc[:, mask]
        try:
            estimator = estimator_factory(X_subset.shape[1])
        except TypeError:
            estimator = estimator_factory()
        try:
            scores = cross_val_score(clone(estimator), X_subset, y, cv=cv, scoring=scoring)
            score = float(np.nanmean(scores))
        except Exception:
            score = -np.inf
        if scoring_metric == "Combined score" and np.isfinite(score):
            score = score - 0.002 * int(mask.sum())
        cache[key] = score
        return score

    def tournament(population: list[np.ndarray], scores: list[float]) -> np.ndarray:
        candidates = rng.choice(np.arange(len(population)), size=tournament_size, replace=False)
        best_idx = max(candidates, key=lambda idx: scores[idx])
        return population[best_idx].copy()

    population = [random_mask() for _ in range(population_size)]
    history_rows: list[dict[str, float]] = []

    for generation in range(generations + 1):
        scores = [fitness(mask) for mask in population]
        best_idx = int(np.nanargmax(scores))
        finite_scores = [score for score in scores if np.isfinite(score)]
        history_rows.append(
            {
                "generation": generation,
                "best_score": float(scores[best_idx]),
                "mean_score": float(np.mean(finite_scores)) if finite_scores else np.nan,
                "best_descriptor_count": int(population[best_idx].sum()),
            }
        )
        if generation == generations:
            break

        new_population = [population[best_idx].copy()]
        while len(new_population) < population_size:
            parent1 = tournament(population, scores)
            parent2 = tournament(population, scores)
            child1, child2 = parent1.copy(), parent2.copy()
            if rng.random() < crossover_probability and n_features > 1:
                point = rng.integers(1, n_features)
                child1[:point], child2[:point] = parent2[:point], parent1[:point]
            for child in (child1, child2):
                mutation_mask = rng.random(n_features) < mutation_probability
                child[mutation_mask] = ~child[mutation_mask]
                child = _repair_mask(child, rng, min_descriptors, max_descriptors)
                new_population.append(child)
                if len(new_population) >= population_size:
                    break
        population = new_population

    final_scores = [fitness(mask) for mask in population]
    best_idx = int(np.nanargmax(final_scores))
    best_mask = _repair_mask(population[best_idx], rng, min_descriptors, max_descriptors)
    selected = X.columns[best_mask].tolist()
    return GAResult(selected, float(final_scores[best_idx]), pd.DataFrame(history_rows))


@dataclass
class FeatureSelector(BaseEstimator, TransformerMixin):
    method: str = "None"
    params: dict = field(default_factory=dict)
    selected_descriptors_: list[str] = field(default_factory=list)
    ga_history_: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator_factory: Callable[[], object] | None = None,
    ) -> "FeatureSelector":
        X = pd.DataFrame(X).copy()
        self.input_descriptors_ = X.columns.astype(str).tolist()
        method = self.method

        if method == "None":
            self.selected_descriptors_ = self.input_descriptors_
        elif method == "Manual":
            requested = [str(col) for col in self.params.get("manual_descriptors", [])]
            self.selected_descriptors_ = [col for col in requested if col in X.columns]
            if not self.selected_descriptors_:
                raise ValueError("Manual feature selection did not include any available descriptors.")
        elif method == "Variance threshold":
            threshold = float(self.params.get("threshold", 0.0))
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X, y)
            self.selected_descriptors_ = X.columns[selector.get_support()].tolist()
        elif method == "SelectKBest":
            k = min(int(self.params.get("k", min(10, X.shape[1]))), X.shape[1])
            selector = SelectKBest(score_func=f_regression, k=k)
            selector.fit(X, y)
            self.selected_descriptors_ = X.columns[selector.get_support()].tolist()
        elif method == "RFE":
            n_features = min(int(self.params.get("n_features", min(10, X.shape[1]))), X.shape[1])
            selector = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
            selector.fit(X, y)
            self.selected_descriptors_ = X.columns[selector.get_support()].tolist()
        elif method == "Genetic Algorithm":
            if estimator_factory is None:
                raise ValueError("GA feature selection requires an estimator factory.")
            result = run_genetic_algorithm_selection(
                X,
                y,
                estimator_factory=estimator_factory,
                population_size=int(self.params.get("population_size", 30)),
                generations=int(self.params.get("generations", 20)),
                crossover_probability=float(self.params.get("crossover_probability", 0.8)),
                mutation_probability=float(self.params.get("mutation_probability", 0.05)),
                tournament_size=int(self.params.get("tournament_size", 3)),
                min_descriptors=int(self.params.get("min_descriptors", 1)),
                max_descriptors=int(self.params.get("max_descriptors", X.shape[1])),
                scoring_metric=self.params.get("scoring_metric", "Q2 / CV R2"),
                cv_folds=int(self.params.get("cv_folds", 5)),
                random_seed=int(self.params.get("random_seed", 42)),
            )
            self.selected_descriptors_ = result.selected_descriptors
            self.ga_history_ = result.history
            self.ga_best_score_ = result.best_score
        else:
            raise ValueError(f"Unsupported feature-selection method: {method}")

        if not self.selected_descriptors_:
            raise ValueError("No descriptors were selected.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_descriptors_:
            raise RuntimeError("FeatureSelector has not been fitted yet.")
        X = pd.DataFrame(X).copy()
        missing = [col for col in self.selected_descriptors_ if col not in X.columns]
        if missing:
            for col in missing:
                X[col] = np.nan
        return X.loc[:, self.selected_descriptors_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator_factory: Callable[[], object] | None = None,
    ) -> pd.DataFrame:
        return self.fit(X, y, estimator_factory=estimator_factory).transform(X)

    def selected_descriptors_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"descriptor": self.selected_descriptors_})
