"""CSE-307 Track 1: learning-augmented page replacement.

Run this file to reproduce the experiment, write CSV tables, and generate charts.
The experiment is deterministic so the reported numbers can be checked.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FRAMES = 4
SEED = 307


@dataclass
class PolicyResult:
    policy: str
    phase: str
    references: int
    hits: int
    faults: int
    hit_ratio: float


def make_trace(seed: int = SEED, phase_length: int = 400) -> tuple[list[int], list[str]]:
    """Create a locality-heavy first phase and random/bursty second phase."""
    rng = np.random.default_rng(seed)
    first = []
    for i in range(phase_length):
        if i % 17 < 12:
            first.append(int(rng.choice([0, 1, 2, 3, 4], p=[0.35, 0.25, 0.20, 0.12, 0.08])))
        else:
            first.append(int((i // 17) % 10))

    second = []
    for i in range(phase_length):
        if i % 20 < 8:
            burst_base = 5 + ((i // 20) % 2) * 3
            second.append(int(rng.choice([burst_base, burst_base + 1, burst_base + 2])))
        else:
            second.append(int(rng.integers(0, 10)))
    return first + second, ["locality"] * phase_length + ["random_bursty"] * phase_length


def next_use(trace: list[int], position: int, page: int) -> int:
    try:
        return trace.index(page, position + 1)
    except ValueError:
        return len(trace) + 1


def simulate(
    trace: list[int],
    phases: list[str],
    policy: str,
    frame_count: int = FRAMES,
    learned_selector: Callable[[list[int], int, list[int]], int] | None = None,
) -> list[PolicyResult]:
    frames: list[int] = []
    fifo_order: deque[int] = deque()
    last_used: dict[int, int] = {}
    frequency: Counter[int] = Counter()
    hits = Counter[str]()
    faults = Counter[str]()
    for position, page in enumerate(trace):
        phase = phases[position]
        frequency[page] += 1
        if page in frames:
            hits[phase] += 1
            last_used[page] = position
            continue

        faults[phase] += 1
        if len(frames) >= frame_count:
            if policy == "FIFO":
                victim = fifo_order.popleft()
            elif policy == "LRU":
                victim = min(frames, key=lambda candidate: last_used.get(candidate, -1))
            elif policy == "Optimal":
                victim = max(frames, key=lambda candidate: next_use(trace, position, candidate))
            elif policy == "Learned":
                if learned_selector is None:
                    raise ValueError("Learned policy requires a selector")
                victim = learned_selector(frames, position, trace)
            else:
                raise ValueError(f"Unknown policy: {policy}")
            frames.remove(victim)

        frames.append(page)
        fifo_order.append(page)
        last_used[page] = position

    output = []
    for phase in ("locality", "random_bursty"):
        references = phases.count(phase)
        hit_ratio = hits[phase] / references
        output.append(PolicyResult(policy, phase, references, hits[phase], faults[phase], hit_ratio))
    return output


def candidate_features(frames: list[int], position: int, trace: list[int], candidate: int) -> list[float]:
    """Features available at decision time; no future information is used."""
    recent_positions = [i for i in range(max(0, position - 40), position) if trace[i] == candidate]
    age = position - max(recent_positions, default=-1)
    frequency = sum(1 for i in range(position) if trace[i] == candidate)
    recency_rank = sorted(
        frames,
        key=lambda p: position - max((i for i in range(position) if trace[i] == p), default=-1),
    ).index(candidate)
    return [float(age), float(frequency), float(recency_rank), float(candidate)]


def train_learned_selector(training_trace: list[int]) -> Callable[[list[int], int, list[int]], int]:
    """Train a small tree to predict the candidate with the farthest next use."""
    x: list[list[float]] = []
    y: list[int] = []
    frames: list[int] = []
    for position, page in enumerate(training_trace):
        if page not in frames and len(frames) >= FRAMES:
            victim = max(frames, key=lambda candidate: next_use(training_trace, position, candidate))
            for candidate in frames:
                x.append(candidate_features(frames, position, training_trace, candidate))
                y.append(int(candidate == victim))
            frames.remove(victim)
        if page not in frames:
            frames.append(page)

    model = DecisionTreeClassifier(max_depth=4, random_state=SEED, class_weight="balanced")
    model.fit(x, y)

    def select(frames: list[int], position: int, trace: list[int]) -> int:
        features = [candidate_features(frames, position, trace, candidate) for candidate in frames]
        probabilities = model.predict_proba(features)
        positive_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
        scores = probabilities[:, positive_index]
        return frames[int(np.argmax(scores))]

    return select


def run_experiment() -> pd.DataFrame:
    RESULTS.mkdir(exist_ok=True)
    training_trace, _ = make_trace(seed=SEED + 1, phase_length=400)
    trace, phases = make_trace(seed=SEED, phase_length=400)
    selector = train_learned_selector(training_trace)

    rows = []
    for policy in ("FIFO", "LRU", "Optimal", "Learned"):
        rows.extend(simulate(trace, phases, policy, learned_selector=selector))
    results = pd.DataFrame([row.__dict__ for row in rows])
    results.to_csv(RESULTS / "policy_results.csv", index=False)
    pd.DataFrame({"position": range(len(trace)), "page": trace, "phase": phases}).to_csv(
        RESULTS / "trace.csv", index=False
    )
    create_charts(results)
    return results


def create_charts(results: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    pivot = results.pivot(index="policy", columns="phase", values="hit_ratio")
    ax = pivot.plot(kind="bar", figsize=(8, 5), color=["#4472C4", "#ED7D31"])
    ax.set_title("Hit ratio before and after workload shift")
    ax.set_ylabel("Hit ratio")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(title="Phase")
    plt.tight_layout()
    plt.savefig(RESULTS / "hit_ratio_by_phase.png", dpi=180)
    plt.close()

    faults = results.pivot(index="policy", columns="phase", values="faults")
    ax = faults.plot(kind="bar", figsize=(8, 5), color=["#70AD47", "#A5A5A5"])
    ax.set_title("Page faults before and after workload shift")
    ax.set_ylabel("Page faults")
    ax.set_xlabel("")
    ax.legend(title="Phase")
    plt.tight_layout()
    plt.savefig(RESULTS / "faults_by_phase.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    result = run_experiment()
    print(result.to_string(index=False))
