"""Generate golden snapshots (run against the baseline sharp via PYTHONPATH)."""

import os
from itertools import product

import numpy as np


def _gauss3():
    X = np.random.RandomState(0).normal(size=(30, 3))
    return X, lambda X: 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]


def _normbern():
    rs = np.random.RandomState(0)
    X = np.concatenate([rs.normal(size=(30, 1)), rs.binomial(1, 0.5, size=(30, 1))], axis=1)
    return X, lambda X: 0.5 * X[:, 0] + 0.5 * X[:, 1]


def _int4():
    X = np.random.RandomState(0).randint(0, 4, size=(25, 4)).astype(float)
    return X, lambda X: X[:, 0] + 2 * X[:, 1] + X[:, 2] + X[:, 3]


DATASETS = {"gauss3": _gauss3(), "normbern": _normbern(), "int4": _int4()}

CASES = []
for _ds, (_qoi, _k), (_ss, _rep), _cs, _seed in product(
    DATASETS,
    [("rank", None), ("top_k", 1), ("top_k", 5), ("top_k", 10), ("rank_score", None)],
    [(None, False), (7, False), (7, True)],
    [None, 1],
    [0, 123],
):
    if _cs == 1 and _ds == "normbern":
        continue
    CASES.append(
        dict(
            id=f"{_ds}-{_qoi}{_k or ''}-ss{_ss}-rep{int(_rep)}-cs{_cs}-seed{_seed}",
            dataset=_ds, qoi=_qoi, top_k=_k, sample_size=_ss,
            replace=_rep, coalition_size=_cs, seed=_seed,
        )
    )


def run_case(sharp_module, case, **extra):
    X, f = DATASETS[case["dataset"]]
    kw = {"top_k": case["top_k"]} if case["top_k"] else {}
    s = sharp_module.ShaRP(
        qoi=case["qoi"], target_function=f, measure="shapley",
        sample_size=case["sample_size"], coalition_size=case["coalition_size"],
        replace=case["replace"], random_state=case["seed"], n_jobs=1, **kw, **extra,
    )
    s.fit(X)
    out = {"all": s.all(X)}
    for i in (0, 7, 13):
        out[f"ind{i}"] = s.individual(i, X)
    out["feat0"] = s.feature(0, X)
    out["pw"] = s.pairwise(X[2], X[3])
    out["pwgroup"] = s.pairwise(X[2], X[5:10])
    out["pwset"] = s.pairwise_set([X[2], X[2], X[4]], [X[3], X[4], X[2]])
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def main():
    import sharp

    print("sharp from", sharp.__file__)
    data = {f"X/{n}": X for n, (X, _) in DATASETS.items()}
    for c in CASES:
        for k, v in run_case(sharp, c).items():
            data[f"{c['id']}/{k}"] = v
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "golden_6543226.npz")
    np.savez_compressed(path, **data)
    print("cases", len(CASES), "wrote", path)


if __name__ == "__main__":
    main()
