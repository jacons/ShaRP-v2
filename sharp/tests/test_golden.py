"""Outputs must match the snapshots frozen from baseline commit 6543226."""

import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import sharp
from sharp.tests.make_golden import CASES, DATASETS, run_case

EXTRA: dict = {}
GOLDEN = np.load(os.path.join(os.path.dirname(__file__), "data", "golden_6543226.npz"))


def test_datasets():
    for name, (X, _) in DATASETS.items():
        assert_array_equal(GOLDEN[f"X/{name}"], X)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_golden(case):
    for key, val in run_case(sharp, case, **EXTRA).items():
        assert_array_equal(GOLDEN[f"{case['id']}/{key}"], val, err_msg=key, strict=True)
