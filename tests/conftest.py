import numpy as np
import pytest

from urkin import UR3e, UR5e, UR10e


@pytest.fixture(params=[UR3e, UR5e, UR10e], ids=["UR3e", "UR5e", "UR10e"])
def robot(request):
    return request.param()


@pytest.fixture
def rng():
    return np.random.default_rng(2026)


def random_configs(rng, n, margin=0.05):
    """Random joint vectors that stay away from the three UR singularities."""
    out = []
    while len(out) < n:
        q = rng.uniform(-np.pi, np.pi, 6)
        if abs(np.sin(q[4])) > margin and abs(np.sin(q[2])) > margin:
            out.append(q)
    return np.array(out)
