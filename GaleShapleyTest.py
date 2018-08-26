from hypothesis import given
from hypothesis.strategies import text, integers, lists, composite

import GaleShapley as gs
import copy


@composite
def preference_list(draw, b=2, t=100):
    x = draw(integers(b, t))
    l = list(range(x))
    return [draw(lists(integers(0, x-1), min_size=0, max_size=x, unique=True))
            for y in range(x)]


@given(preference_list())
def test_gale_shapely_runs(x):
    pref = {}
    rs = gs.GaleShapley(pref, x)

    xs = list(rs.keys())
    ys = list(rs.values())
    xs.sort()
    ys.sort()

    assert len(rs) == len(x)
    assert xs == ys



def test_gale_shapley_1():
    ps = [[], [4], [4, 2, 1, 3, 0], [1, 4, 2, 3, 0], []]
    rs = [(0, 0), (1, 4), (2, 2), (3, 1), (4, 3)]
    pref = {}

    rx = [(a,b) for (b,a) in gs.GaleShapley(pref, ps).items()]
    rx.sort()
    print(rx)
    assert rx == rs




