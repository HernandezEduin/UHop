"""Small deterministic checks for independent THESEUS navigation baselines."""
import sys
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "script"))
from evaluate_theseus_baselines import (legal_edges, walk, shortest,
                                       training_stop_probability, score)


def example_graph():
    return {
        "a": {"r": {"a", "b"}, "no_operation": {"a"}},
        "b": {"s": {"c"}, "t": {"a"}},
        "c": {},
    }


def test_no_op_and_self_edges_excluded():
    assert legal_edges(example_graph(), "a") == [("r", "b")]


def test_walk_stop_policy_and_dead_end():
    graph = example_graph()
    path, reason = walk(graph, "a", 5, random.Random(1), 1.0)
    assert path == [("a", "r", "b")]
    assert reason == "random_termination"
    path, reason = walk(graph, "a", 1, random.Random(1), 0.5)
    assert len(path) == 1 and reason == "max_hops"
    path, reason = walk(graph, "c", 5, random.Random(1), 0.5)
    assert path == [] and reason == "no_outgoing_relations"


def test_oracle_shortest_and_cap():
    graph = example_graph()
    assert shortest(graph, "a", {"c"}, 2) == (
        [("a", "r", "b"), ("b", "s", "c")], "oracle_answer"
    )
    assert shortest(graph, "a", {"c"}, 1) == (None, "no_path_within_budget")


def test_train_probability_and_path_metrics():
    assert training_stop_probability([{"hops": 2}, {"hops": 3}]) == 0.4
    row = {"source": "a", "gold_relations": ["r", "s"],
           "reference_paths": [[("a", "r", "b"), ("b", "s", "c")]]}
    result = score([("a", "r", "b"), ("b", "s", "c")], row, {"c"})
    assert result["hits_at_1"] == 1
    assert result["PED"] == 0 and result["F1_SG"] == 1
