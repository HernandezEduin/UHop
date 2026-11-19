#!/usr/bin/env python3
"""Gold-blind Monte Carlo random walks and answer-oracle BFS for THESEUS.

Both operate on *entity edges* of the released directed KG (unlike UHop's
relation-frontier predictions). Neither loads or alters UHop model weights.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import random

from preprocess.prepare_theseus import read_graph
from theseus_evaluation_metrics import relation_edit_distance, relation_f1


REPO = Path(__file__).resolve().parents[1]


def legal_edges(graph, entity):
    """Return stable, real outgoing edges; no synthetic NO_OP or self-loops."""
    return sorted((str(relation), str(tail))
                  for relation, tails in graph.get(entity, {}).items()
                  if relation.lower() not in {"no_op", "no-op", "noop"}
                  for tail in tails if tail != entity)


def walk(graph, source, max_hops, rng, stop_probability):
    current, path = source, []
    for step in range(max_hops):
        edges = legal_edges(graph, current)
        if not edges:
            return path, "no_outgoing_relations"
        # At least one hop is taken, consistent with original UHop's RC/TD order.
        if step and rng.random() < stop_probability:
            return path, "random_termination"
        relation, tail = edges[rng.randrange(len(edges))]
        path.append((current, relation, tail))
        current = tail
    return path, "max_hops"


def shortest(graph, source, answers, max_hops=None):
    """Deterministic BFS, shortest path to any answer; oracle uses answer labels."""
    if source in answers:
        return [], "answer_at_source"
    queue = deque([(source, [])])
    visited = {source}
    while queue:
        current, path = queue.popleft()
        if max_hops is not None and len(path) >= max_hops:
            continue
        for relation, tail in legal_edges(graph, current):
            if tail in visited:
                continue
            next_path = path + [(current, relation, tail)]
            if tail in answers:
                return next_path, "oracle_answer"
            visited.add(tail)
            queue.append((tail, next_path))
    return None, "no_path_within_budget" if max_hops is not None else "no_path"


def training_stop_probability(rows):
    """Geometric hazard MLE from TRAIN ONLY: p = N / sum(gold hop counts)."""
    hops = [int(row["hops"]) for row in rows]
    if not hops or any(h < 1 for h in hops):
        raise ValueError("Training data must have strictly positive hop counts")
    return len(hops) / sum(hops)


def sequence_distance(left, right):
    previous = list(range(len(right) + 1))
    for index, item in enumerate(left, 1):
        current = [index]
        for j, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (item != other)))
        previous = current
    return previous[-1]


def subgraph_f1(left, right):
    a, b = set(left), set(right)
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0


def score(path, row, answers):
    relations = [step[1] for step in path] if path is not None else []
    chains = row.get("gold_relation_chains") or [row["gold_relations"]]
    best = min(chains, key=lambda chain: (relation_edit_distance(relations, chain),
                                         -relation_f1(relations, chain)))
    endpoint = path[-1][2] if path else row["source"]
    references = [[tuple(map(str, edge)) for edge in reference]
                  for reference in row.get("reference_paths", [])]
    # Single-answer paths are explicit. Multi-answer datasets can have
    # additional semantically valid paths beyond these supplied references.
    ped = min((sequence_distance(path, reference) for reference in references),
              default=None) if path is not None else None
    f1_sg = max((subgraph_f1(path, reference) for reference in references),
                default=None) if path is not None else None
    return {
        "predicted_relations": relations,
        "predicted_path": [list(edge) for edge in path] if path is not None else None,
        "predicted_hops": len(path) if path is not None else None,
        "selected_answer": endpoint if path is not None else None,
        "hits_at_1": int(path is not None and endpoint in answers),
        "relation_chain_correct": path is not None and relations in chains,
        "RED": relation_edit_distance(relations, best) if path is not None else None,
        "F1_REL": max(relation_f1(relations, chain) for chain in chains)
                  if path is not None else None,
        "PED": ped,
        "F1_SG": f1_sg,
    }


def summarize(rows):
    fields = ("hits_at_1", "relation_chain_correct", "RED", "F1_REL",
              "PED", "F1_SG", "predicted_hops")
    result = {"count": len(rows), "question_count": len({str(r["question_number"]) for r in rows})}
    for field in fields:
        values = [float(r[field]) for r in rows if r[field] is not None]
        result[field] = sum(values) / len(values) if values else None
    result["stop_reasons"] = dict(sorted(Counter(r["stop_reason"] for r in rows).items()))
    return result


def load_dataset(data_dir, offline_source):
    source = json.loads((data_dir / "source.json").read_text(encoding="utf-8"))
    if source.get("local_kg_file"):
        root = offline_source or Path(source["reconstruction_root"])
        graph_path = root / source["local_kg_file"]
    elif offline_source:
        graph_path = offline_source / source["kg_file"]
    else:
        from huggingface_hub import hf_hub_download
        graph_path = Path(hf_hub_download(source["huggingface_repo"], source["kg_file"],
                                               repo_type="dataset", revision=source["revision"]))
    graph, _, _ = read_graph(graph_path)
    return graph, source


def metadata(data_dir, split):
    return [json.loads(line) for line in (data_dir / f"{split}_metadata.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]


def evaluate(args):
    data_dir = args.data_dir.resolve()
    graph, source = load_dataset(data_dir, args.offline_source)
    rows = metadata(data_dir, args.split)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Empty evaluation split")
    p = args.stop_probability
    if args.baseline == "random_walk" and args.stop_policy == "train":
        p = training_stop_probability(metadata(data_dir, "train"))
    elif args.baseline == "random_walk" and args.stop_policy == "fixed":
        p = 0.5 if p is None else p
    else:
        p = 0.0
    if not 0 <= p <= 1:
        raise ValueError("Stop probability must lie in [0, 1]")

    results = []
    for row in rows:
        answers = set(row["graph_answers"] if args.answer_reference == "graph"
                      else row["answers"])
        if args.baseline == "shortest_path_oracle":
            trajectory, reason = shortest(graph, row["source"], answers,
                                          args.max_hops if args.cap_oracle else None)
            trials = [(trajectory, reason)]
        else:
            trials = []
            for trial in range(args.num_walks):
                key = json.dumps([args.seed, str(row["question_number"]), trial])
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
                trials.append(walk(graph, row["source"], args.max_hops,
                                   random.Random(seed), p))
        for trial, (trajectory, reason) in enumerate(trials):
            results.append({
                "question_number": row["question_number"], "question": row["question"],
                "source": row["source"], "gold_answers": sorted(answers),
                "gold_relations": row["gold_relations"], "hops": row["hops"],
                "trial": trial, "stop_reason": reason,
                **score(trajectory, row, answers),
            })
    by_hop = defaultdict(list)
    for result in results:
        by_hop[str(result["hops"])].append(result)
    summary = {
        "baseline": args.baseline, "dataset": data_dir.name, "split": args.split,
        "seed": args.seed, "max_hops": args.max_hops,
        "num_walks": args.num_walks if args.baseline == "random_walk" else 1,
        "stop_policy": args.stop_policy if args.baseline == "random_walk" else "oracle_answer",
        "stop_probability": p if args.baseline == "random_walk" else None,
        "cap_oracle": args.cap_oracle if args.baseline == "shortest_path_oracle" else None,
        "answer_reference": args.answer_reference,
        "graph_regime": source.get("graph_regime", "directed"),
        "self_loops_excluded": True,
        "all": summarize(results),
        "by_hop": {hop: summarize(group) for hop, group in sorted(by_hop.items())},
        "metric_warning": (
            "Random-walk averages are Monte Carlo expectations over independent entity-edge "
            "rollouts, not an oracle-selected best walk. The shortest-path baseline uses "
            "gold answer entities and is an oracle upper bound, not a deployable predictor. "
            "Unreachable oracle questions have null fidelity metrics. "
            "PED and F1_SG use available explicitly released reference paths; "
            "multi-answer semantic-path expansion is not performed."
        ),
    }
    output = args.output or data_dir / f"{args.split}_{args.baseline}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, choices=("random_walk", "shortest_path_oracle"))
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Prepared data/THESEUS/<dataset> directory")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--offline-source", type=Path)
    parser.add_argument("--answer-reference", choices=("released", "graph"), default="released")
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--cap-oracle", action="store_true",
                        help="Restrict oracle BFS to max-hops; otherwise search full graph")
    parser.add_argument("--num-walks", type=int, default=20)
    parser.add_argument("--stop-policy", choices=("fixed", "train", "none"), default="fixed")
    parser.add_argument("--stop-probability", type=float,
                        help="For fixed policy; default 0.5. Ignored by train/none.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        help="Output prefix for .jsonl and .summary.json")
    args = parser.parse_args()
    if args.max_hops < 1 or args.num_walks < 1 or args.limit < 0:
        parser.error("max-hops and num-walks must be positive; limit cannot be negative")
    if args.stop_policy != "fixed" and args.stop_probability is not None:
        parser.error("--stop-probability is only meaningful with --stop-policy fixed")
    evaluate(args)


if __name__ == "__main__":
    main()
