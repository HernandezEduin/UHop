#!/usr/bin/env python3
"""Greedy, free-running relation navigation for THESEUS datasets using trained UHop.

This is an inference-only wrapper. UHop itself predicts relations, not unique
entity-edge actions. Answer-set reachability is therefore not Hits@1.
"""
import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from preprocess.prepare_theseus import read_graph


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def tokenize(text, vocabulary, delimiter="."):
    if delimiter == "._":
        text = text.replace("_", ".")
    return [vocabulary.get(token, vocabulary["<unk>"]) for token in text.strip().split("."
            if delimiter == "._" or delimiter == "." else delimiter)]


def pad(rows, length, padding, device):
    return torch.tensor(
        [row + [padding] * (length - len(row)) for row in rows],
        dtype=torch.long, device=device,
    )


def score_candidates(model, question, relations, history, word2id, rela2id,
                     relation_text, args, batch_size):
    """Pack candidates exactly as Framework._single_UHop_step does."""
    if not relations:
        return {}
    device = args.device
    question_ids = [word2id.get(w, word2id["<unk>"]) for w in question.strip().split(" ")]
    relation_ids = [tokenize(rel, rela2id) for rel in relations]
    relation_words = [tokenize(relation_text.get(rel, rel), word2id, "._") for rel in relations]
    previous_ids = [tokenize(rel, rela2id) for rel in history]
    previous_words = [tokenize(relation_text.get(rel, rel), word2id, "._") for rel in history]
    if args.dynamic == "none":
        relation_ids = [sum(previous_ids + [rel], []) for rel in relation_ids]
        relation_words = [sum(previous_words + [rel], []) for rel in relation_words]
    max_rel_id = max(len(rel) for rel in relation_ids)
    max_rel_word = max(len(rel) for rel in relation_words)
    all_scores = []
    for start in range(0, len(relations), batch_size):
        stop = start + batch_size
        width = len(relations[start:stop])
        question_batch = pad([question_ids] * width, max(5, len(question_ids)),
                             word2id["PADDING"], device)
        rela_batch = pad(relation_ids[start:stop], max_rel_id,
                         rela2id["PADDING"], device)
        words_batch = pad(relation_words[start:stop], max_rel_word,
                          word2id["PADDING"], device)
        prev_batches, prev_word_batches = [], []
        if args.dynamic == "flatten" and history:
            flat_ids = sum(previous_ids, [])
            flat_words = sum(previous_words, [])
            prev_batches = [pad([flat_ids] * width, len(flat_ids),
                                rela2id["PADDING"], device)]
            prev_word_batches = [pad([flat_words] * width, len(flat_words),
                                     word2id["PADDING"], device)]
        elif args.dynamic == "recurrent":
            prev_batches = [pad([prev] * width, len(prev),
                                rela2id["PADDING"], device) for prev in previous_ids]
            prev_word_batches = [pad([prev] * width, len(prev),
                                     word2id["PADDING"], device) for prev in previous_words]
        scores = model(question_batch, words_batch, rela_batch,
                       prev_word_batches, prev_batches)
        all_scores.extend(scores.reshape(-1).detach().cpu().tolist())
    return dict(zip(relations, all_scores))


def navigate(model, question, source, graph, word2id, rela2id,
             relation_text, model_args, eval_args):
    frontier = {source}
    history = []
    previous_score = None
    # UHop's learned TD compares best next score to the chosen previous score.
    for _ in range(eval_args.max_hops):
        choices = sorted({rel for node in frontier for rel in graph.get(node, {})})
        if not choices:
            break
        scores = score_candidates(model, question, choices, history,
                                  word2id, rela2id, relation_text, model_args, eval_args.batch_size)
        best_relation = max(choices, key=lambda rel: scores[rel])
        best_score = scores[best_relation]
        if previous_score is not None and best_score <= previous_score:
            break
        history.append(best_relation)
        frontier = {
            tail for node in frontier for tail in graph.get(node, {}).get(best_relation, ())
        }
        previous_score = best_score
    return history, sorted(frontier)


def aggregate(results):
    def one_group(rows):
        count = len(rows)
        return {
            "count": count,
            "relation_chain_accuracy": sum(r["relation_chain_correct"] for r in rows) / count,
            "answer_reachability": sum(r["answer_reachable"] for r in rows) / count,
            "answer_set_exact_match": sum(r["answer_set_exact"] for r in rows) / count,
            "mean_terminal_frontier_size": sum(r["terminal_frontier_size"] for r in rows) / count,
        }
    by_hop = defaultdict(list)
    for result in results:
        by_hop[str(result["hops"])].append(result)
    return {
        "all": one_group(results),
        "by_hop": {hop: one_group(rows) for hop, rows in sorted(by_hop.items())},
        "metric_warning": (
            "Answer reachability means any gold entity remains in the terminal "
            "frontier. It is not Hits@1, because UHop predicts relations rather "
            "than choosing one entity edge."
        ),
    }


def evaluate(args):
    checkpoint = Path(args.checkpoint).resolve()
    model_args = json.loads((checkpoint / "args.txt").read_text(encoding="utf-8"))
    dataset = model_args["dataset"].lower()
    if dataset not in ("kinship", "mquake_st_single", "mquake_st_multi"):
        raise ValueError("Checkpoint is not a THESEUS dataset run")
    if model_args.get("q_representation") != "lstm":
        raise ValueError("This wrapper supports UHop's original LSTM question mode")
    model_args["device"] = args.device
    if model_args.get("data_dir"):
        data_dir = Path(model_args["data_dir"])
        if not data_dir.is_absolute():
            data_dir = REPO / "src" / data_dir
    else:
        data_dir = REPO / "data" / "THESEUS" / dataset
    data_dir = data_dir.resolve()
    model_args["word_embedding"] = np.load(data_dir / "word_emb.npy")
    word2id = json.loads((data_dir / "word2id.json").read_text(encoding="utf-8"))
    rela2id = json.loads((data_dir / "rela2id.json").read_text(encoding="utf-8"))
    relation_text = json.loads((data_dir / "relation_text.json").read_text(encoding="utf-8"))
    model_args["rela_vocab_size"] = len(rela2id)
    original = SimpleNamespace(**model_args)
    module = importlib.import_module("model." + original.model)
    model = module.Model(original).to(args.device)
    model.load_state_dict(torch.load(checkpoint / "model.pth", map_location=args.device))
    model.eval()
    source = json.loads((data_dir / "source.json").read_text(encoding="utf-8"))
    kg_file = source["kg_file"]
    if args.offline_source:
        graph_path = args.offline_source / kg_file
    else:
        from huggingface_hub import hf_hub_download
        graph_path = hf_hub_download(
            source["huggingface_repo"], kg_file, repo_type="dataset",
            revision=source["revision"]
        )
    graph, _, _ = read_graph(graph_path)
    rows = [json.loads(line) for line in
            (data_dir / f"{args.split}_metadata.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("Evaluation split is empty")
    results = []
    with torch.no_grad():
        for index, row in enumerate(rows, 1):
            predicted, terminal = navigate(
                model, row["question"], row["source"], graph, word2id, rela2id,
                relation_text, original, args
            )
            gold = set(row["answers"])
            found = set(terminal)
            results.append({
                "question_number": row["question_number"],
                "hops": row["hops"],
                "predicted_relations": predicted,
                "gold_relations": row["gold_relations"],
                "relation_chain_correct": predicted == row["gold_relations"],
                "answer_reachable": bool(gold & found),
                "answer_set_exact": gold == found,
                "terminal_frontier_size": len(terminal),
                "terminal_entities": terminal,
                "gold_answers": row["answers"],
            })
            if index % 100 == 0:
                print(f"Evaluated {index}/{len(rows)}", flush=True)
    summary = aggregate(results)
    output = args.output or checkpoint / f"{args.split}_navigation"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix(".jsonl"), "w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="UHop saved_model directory containing model.pth and args.txt")
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--offline-source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N examples for a smoke test")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.batch_size <= 0 or args.max_hops <= 0:
        raise ValueError("max-hops and batch-size must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
