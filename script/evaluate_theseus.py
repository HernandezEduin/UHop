#!/usr/bin/env python3
"""Greedy, free-running relation navigation for THESEUS datasets using trained UHop.

This is an inference-only wrapper. UHop itself predicts relations, not unique
entity-edge actions. Answer-set reachability is therefore not Hits@1.
"""
import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from preprocess.prepare_theseus import (
    make_embeddings, read_graph, read_release_file,
)
from theseus_evaluation_metrics import (
    aggregate, expected_uniform_hits_at_1, load_paraphrases, question_variants,
    relation_edit_distance, relation_f1, select_entity,
)


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


def extend_paraphrase_vocabulary(model, word2id, questions, source, args):
    """Append frozen GloVe vectors for unseen paraphrase tokens after loading the checkpoint.

    Existing word IDs and all trained weights remain unchanged.
    """
    needed = {
        token for question in questions for token in question.strip().split(" ")
        if token not in word2id
    }
    if not needed:
        print("Paraphrase vocabulary already covered by the checkpoint.")
        return
    glove_source = source.get("glove_source", "huggingface")
    if glove_source == "local" and args.glove is None:
        raise ValueError(
            "This checkpoint uses local GloVe. Pass --glove /path/to/glove.txt "
            "to initialize unseen paraphrase tokens."
        )
    with tempfile.TemporaryDirectory(prefix="uhop-paraphrase-") as temp:
        directory = Path(temp)
        make_embeddings(
            directory, questions, {}, args.glove,
            model.word_embedding.weight.shape[1],
            glove_source=glove_source,
            glove_model=source.get("glove_model") or "NeuML/glove-6B",
            glove_revision=source.get("glove_revision") or "main",
        )
        lookup = json.loads((directory / "word2id.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "word_emb.npy")
        new_tokens = sorted(needed)
        extra = vectors[[lookup[token] for token in new_tokens]].copy()
    previous = model.word_embedding.weight.detach()
    additional = torch.from_numpy(extra).to(device=previous.device, dtype=previous.dtype)
    model.word_embedding = torch.nn.Embedding.from_pretrained(
        torch.cat((previous, additional), dim=0), freeze=True
    )
    word2id.update({
        token: index for index, token in enumerate(new_tokens, len(word2id))
    })
    print(f"Added {len(new_tokens)} frozen paraphrase tokens at inference time.")


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
    if args.question_variant == "paraphrased":
        qa_path = read_release_file(
            dataset, source["qa_file"], args.offline_source, source["revision"]
        )
        with open(qa_path, encoding="utf-8-sig", newline="") as stream:
            paraphrases = load_paraphrases(stream, rows)
    else:
        paraphrases = None
    variants = question_variants(
        rows, args.question_variant, paraphrases, args.paraphrase_index
    )
    if args.question_variant == "paraphrased":
        extend_paraphrase_vocabulary(
            model, word2id, [question for _, question, _ in variants], source, args
        )
    results = []
    with torch.no_grad():
        for index, (row, question, variant_index) in enumerate(variants, 1):
            predicted, terminal = navigate(
                model, question, row["source"], graph, word2id, rela2id,
                relation_text, original, args
            )
            gold = set(row["answers"])
            found = set(terminal)
            selected = select_entity(
                terminal, dataset, row["question_number"], variant_index,
                selector=args.entity_selector, seed=args.seed,
            )
            results.append({
                "question_number": row["question_number"],
                "question": question,
                "question_variant": args.question_variant,
                "paraphrase_index": variant_index,
                "hops": row["hops"],
                "predicted_relations": predicted,
                "gold_relations": row["gold_relations"],
                "relation_chain_correct": predicted == row["gold_relations"],
                "RED": relation_edit_distance(predicted, row["gold_relations"]),
                "F1_REL": relation_f1(predicted, row["gold_relations"]),
                "selected_answer": selected,
                "hits_at_1": int(selected in gold),
                "expected_hits_at_1_uniform": expected_uniform_hits_at_1(
                    terminal, gold
                ),
                "answer_reachable": bool(gold & found),
                "answer_set_exact": gold == found,
                "terminal_frontier_size": len(terminal),
                "terminal_entities": terminal,
                "gold_answers": row["answers"],
            })
            if index % 100 == 0:
                print(f"Evaluated {index}/{len(variants)}", flush=True)
    summary = aggregate(
        results, args.question_variant, args.entity_selector, args.seed
    )
    if args.output:
        output = args.output
    elif args.question_variant == "canonical":
        output = checkpoint / f"{args.split}_navigation"
    else:
        suffix = ("" if args.paraphrase_index == "all"
                  else f"_{args.paraphrase_index}")
        output = checkpoint / f"{args.split}_navigation_paraphrased{suffix}"
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
                        help="Evaluate only the first N QA records, before paraphrase expansion")
    parser.add_argument("--question-variant", choices=("canonical", "paraphrased"),
                        default="canonical",
                        help="Use Question or released Question-Paraphrased variants")
    parser.add_argument("--paraphrase-index", default="all",
                        help="Evaluate all paraphrases (default) or one zero-based index")
    parser.add_argument("--entity-selector", choices=("uniform", "first"),
                        default="uniform",
                        help="Gold-blind single-answer choice among reachable entities")
    parser.add_argument("--seed", type=int, default=42,
                        help="Stable per-question uniform tie-breaking seed")
    parser.add_argument("--glove", type=Path, default=None,
                        help="Required only for unseen paraphrase words with local GloVe")
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
