#!/usr/bin/env python3
"""Greedy, free-running relation navigation for THESEUS datasets using trained UHop.

This is an inference-only wrapper. UHop itself predicts relations, not unique
entity-edge actions. Answer-set reachability is therefore not Hits@1.
"""
import argparse
from collections import Counter, defaultdict
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


def score_actions(model, question, actions, word2id, rela2id,
                  relation_text, args, batch_size):
    """Score (relation, history) pairs with UHop's original shared padding.

    In the original Framework._single_UHop_step, candidates at different depths
    are packed into a single forward pass. In particular, termination compares
    the previous chosen action (with its earlier history) and possible next
    actions (with the updated history). Cached scores from separate passes are
    not interchangeable because the LSTM consumes padded relation sequences.

    Fixed global padding widths across chunks keep scores consistent even when
    a large frontier requires multiple model calls.
    """
    if not actions:
        return []
    device = args.device
    question_ids = [word2id.get(w, word2id["<unk>"])
                    for w in question.strip().split(" ")]
    relation_ids = []
    relation_words = []
    histories_ids = []
    histories_words = []
    for relation, history in actions:
        ids = tokenize(relation, rela2id)
        words = tokenize(relation_text.get(relation, relation), word2id, "._")
        prev_ids = [tokenize(prev, rela2id) for prev in history]
        prev_words = [
            tokenize(relation_text.get(prev, prev), word2id, "._")
            for prev in history
        ]
        if args.dynamic == "none":
            ids = sum(prev_ids + [ids], [])
            words = sum(prev_words + [words], [])
        relation_ids.append(ids)
        relation_words.append(words)
        histories_ids.append(prev_ids)
        histories_words.append(prev_words)

    max_rel_id = max(len(ids) for ids in relation_ids)
    max_rel_word = max(len(words) for words in relation_words)
    depth = max(len(history) for history in histories_ids)
    if args.dynamic == "flatten":
        flattened_ids = [sum(history, []) for history in histories_ids]
        flattened_words = [sum(history, []) for history in histories_words]
        prev_id_widths = [max(map(len, flattened_ids))]
        prev_word_widths = [max(map(len, flattened_words))]
    elif args.dynamic == "recurrent":
        prev_id_widths = [
            max(len(history[step]) if step < len(history) else 0
                for history in histories_ids)
            for step in range(depth)
        ]
        prev_word_widths = [
            max(len(history[step]) if step < len(history) else 0
                for history in histories_words)
            for step in range(depth)
        ]
    elif args.dynamic != "none":
        raise ValueError(f"Unsupported UHop history mode: {args.dynamic}")

    all_scores = []
    for start in range(0, len(actions), batch_size):
        stop = min(start + batch_size, len(actions))
        width = stop - start
        question_batch = pad(
            [question_ids] * width, max(5, len(question_ids)),
            word2id["PADDING"], device
        )
        rela_batch = pad(
            relation_ids[start:stop], max_rel_id, rela2id["PADDING"], device
        )
        words_batch = pad(
            relation_words[start:stop], max_rel_word, word2id["PADDING"], device
        )
        prev_batches, prev_word_batches = [], []
        if args.dynamic == "flatten":
            if prev_id_widths[0]:
                prev_batches = [pad(
                    flattened_ids[start:stop], prev_id_widths[0],
                    rela2id["PADDING"], device
                )]
            if prev_word_widths[0]:
                prev_word_batches = [pad(
                    flattened_words[start:stop], prev_word_widths[0],
                    word2id["PADDING"], device
                )]
        elif args.dynamic == "recurrent":
            for step in range(depth):
                prev_batches.append(pad(
                    [history[step] if step < len(history) else []
                     for history in histories_ids[start:stop]],
                    prev_id_widths[step], rela2id["PADDING"], device
                ))
                prev_word_batches.append(pad(
                    [history[step] if step < len(history) else []
                     for history in histories_words[start:stop]],
                    prev_word_widths[step], word2id["PADDING"], device
                ))
        scores = model(
            question_batch, words_batch, rela_batch,
            prev_word_batches, prev_batches
        )
        all_scores.extend(scores.reshape(-1).detach().cpu().tolist())
    return all_scores


def score_candidates(model, question, relations, history, word2id, rela2id,
                     relation_text, args, batch_size):
    """Rank outgoing relations using the same scoring inputs as UHop's RC."""
    scores = score_actions(
        model, question, [(relation, history) for relation in relations],
        word2id, rela2id, relation_text, args, batch_size
    )
    return dict(zip(relations, scores))


def navigate(model, question, source, graph, word2id, rela2id,
             relation_text, model_args, eval_args):
    frontier = {source}
    history = []
    stop_reason = "max_hops"
    for _ in range(eval_args.max_hops):
        choices = sorted({
            rel for node in frontier for rel in graph.get(node, {})
        })
        if not choices:
            stop_reason = "no_outgoing_relations"
            break

        # TD comes before the next RC decision. Re-score the previous action
        # with history[:-1] and all next actions with history in the SAME
        # padded candidate set, as Framework._termination_decision does.
        # No gold relations or answer entities enter this comparison.
        if history and eval_args.termination_mode == "learned":
            actions = [(history[-1], history[:-1])]
            actions.extend((relation, history) for relation in choices)
            td_scores = score_actions(
                model, question, actions, word2id, rela2id,
                relation_text, model_args, eval_args.batch_size
            )
            if max(td_scores[1:]) <= td_scores[0]:
                stop_reason = "learned_termination"
                break

        # RC is evaluated separately after TD, matching UHop's training order.
        scores = score_candidates(
            model, question, choices, history,
            word2id, rela2id, relation_text, model_args, eval_args.batch_size
        )
        best_relation = max(choices, key=lambda relation: scores[relation])
        history.append(best_relation)
        frontier = {
            tail for node in frontier
            for tail in graph.get(node, {}).get(best_relation, ())
        }
    return history, sorted(frontier), stop_reason


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


def score_reference_chains(predicted, gold_chains):
    """Score against any valid released relation chain, without gold at inference."""
    if not gold_chains:
        raise ValueError("No relation reference chains")
    closest = min(
        gold_chains,
        key=lambda chain: (
            relation_edit_distance(predicted, chain),
            -relation_f1(predicted, chain),
        ),
    )
    return {
        "matched_gold_relations": closest,
        "relation_chain_correct": predicted in gold_chains,
        "RED": relation_edit_distance(predicted, closest),
        "F1_REL": max(relation_f1(predicted, chain) for chain in gold_chains),
    }


def family_macro_summary(results):
    """Average QA metrics within a family, then weight all families equally."""
    metrics = (
        "hits_at_1", "expected_hits_at_1_uniform", "relation_chain_correct",
        "RED", "F1_REL", "answer_reachable", "answer_set_exact",
        "terminal_frontier_size",
    )
    public_names = {
        "relation_chain_correct": "relation_chain_accuracy",
        "answer_reachable": "answer_reachability",
        "answer_set_exact": "answer_set_exact_match",
        "terminal_frontier_size": "mean_terminal_frontier_size",
    }

    def summarize(records):
        families = defaultdict(list)
        for item in records:
            if not item.get("question_family_id"):
                raise ValueError("Missing family ID in reconstructed PQ/PQL metadata")
            families[item["question_family_id"]].append(item)
        output = {
            "family_count": len(families),
            "question_count": len({str(item["question_number"]) for item in records}),
            "count": len(records),
        }
        for metric in metrics:
            output[public_names.get(metric, metric)] = (
                sum(
                    sum(item[metric] for item in group) / len(group)
                    for group in families.values()
                ) / len(families)
            )
        return output

    by_hop = defaultdict(list)
    for record in results:
        by_hop[str(record["hops"])].append(record)
    return {
        "all": summarize(results),
        "by_hop": {
            hops: summarize(group) for hops, group in sorted(by_hop.items())
        },
    }


def evaluate(args):
    checkpoint = Path(args.checkpoint).resolve()
    model_args = json.loads((checkpoint / "args.txt").read_text(encoding="utf-8"))
    dataset = model_args["dataset"].lower()
    if dataset not in (
        "kinship", "mquake_st_single", "mquake_st_multi",
        "pq_filtered", "pq_source", "pql_filtered", "pql_source",
    ):
        raise ValueError("Unsupported UHop navigation dataset")
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
    if source.get("local_kg_file"):
        reconstruction_root = args.offline_source or Path(source["reconstruction_root"])
        graph_path = reconstruction_root / source["local_kg_file"]
        if not graph_path.is_file():
            raise FileNotFoundError(
                f"Missing reconstructed KG: {graph_path}. "
                "Pass --offline-source /path/to/PathQuestion-Family/generated"
            )
    else:
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
        if source.get("local_qa_file"):
            reconstruction_root = args.offline_source or Path(source["reconstruction_root"])
            qa_path = reconstruction_root / source["local_qa_file"]
            if not qa_path.is_file():
                raise FileNotFoundError(f"Missing reconstructed QA: {qa_path}")
        else:
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
            predicted, terminal, stop_reason = navigate(
                model, question, row["source"], graph, word2id, rela2id,
                relation_text, original, args
            )
            if args.answer_reference == "graph":
                if "graph_answers" not in row:
                    raise ValueError("Graph-answer reference is unavailable for this dataset")
                gold = set(row["graph_answers"])
            else:
                gold = set(row["answers"])
            gold_chains = row.get("gold_relation_chains") or [row["gold_relations"]]
            relation_metrics = score_reference_chains(predicted, gold_chains)
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
                "stop_reason": stop_reason,
                "predicted_hops": len(predicted),
                "gold_relations": row["gold_relations"],
                "gold_relation_chains": gold_chains,
                "question_family_id": row.get("question_family_id"),
                **relation_metrics,
                "selected_answer": selected,
                "hits_at_1": int(selected in gold),
                "expected_hits_at_1_uniform": expected_uniform_hits_at_1(
                    terminal, gold
                ),
                "answer_reachable": bool(gold & found),
                "answer_set_exact": gold == found,
                "terminal_frontier_size": len(terminal),
                "terminal_entities": terminal,
                "gold_answers": sorted(gold),
                "answer_reference": args.answer_reference,
            })
            if index % 100 == 0:
                print(f"Evaluated {index}/{len(variants)}", flush=True)
    summary = aggregate(
        results, args.question_variant, args.entity_selector, args.seed
    )
    summary["termination_mode"] = args.termination_mode
    summary["answer_reference"] = args.answer_reference
    if source.get("graph_regime"):
        summary["graph_regime"] = source["graph_regime"]
    if any(row.get("question_family_id") for row in results):
        summary["family_macro"] = family_macro_summary(results)
        summary["metric_warning"] += (
            " PQ/PQL additionally reports family_macro, which weights each "
            "structural question family equally. Relation fidelity uses the "
            "closest released relation chain when alternatives are supplied."
        )
    for group, records in [
        (summary["all"], results),
        *[(summary["by_hop"][str(hops)],
           [row for row in results if row["hops"] == hops])
          for hops in sorted({row["hops"] for row in results})],
    ]:
        group["stop_reasons"] = dict(sorted(Counter(
            row["stop_reason"] for row in records
        ).items()))
        group["premature_stop_reasons"] = dict(sorted(Counter(
            row["stop_reason"] for row in records
            if row["predicted_hops"] < row["hops"]
        ).items()))
    if args.termination_mode == "ignore":
        summary["metric_warning"] += (
            " Termination was DISABLED in this diagnostic run. "
            "Its answer metrics are not a valid UHop benchmark."
        )
    if args.output:
        output = args.output
    else:
        suffix = "" if args.question_variant == "canonical" else "_paraphrased"
        if args.question_variant == "paraphrased" and args.paraphrase_index != "all":
            suffix += f"_{args.paraphrase_index}"
        if args.termination_mode == "ignore":
            suffix += "_no_td"
        output = checkpoint / f"{args.split}_navigation{suffix}"
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
    parser.add_argument("--answer-reference", choices=("released", "graph"),
                        default="released",
                        help="Evaluate released answers or active-KG graph-derived answers (PQ/PQL)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument(
        "--termination-mode", choices=("learned", "ignore"),
        default="learned",
        help="Original UHop joint TD by default. 'ignore' disables TD for "
             "diagnosis only and is not a valid UHop benchmark."
    )
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
