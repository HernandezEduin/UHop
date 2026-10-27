#!/usr/bin/env python3
"""Adapt PathQuestion-Family PQ/PQL mixed-hop reconstructions for UHop.

UHop supervises one ordered relation chain per example, never entity-level
actions. Alternative released relation chains are kept as evaluation references.
The selected active graph may be filtered or source, but its released QA
answers and family-controlled splits remain unchanged.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    from .prepare_theseus import (
        SPLITS, candidates_and_frontier, make_embeddings, parse_answers,
        parse_literal, read_graph,
    )
except ImportError:
    from prepare_theseus import (
        SPLITS, candidates_and_frontier, make_embeddings, parse_answers,
        parse_literal, read_graph,
    )


def parse_relation_chains(value, hops):
    """Read the multiple released relation keys without collapsing alternatives."""
    keys = parse_literal(value)
    if not isinstance(keys, list) or not keys:
        raise ValueError("Expected nonempty Multi-Paths-Key list")
    chains = []
    for key in keys:
        chain = [part.strip() for part in str(key).split("->")]
        if len(chain) != hops or not all(chain):
            raise ValueError(f"Invalid {hops}-hop relation key: {key!r}")
        if chain not in chains:
            chains.append(chain)
    return chains


def optional_answers(value):
    """Graph-derived answer sets can be empty even when released answers exist."""
    parsed = parse_literal(value)
    if parsed is None or parsed == []:
        return []
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]
    return list(dict.fromkeys(str(answer) for answer in parsed))


def resolve_artifacts(root, dataset, regime):
    """Require named graph regimes to prevent mixing filtered and source files."""
    name = dataset.upper()
    kg_relative = Path("link_prediction") / name / regime / "triplets.txt"
    qa_relative = Path("qa") / name / regime / f"{dataset}_qa_nhop.csv"
    for path in (kg_relative, qa_relative):
        if not (root / path).is_file():
            raise FileNotFoundError(
                f"Missing {root / path}. Reconstruct with "
                "scripts/reconstruct.py --dataset all --graph both --seed 42 --encode-ids"
            )
    return kg_relative, qa_relative


def relation_labels(root, dataset, regime, relations):
    """Use the released relation titles as the GloVe-facing relation text."""
    names = {relation: relation for relation in relations}
    mapping = root / "link_prediction" / dataset.upper() / regime / "relation_data.csv"
    if mapping.is_file():
        with open(mapping, encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                relation = (row.get("RID") or "").strip()
                title = (row.get("Title") or "").strip()
                if relation in names and title:
                    names[relation] = (
                        re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
                        or relation
                    )
    return names


def prepare(args):
    root = args.reconstruction_root.resolve()
    dataset = args.dataset.lower()
    regime = args.graph_regime
    tag = f"{dataset}_{regime}"
    kg_relative, qa_relative = resolve_artifacts(root, dataset, regime)
    graph, relations, triples = read_graph(root / kg_relative)

    repo_root = Path(__file__).resolve().parents[2]
    output = (args.output_root or repo_root / "data" / "THESEUS") / tag
    output.mkdir(parents=True, exist_ok=True)
    texts = relation_labels(root, dataset, regime, relations)
    rela2id = {"PADDING": 0, "<unk>": 1}
    rela2id.update({relation: index for index, relation in enumerate(relations, 2)})
    (output / "rela2id.json").write_text(
        json.dumps(rela2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "relation_text.json").write_text(
        json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    examples = defaultdict(list)
    metadata = defaultdict(list)
    questions = []
    released_unreachable = 0
    total_questions = 0
    alternate_chains = 0

    with open(root / qa_relative, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split_name = (row.get("SplitLabel") or "").strip().lower()
            if split_name not in SPLITS:
                raise ValueError(f"Unexpected SplitLabel: {split_name!r}")
            split = SPLITS[split_name]
            number = str(row["Question-Number"]).strip()
            question = (row.get("Question") or "").strip()
            source = (row.get("Source-Entity") or "").strip()
            if not question or not source or not number:
                raise ValueError("QA row lacks question text, ID, or source entity")
            hops = int(row["Hops"])
            chains = parse_relation_chains(row.get("Multi-Paths-Key"), hops)
            answers = parse_answers(row.get("Answer-Entity"))
            graph_answers = optional_answers(row.get("Graph-Answer-Entity"))
            reference_paths = parse_literal(row.get("Multi-Paths")) or []
            family_id = (row.get("Question-Family-ID") or "").strip()
            if not family_id:
                raise ValueError(f"QA {number} lacks Question-Family-ID")

            supervision = []
            reachable = False
            for chain_index, chain in enumerate(chains):
                try:
                    steps, terminal = candidates_and_frontier(graph, source, chain)
                except ValueError as exc:
                    raise ValueError(
                        f"QA {number}: released chain {chain_index} is not "
                        f"traversable on the {regime} graph"
                    ) from exc
                reachable |= bool(set(terminal) & set(answers))
                if args.supervision_chains == "all" or chain_index == 0:
                    supervision.append([f"{number}:{chain_index}", question, steps])

            examples[split].extend(supervision)
            metadata[split].append({
                "question_number": number,
                "question": question,
                "source": source,
                "answers": answers,
                "graph_answers": graph_answers,
                "hops": hops,
                "gold_relations": chains[0],
                "gold_relation_chains": chains,
                "reference_paths": reference_paths,
                "question_family_id": family_id,
            })
            questions.append(question)
            released_unreachable += int(not reachable)
            alternate_chains += int(len(chains) > 1)
            total_questions += 1

    for split in ("train", "valid", "test"):
        if not examples[split] or not metadata[split]:
            raise ValueError(f"Empty {split} split in {root / qa_relative}")
        with open(output / f"{split}_data.txt", "w", encoding="utf-8") as stream:
            for example in examples[split]:
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
        with open(output / f"{split}_metadata.jsonl", "w", encoding="utf-8") as stream:
            for record in metadata[split]:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    glove_source = args.glove_source
    if glove_source == "auto":
        glove_source = "local" if args.glove else "huggingface"
    if glove_source == "huggingface" and args.glove:
        raise ValueError("Use --glove-source local with --glove")
    make_embeddings(
        output, questions, texts, args.glove, args.emb_size,
        glove_source=glove_source, glove_model=args.glove_model,
        glove_revision=args.glove_revision,
    )
    (output / "source.json").write_text(
        json.dumps({
            "dataset": tag,
            "pathquestion_dataset": dataset,
            "graph_regime": regime,
            "reconstruction_root": str(root),
            "local_kg_file": str(kg_relative),
            "local_qa_file": str(qa_relative),
            "answer_reference": "released",
            "graph_direction": "forward_only",
            "triple_count": len(triples),
            "question_counts": {s: len(metadata[s]) for s in ("train", "valid", "test")},
            "supervision_example_counts": {s: len(examples[s]) for s in ("train", "valid", "test")},
            "supervision_chains": args.supervision_chains,
            "multi_chain_question_count": alternate_chains,
            "released_unreachable_question_count": released_unreachable,
            "glove_source": glove_source,
            "glove_model": args.glove_model if glove_source == "huggingface" else None,
            "glove_revision": args.glove_revision if glove_source == "huggingface" else None,
            "glove_dimensions": args.emb_size,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {tag} at {output}")
    print({s: len(metadata[s]) for s in ("train", "valid", "test")})
    print(f"Alternative-chain questions: {alternate_chains}/{total_questions}")
    if released_unreachable:
        print(
            f"Warning: {released_unreachable} questions have no released answer "
            f"reachable through an annotated chain on the active {regime} graph."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("pq", "pql"), required=True)
    parser.add_argument("--graph-regime", choices=("filtered", "source"), default="filtered")
    parser.add_argument("--reconstruction-root", type=Path, required=True,
                        help="PathQuestion-Family generated/ directory from --graph both --encode-ids")
    parser.add_argument("--supervision-chains", choices=("first", "all"), default="first",
                        help="UHop expects a single positive relation chain per example. "
                             "Choose the first released chain, or expand all alternatives.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--glove", type=Path, default=None)
    parser.add_argument("--glove-source", choices=("auto", "huggingface", "local"),
                        default="auto")
    parser.add_argument("--glove-model", default="NeuML/glove-6B")
    parser.add_argument("--glove-revision", default="main")
    parser.add_argument("--emb-size", type=int, default=300, choices=(50, 300))
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
