#!/usr/bin/env python3
"""Convert THESEUS Kinship/MQuAKE-ST releases to UHop's original relation-step format.

The generated labels supervise UHop's original relation choice (RC) and
termination decision (TD). They do not label individual entity-edge actions.
"""
import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASETS = {
    "kinship": ("HalcyonSolutions/Kinship", "kg/orig/triplets.txt",
                "qa/kinship_qa_nhop.csv"),
    "mquake_st_single": ("HalcyonSolutions/MQuAKE-ST", "kg/triplets.txt",
                         "qa/single_answers/qa_nhop.csv"),
    "mquake_st_multi": ("HalcyonSolutions/MQuAKE-ST", "kg/triplets.txt",
                        "qa/multi_answers/qa_nhop.csv"),
}
SPLITS = {"train": "train", "dev": "valid", "valid": "valid",
          "validation": "valid", "test": "test"}


def read_release_file(dataset, filename, offline_source, revision):
    if offline_source:
        path = Path(offline_source) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing source dataset file: {path}")
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Install the adapter dependencies: pip install -r requirements-theseus.txt"
        ) from exc
    repo_id = DATASETS[dataset][0]
    return Path(hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision
    ))


def read_graph(path):
    graph = defaultdict(lambda: defaultdict(set))
    relations = set()
    triples = set()
    with open(path, encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) != 3:
                raise ValueError(f"Malformed triple at {path}:{lineno}")
            head, relation, tail = parts
            graph[head][relation].add(tail)
            relations.add(relation)
            triples.add((head, relation, tail))
    if not triples:
        raise ValueError(f"Graph is empty: {path}")
    return graph, sorted(relations), triples


def parse_literal(value):
    if value is None or not str(value).strip() or str(value).strip().lower() == "nan":
        return None
    try:
        return ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return str(value).strip()


def parse_answers(value):
    parsed = parse_literal(value)
    if parsed is None:
        raise ValueError("QA has no Answer-Entity")
    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]
    answers = list(dict.fromkeys(str(answer) for answer in parsed))
    if not answers:
        raise ValueError("QA has empty Answer-Entity")
    return answers


def parse_paths(value):
    parsed = parse_literal(value)
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        raise ValueError("Paths must contain an ordered list of triples")
    if not parsed:
        return []
    if len(parsed[0]) == 3 and all(isinstance(v, str) for v in parsed[0]):
        return [parsed]
    if all(isinstance(path, list) for path in parsed):
        return parsed
    raise ValueError("Unrecognized reference-path structure")


def gold_chain(row, paths):
    chain_str = (row.get("Path-Key") or "").strip()
    if chain_str:
        chain = chain_str.split("->")
    elif paths:
        chain = [str(triple[1]) for triple in paths[0]]
    else:
        raise ValueError("Neither Path-Key nor Paths contains a gold relation chain")
    if not chain or any(not relation for relation in chain):
        raise ValueError(f"Invalid Path-Key: {chain_str!r}")
    for path in paths:
        if [str(triple[1]) for triple in path] != chain:
            raise ValueError("Distinct gold relation chains require separate UHop examples")
    return chain


def candidates_and_frontier(graph, source, chain):
    frontier = {source}
    previous = []
    steps = []
    for relation in chain:
        candidate_relations = sorted({
            candidate for node in frontier for candidate in graph.get(node, {})
        })
        if relation not in candidate_relations:
            raise ValueError(
                f"Gold relation {relation!r} is not reachable from frontier {frontier}"
            )
        steps.append([
            [candidate, list(previous), int(candidate == relation)]
            for candidate in candidate_relations
        ])
        frontier = {
            tail for node in frontier for tail in graph.get(node, {}).get(relation, ())
        }
        previous.append(relation)
    # UHop's last step contains only negative relations for the stop decision.
    last_relations = sorted({
        candidate for node in frontier for candidate in graph.get(node, {})
    })
    steps.append([[candidate, list(previous), 0] for candidate in last_relations])
    return steps, frontier


def read_relation_text(dataset, offline_source, revision, relations):
    texts = {relation: relation for relation in relations}
    if dataset == "kinship":
        return texts
    path = read_release_file(dataset, "metadata/relation_data.csv", offline_source, revision)
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            relation = (row.get("Property") or "").strip()
            title = (row.get("Title") or "").strip()
            if relation in texts and title:
                # Existing UHop text channel tokenizes relation labels at "_".
                text = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
                texts[relation] = text or relation
    return texts


def make_embeddings(directory, questions, relation_text, glove, dim,
                    glove_source="local", glove_model="NeuML/glove-6B",
                    glove_revision="main"):
    words = set()
    for question in questions:
        words.update(question.strip().split(" "))
    for title in relation_text.values():
        words.update(title.replace("_", ".").split("."))
    words.discard("PADDING")
    words.discard("<unk>")
    word2id = {"PADDING": 0, "<unk>": 1}
    word2id.update({word: index for index, word in enumerate(sorted(words), 2)})
    weights = np.zeros((len(word2id), dim), dtype=np.float32)

    if glove_source == "local":
        if glove is None:
            raise ValueError("--glove is required with --glove-source local")
        found = 0
        with open(glove, encoding="utf-8") as stream:
            for line in stream:
                fields = line.rstrip().split(" ")
                if len(fields) != dim + 1:
                    continue
                index = word2id.get(fields[0])
                if index is not None and index > 1:
                    weights[index] = np.asarray(fields[1:], dtype=np.float32)
                    found += 1
    elif glove_source == "huggingface":
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise RuntimeError(
                "Install the adapter dependencies: pip install -r requirements-theseus.txt"
            ) from exc

        model_path = hf_hub_download(
            repo_id=glove_model, filename="model.safetensors",
            revision=glove_revision
        )
        vocab_path = hf_hub_download(
            repo_id=glove_model, filename="vocab.json",
            revision=glove_revision
        )
        with open(vocab_path, encoding="utf-8") as stream:
            tokens = json.load(stream)["tokens"]
        # Safetensors loads through mmap, so the full 300d vectors need not be
        # exported to a temporary text file before selecting dataset tokens.
        vectors = load_file(model_path)["vectors"]
        if vectors.ndim != 2 or vectors.shape[1] != dim:
            raise ValueError(
                f"{glove_model} has vector dimension {vectors.shape}, "
                f"but --emb-size is {dim}"
            )
        found = 0
        for word, index in word2id.items():
            if index < 2:
                continue
            source_index = tokens.get(word)
            if source_index is None:
                continue
            if not isinstance(source_index, int) or not 0 <= source_index < len(vectors):
                raise ValueError(f"Invalid vocabulary index for {word!r}")
            weights[index] = vectors[source_index]
            found += 1
    else:
        raise ValueError(f"Unknown GloVe source: {glove_source}")

    if not found:
        raise ValueError(
            f"No GloVe vocabulary tokens matched ({glove_source}, {dim}d)."
        )
    (directory / "word2id.json").write_text(
        json.dumps(word2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(directory / "word_emb.npy", weights)
    print(f"GloVe coverage: {found}/{len(word2id) - 2} tokens ({dim}d)")

def prepare(args):
    dataset = args.dataset
    _, kg_file, qa_file = DATASETS[dataset]
    root = Path(__file__).resolve().parents[2]
    destination = (args.output_root or root / "data" / "THESEUS") / dataset
    destination.mkdir(parents=True, exist_ok=True)
    graph, relations, triples = read_graph(
        read_release_file(dataset, kg_file, args.offline_source, args.revision)
    )
    texts = read_relation_text(dataset, args.offline_source, args.revision, relations)
    rela2id = {"PADDING": 0, "<unk>": 1}
    rela2id.update({relation: index for index, relation in enumerate(relations, 2)})
    (destination / "rela2id.json").write_text(
        json.dumps(rela2id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "relation_text.json").write_text(
        json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    split_examples = defaultdict(list)
    split_metadata = defaultdict(list)
    questions = []
    qa_path = read_release_file(dataset, qa_file, args.offline_source, args.revision)
    with open(qa_path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split_label = (row.get("SplitLabel") or "").strip().lower()
            if split_label not in SPLITS:
                raise ValueError(f"Unrecognized SplitLabel: {split_label!r}")
            split = SPLITS[split_label]
            source = (row.get("Source-Entity") or "").strip()
            paths = parse_paths(row.get("Paths"))
            chain = gold_chain(row, paths)
            answers = parse_answers(row.get("Answer-Entity"))
            hops = int(row["Hops"])
            if len(chain) != hops:
                raise ValueError("Path-Key and Hops disagree")
            for path in paths:
                if not path or str(path[0][0]) != source or len(path) != hops:
                    raise ValueError("Reference path is noncontiguous or has wrong source")
                for i, triple in enumerate(path):
                    triple = tuple(map(str, triple))
                    if triple not in triples:
                        raise ValueError(f"Reference triple missing from KG: {triple}")
                    if i and str(path[i - 1][2]) != triple[0]:
                        raise ValueError("Reference path has disconnected edges")
            steps, terminal = candidates_and_frontier(graph, source, chain)
            if not any(answer in terminal for answer in answers):
                raise ValueError(
                    f"No answer is reachable for question {row['Question-Number']}"
                )
            question = (row.get("Question") or "").strip()
            if not question:
                raise ValueError("Empty Question")
            number = row["Question-Number"]
            split_examples[split].append([number, question, steps])
            split_metadata[split].append({
                "question_number": number,
                "question": question,
                "source": source,
                "answers": answers,
                "hops": hops,
                "gold_relations": chain,
                "reference_paths": paths,
            })
            questions.append(question)

    for split in ("train", "valid", "test"):
        if not split_examples[split]:
            raise ValueError(f"No questions in {split} split")
        with open(destination / f"{split}_data.txt", "w", encoding="utf-8") as stream:
            for example in split_examples[split]:
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
        with open(destination / f"{split}_metadata.jsonl", "w", encoding="utf-8") as stream:
            for metadata in split_metadata[split]:
                stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    glove_source = getattr(args, "glove_source", "auto")
    glove_path = getattr(args, "glove", None)
    if glove_source == "auto":
        glove_source = "local" if glove_path else "huggingface"
    if glove_source == "huggingface" and glove_path:
        raise ValueError("Use --glove-source local with --glove")
    glove_model = getattr(args, "glove_model", "NeuML/glove-6B")
    glove_revision = getattr(args, "glove_revision", "main")
    make_embeddings(
        destination, questions, texts, glove_path, args.emb_size,
        glove_source=glove_source, glove_model=glove_model,
        glove_revision=glove_revision
    )
    (destination / "source.json").write_text(
        json.dumps({
            "dataset": dataset,
            "huggingface_repo": DATASETS[dataset][0],
            "revision": args.revision,
            "kg_file": kg_file,
            "qa_file": qa_file,
            "graph_direction": "forward_only",
            "triple_count": len(triples),
            "question_counts": {s: len(split_examples[s]) for s in ("train", "valid", "test")},
            "format": "UHop relation choices, not entity-edge labels",
            "glove_source": glove_source,
            "glove_model": glove_model if glove_source == "huggingface" else None,
            "glove_revision": glove_revision if glove_source == "huggingface" else None,
            "glove_dimensions": args.emb_size,
        }, indent=2), encoding="utf-8"
    )
    print(f"Prepared {dataset} at {destination}")
    print({s: len(split_examples[s]) for s in ("train", "valid", "test")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--glove", type=Path, default=None,
                        help="Optional local Stanford GloVe .txt, used as offline fallback")
    parser.add_argument("--glove-source", choices=("auto", "huggingface", "local"),
                        default="auto",
                        help="Default auto uses Hugging Face, or local when --glove is supplied")
    parser.add_argument("--glove-model", default="NeuML/glove-6B",
                        help="Hugging Face model with vocab.json and model.safetensors")
    parser.add_argument("--glove-revision", default="main",
                        help="Pin the GloVe model revision independently of the dataset")
    parser.add_argument("--emb-size", type=int, default=300, choices=(50, 300))
    parser.add_argument("--revision", default="main",
                        help="Hugging Face revision. Pin a commit for reproducibility.")
    parser.add_argument("--offline-source", type=Path, default=None,
                        help="Pre-downloaded dataset root, instead of Hugging Face")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Base output directory, default data/THESEUS")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
