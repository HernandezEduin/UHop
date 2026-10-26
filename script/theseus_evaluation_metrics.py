"""Gold-blind entity tie-breaking and THESEUS relation-fidelity diagnostics.

UHop scores relations but does not rank the entities in the reached frontier.
Selecting one entity is a separately specified inference convention.
"""
import ast
import csv
import hashlib
import json
import random
from collections import defaultdict


def load_paraphrases(stream, metadata_rows):
    """Join released paraphrase lists to prepared QA by stable question ID."""
    expected = {str(row["question_number"]): row["question"] for row in metadata_rows}
    if len(expected) != len(metadata_rows):
        raise ValueError("Duplicate question numbers in prepared QA metadata")
    collected = {}
    for record in csv.DictReader(stream):
        key = (record.get("Question-Number") or "").strip()
        if key not in expected:
            continue
        if key in collected:
            raise ValueError(f"Duplicate question number in source QA: {key}")
        if (record.get("Question") or "").strip() != expected[key]:
            raise ValueError(f"Question text mismatch for QA {key}")
        raw = (record.get("Question-Paraphrased") or "").strip()
        try:
            paraphrases = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Invalid Question-Paraphrased for QA {key}") from exc
        if not isinstance(paraphrases, list) or not paraphrases or any(
            not isinstance(text, str) or not text.strip() for text in paraphrases
        ):
            raise ValueError(f"Expected a nonempty list of paraphrases for QA {key}")
        collected[key] = [text.strip() for text in paraphrases]
    missing = sorted(set(expected) - set(collected))
    if missing:
        raise ValueError(f"Missing paraphrases for {len(missing)} QA records: {missing[:5]}")
    return collected


def question_variants(rows, variant, paraphrases=None, paraphrase_index="all"):
    """Return (metadata, question text, index) without mutating the reference QA."""
    if variant == "canonical":
        if paraphrase_index != "all":
            raise ValueError("--paraphrase-index requires --question-variant paraphrased")
        return [(row, row["question"], None) for row in rows]
    if variant != "paraphrased" or paraphrases is None:
        raise ValueError("Paraphrased evaluation requires released paraphrases")
    if paraphrase_index != "all":
        try:
            selected_index = int(paraphrase_index)
        except ValueError as exc:
            raise ValueError("--paraphrase-index must be 'all' or a nonnegative integer") from exc
        if selected_index < 0:
            raise ValueError("--paraphrase-index must be nonnegative")
    else:
        selected_index = None
    out = []
    for row in rows:
        options = paraphrases[str(row["question_number"])]
        indices = range(len(options)) if selected_index is None else [selected_index]
        for index in indices:
            if index >= len(options):
                raise ValueError(
                    f"Question {row['question_number']} has only {len(options)} paraphrases"
                )
            out.append((row, options[index], index))
    return out


def select_entity(terminal, dataset, question_number, variant_index,
                  selector="uniform", seed=42):
    """Select exactly one answer without access to gold answer labels."""
    if not terminal:
        return None
    candidates = sorted(terminal)
    if selector == "first":
        return candidates[0]
    if selector != "uniform":
        raise ValueError(f"Unknown entity selector: {selector}")
    key = json.dumps(
        [seed, dataset, str(question_number), variant_index],
        ensure_ascii=False
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return candidates[rng.randrange(len(candidates))]


def expected_uniform_hits_at_1(terminal, answers):
    """Expected Hits@1 under a uniform gold-blind choice from the frontier."""
    return len(set(terminal) & set(answers)) / len(terminal) if terminal else 0.0


def relation_edit_distance(predicted, reference):
    """Levenshtein distance between ordered relation labels."""
    previous = list(range(len(reference) + 1))
    for i, relation in enumerate(predicted, 1):
        current = [i]
        for j, gold in enumerate(reference, 1):
            current.append(min(
                current[j - 1] + 1,
                previous[j] + 1,
                previous[j - 1] + (relation != gold),
            ))
        previous = current
    return previous[-1]


def relation_f1(predicted, reference):
    """THESEUS F1_REL uses distinct relation types, not occurrences."""
    a, b = set(predicted), set(reference)
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0


def aggregate(results, question_variant, entity_selector, seed):
    def one_group(rows):
        count = len(rows)
        return {
            "count": count,
            "question_count": len({str(r["question_number"]) for r in rows}),
            "hits_at_1": sum(r["hits_at_1"] for r in rows) / count,
            "expected_hits_at_1_uniform": sum(
                r["expected_hits_at_1_uniform"] for r in rows
            ) / count,
            "relation_chain_accuracy": sum(
                r["relation_chain_correct"] for r in rows
            ) / count,
            "RED": sum(r["RED"] for r in rows) / count,
            "F1_REL": sum(r["F1_REL"] for r in rows) / count,
            "answer_reachability": sum(
                r["answer_reachable"] for r in rows
            ) / count,
            "answer_set_exact_match": sum(
                r["answer_set_exact"] for r in rows
            ) / count,
            "mean_terminal_frontier_size": sum(
                r["terminal_frontier_size"] for r in rows
            ) / count,
        }
    by_hop = defaultdict(list)
    for result in results:
        by_hop[str(result["hops"])].append(result)
    return {
        "question_variant": question_variant,
        "entity_selector": entity_selector,
        "seed": seed,
        "all": one_group(results),
        "by_hop": {hop: one_group(rows) for hop, rows in sorted(by_hop.items())},
        "metric_warning": (
            "UHop does not score terminal entities. hits_at_1 selects a single "
            "entity from its terminal frontier using the stated gold-blind "
            "tie-breaker, and is not a learned entity-ranking score. "
            "expected_hits_at_1_uniform averages the exact uniform-choice "
            "success probability. answer_reachability is an oracle upper bound, "
            "not Hits@1. RED and F1_REL measure relation-level fidelity only."
        ),
    }
