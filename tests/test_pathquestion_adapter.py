"""Fixture-level checks for the reconstructed PQ/PQL UHop adapter."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))

from script.preprocess.prepare_pathquestion import (
    parse_relation_chains, prepare, resolve_artifacts,
)
from script.evaluate_theseus import family_macro_summary, score_reference_chains


class PathQuestionAdapterTests(unittest.TestCase):
    def test_multiple_gold_chains_are_preserved(self):
        self.assertEqual(
            parse_relation_chains("['r->t', 's->u']", 2),
            [["r", "t"], ["s", "u"]],
        )
        with self.assertRaises(ValueError):
            parse_relation_chains("['r->t']", 3)

    def test_scores_any_valid_relation_chain(self):
        metrics = score_reference_chains(
            ["s", "u"], [["r", "t"], ["s", "u"]]
        )
        self.assertEqual(metrics["RED"], 0)
        self.assertEqual(metrics["F1_REL"], 1)
        self.assertTrue(metrics["relation_chain_correct"])
        self.assertEqual(metrics["matched_gold_relations"], ["s", "u"])

    def test_preparation_uses_family_splits_and_released_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            reconstructed = temp / "generated"
            graph_dir = reconstructed / "link_prediction" / "PQ" / "filtered"
            qa_dir = reconstructed / "qa" / "PQ" / "filtered"
            graph_dir.mkdir(parents=True)
            qa_dir.mkdir(parents=True)

            (graph_dir / "triplets.txt").write_text(
                "a\tr\tb\na\ts\tc\nb\tt\td\nc\tu\te\n",
                encoding="utf-8",
            )
            (graph_dir / "relation_data.csv").write_text(
                "Title,RID\nrelation r,r\nrelation s,s\n"
                "relation t,t\nrelation u,u\n",
                encoding="utf-8",
            )

            fields = [
                "Question-Number", "Question", "Question-Family-ID",
                "Source-Entity", "Answer-Entity", "Graph-Answer-Entity",
                "Multi-Paths", "Multi-Paths-Key", "Hops", "SplitLabel",
            ]
            records = [
                {
                    "Question-Number": str(index),
                    "Question": "Who is related?",
                    "Question-Family-ID": f"family{index}",
                    "Source-Entity": "a",
                    "Answer-Entity": repr(["d", "e"]),
                    "Graph-Answer-Entity": repr(["d", "e"]),
                    "Multi-Paths": repr([
                        [[["a", "r", "b"], ["b", "t", "d"]]],
                        [[["a", "s", "c"], ["c", "u", "e"]]],
                    ]),
                    "Multi-Paths-Key": repr(["r->t", "s->u"]),
                    "Hops": 2,
                    "SplitLabel": split,
                }
                for index, split in enumerate(("train", "dev", "test"))
            ]
            with open(qa_dir / "pq_qa_nhop.csv", "w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)

            self.assertEqual(
                resolve_artifacts(reconstructed, "pq", "filtered")[0],
                Path("link_prediction/PQ/filtered/triplets.txt"),
            )
            args = SimpleNamespace(
                dataset="pq", graph_regime="filtered",
                reconstruction_root=reconstructed,
                supervision_chains="first",
                output_root=temp / "prepared",
                glove=None, glove_source="huggingface",
                glove_model="NeuML/glove-6B", glove_revision="main",
                emb_size=300,
            )
            with patch(
                "script.preprocess.prepare_pathquestion.make_embeddings"
            ) as make_embeddings:
                prepare(args)
                self.assertEqual(make_embeddings.call_count, 1)

            output = temp / "prepared" / "pq_filtered"
            train = [
                json.loads(line) for line in
                (output / "train_data.txt").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(train), 1)
            self.assertEqual(train[0][2][0][-1], ["s", [], 0])
            self.assertEqual(train[0][2][0][0], ["r", [], 1])

            test = json.loads(
                (output / "test_metadata.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(test["gold_relation_chains"], [["r", "t"], ["s", "u"]])
            self.assertEqual(test["answers"], ["d", "e"])
            self.assertEqual(test["question_family_id"], "family2")
            self.assertEqual(test["reference_paths"][0][0][0], ["a", "r", "b"])

            args.supervision_chains = "all"
            with patch("script.preprocess.prepare_pathquestion.make_embeddings"):
                prepare(args)
            expanded = (output / "train_data.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(expanded), 2)


class FamilyMacroTests(unittest.TestCase):
    def test_families_receive_equal_weight(self):
        rows = []
        for number, family, hit in (
            (1, "family_a", 1), (2, "family_a", 1),
            (3, "family_b", 0),
        ):
            rows.append({
                "question_number": number, "question_family_id": family,
                "hops": 2, "hits_at_1": hit,
                "expected_hits_at_1_uniform": float(hit),
                "relation_chain_correct": hit,
                "RED": 1 - hit, "F1_REL": float(hit),
                "answer_reachable": bool(hit), "answer_set_exact": bool(hit),
                "terminal_frontier_size": 1,
            })
        summary = family_macro_summary(rows)
        self.assertEqual(summary["all"]["family_count"], 2)
        self.assertEqual(summary["all"]["question_count"], 3)
        self.assertEqual(summary["all"]["hits_at_1"], 0.5)
        self.assertEqual(summary["by_hop"]["2"]["RED"], 0.5)


if __name__ == "__main__":
    unittest.main()
