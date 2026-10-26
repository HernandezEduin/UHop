"""Small fixture tests for THESEUS-to-UHop relation-choice conversion."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from script.preprocess.prepare_theseus import (
    candidates_and_frontier, gold_chain, parse_answers, parse_paths, prepare,
    read_graph,
)


class TheseusAdapterTests(unittest.TestCase):
    def test_candidate_frontier_is_not_a_single_entity(self):
        graph = {
            "a": {"r": {"b", "c"}, "other": {"z"}},
            "b": {"s": {"d"}, "wrong": {"x"}},
            "c": {"s": {"e"}},
            "d": {"extra": {"q"}},
        }
        steps, frontier = candidates_and_frontier(graph, "a", ["r", "s"])
        self.assertEqual(frontier, {"d", "e"})
        self.assertEqual(steps[0], [
            ["other", [], 0], ["r", [], 1]
        ])
        self.assertEqual(steps[1], [
            ["s", ["r"], 1], ["wrong", ["r"], 0]
        ])
        self.assertEqual(steps[2], [["extra", ["r", "s"], 0]])

    def test_paths_and_answer_literals(self):
        paths = parse_paths("[['a', 'r', 'b'], ['b', 's', 'd']]")
        self.assertEqual(gold_chain({"Path-Key": ""}, paths), ["r", "s"])
        self.assertEqual(parse_answers("['d', 'e']"), ["d", "e"])
        self.assertEqual(parse_answers("d"), ["d"])

    def test_small_kinship_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            (source / "kg" / "orig").mkdir(parents=True)
            (source / "qa").mkdir(parents=True)
            (source / "kg" / "orig" / "triplets.txt").write_text(
                "a\tr\tb\na\tr\tc\na\tother\tz\n"
                "b\ts\td\nc\ts\te\nb\twrong\tx\nd\textra\tq\n",
                encoding="utf-8",
            )
            qa_file = source / "qa" / "kinship_qa_nhop.csv"
            with open(qa_file, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "Question-Number", "Question", "Source-Entity", "Answer-Entity",
                    "Paths", "Hops", "SplitLabel"
                ])
                writer.writeheader()
                for i, split in enumerate(("train", "dev", "test"), 1):
                    writer.writerow({
                        "Question-Number": i,
                        "Question": "Who is related to a?",
                        "Source-Entity": "a",
                        "Answer-Entity": "d",
                        "Paths": "[['a', 'r', 'b'], ['b', 's', 'd']]",
                        "Hops": 2,
                        "SplitLabel": split,
                    })
            glove = Path(tmp) / "glove.50d.txt"
            glove.write_text("Who " + " ".join(["0.5"] * 50) + "\n", encoding="utf-8")
            output = Path(tmp) / "out"
            args = SimpleNamespace(
                dataset="kinship", glove=glove, emb_size=50,
                revision="main", offline_source=source, output_root=output
            )
            prepare(args)
            target = output / "kinship"
            examples = [
                json.loads(line) for line in
                (target / "valid_data.txt").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0][2][0][1], ["r", [], 1])
            self.assertEqual(examples[0][2][1][0], ["s", ["r"], 1])
            metadata = json.loads(
                (target / "test_metadata.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["answers"], ["d"])
            self.assertEqual(metadata["gold_relations"], ["r", "s"])
            self.assertTrue((target / "word_emb.npy").is_file())
            self.assertEqual(json.loads(
                (target / "source.json").read_text(encoding="utf-8")
            )["graph_direction"], "forward_only")

    def test_missing_gold_relation_fails(self):
        with self.assertRaisesRegex(ValueError, "Gold relation"):
            candidates_and_frontier({"a": {"x": {"b"}}}, "a", ["not_in_graph"])


if __name__ == "__main__":
    unittest.main()
