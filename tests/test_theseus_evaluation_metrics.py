"""CPU-only checks for UHop's answer selection and THESEUS evaluation semantics."""
import csv
import io
import unittest

from script.theseus_evaluation_metrics import (
    aggregate, expected_uniform_hits_at_1, load_paraphrases,
    question_variants, relation_edit_distance, relation_f1, select_entity,
)


class TheseusEvaluationTests(unittest.TestCase):
    def test_released_paraphrase_lists_and_indices(self):
        metadata = [
            {"question_number": "1", "question": "Who is the father?", "hops": 2},
            {"question_number": "2", "question": "Who is the spouse?", "hops": 3},
        ]
        stream = io.StringIO()
        writer = csv.DictWriter(
            stream, fieldnames=[
                "Question-Number", "Question", "Question-Paraphrased"
            ]
        )
        writer.writeheader()
        writer.writerow({
            "Question-Number": "1", "Question": "Who is the father?",
            "Question-Paraphrased": repr([
                "Please name the father.", "Can you identify their father?"
            ])
        })
        writer.writerow({
            "Question-Number": "2", "Question": "Who is the spouse?",
            "Question-Paraphrased": repr([
                "Who is married to them?", "Please name their spouse."
            ])
        })
        stream.seek(0)
        paraphrases = load_paraphrases(stream, metadata)
        canonical = question_variants(metadata, "canonical")
        self.assertEqual(len(canonical), 2)
        self.assertIsNone(canonical[0][2])
        self.assertEqual(canonical[0][1], metadata[0]["question"])
        all_variants = question_variants(metadata, "paraphrased", paraphrases)
        self.assertEqual(len(all_variants), 4)
        self.assertEqual([row[2] for row in all_variants], [0, 1, 0, 1])
        selected = question_variants(
            metadata, "paraphrased", paraphrases, "1"
        )
        self.assertEqual(
            [row[1] for row in selected],
            ["Can you identify their father?", "Please name their spouse."]
        )
        with self.assertRaisesRegex(ValueError, "only 2 paraphrases"):
            question_variants(metadata, "paraphrased", paraphrases, "2")
        with self.assertRaisesRegex(ValueError, "requires"):
            question_variants(metadata, "canonical", paraphrase_index="0")

    def test_paraphrase_failures_are_not_silent(self):
        metadata = [{"question_number": "1", "question": "Original?"}]
        with self.assertRaisesRegex(ValueError, "Missing paraphrases"):
            load_paraphrases(io.StringIO(
                "Question-Number,Question,Question-Paraphrased\n"
            ), metadata)
        with self.assertRaisesRegex(ValueError, "nonempty list"):
            load_paraphrases(io.StringIO(
                "Question-Number,Question,Question-Paraphrased\n"
                "1,Original?,[]\n"
            ), metadata)

    def test_gold_blind_entity_selector(self):
        candidates = ["Q9", "Q2", "Q5"]
        self.assertEqual(
            select_entity(candidates, "kinship", "1", None, "first"), "Q2"
        )
        chosen = select_entity(
            candidates, "kinship", "1", 0, "uniform", seed=42
        )
        self.assertIn(chosen, candidates)
        self.assertEqual(
            chosen,
            select_entity(candidates[::-1], "kinship", "1", 0, "uniform", 42)
        )
        self.assertIsNone(select_entity([], "kinship", "1", None))
        self.assertAlmostEqual(
            expected_uniform_hits_at_1(candidates, ["Q9", "Q5"]), 2 / 3
        )
        self.assertEqual(expected_uniform_hits_at_1([], ["Q9"]), 0.0)

    def test_theseus_relation_metrics(self):
        self.assertEqual(relation_edit_distance(["r1", "r2"], ["r2", "r1"]), 2)
        self.assertEqual(relation_f1(["r1", "r2"], ["r2", "r1"]), 1.0)
        self.assertEqual(relation_edit_distance([], ["r1", "r2"]), 2)
        self.assertEqual(relation_f1([], ["r1", "r2"]), 0.0)
        self.assertEqual(relation_f1(["r1", "r1"], ["r1", "r2"]), 2 / 3)

    def test_summary_keeps_tie_breaks_separate_from_reachability(self):
        row = {
            "question_number": "a", "hops": 2,
            "hits_at_1": 0, "expected_hits_at_1_uniform": 0.5,
            "relation_chain_correct": True, "RED": 0, "F1_REL": 1.0,
            "answer_reachable": True, "answer_set_exact": False,
            "terminal_frontier_size": 2,
        }
        result = aggregate([row], "canonical", "uniform", 42)
        self.assertEqual(result["all"]["hits_at_1"], 0)
        self.assertEqual(result["all"]["expected_hits_at_1_uniform"], 0.5)
        self.assertEqual(result["all"]["answer_reachability"], 1.0)
        self.assertEqual(result["by_hop"]["2"]["RED"], 0)


if __name__ == "__main__":
    unittest.main()
