"""Small fixture tests for THESEUS-to-UHop relation-choice conversion."""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pathlib import Path
from types import SimpleNamespace

from script.preprocess.prepare_theseus import (
    candidates_and_frontier, gold_chain, make_embeddings, parse_answers,
    parse_paths, prepare, read_graph,
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
                    "Question-Number", "Question", "Question-Paraphrased",
                    "Source-Entity", "Answer-Entity", "Paths", "Hops", "SplitLabel"
                ])
                writer.writeheader()
                for i, split in enumerate(("train", "dev", "test"), 1):
                    writer.writerow({
                        "Question-Number": i,
                        "Question": "Who is related to a?",
                        "Question-Paraphrased": repr([
                            "Who is connected to a?", "Who is associated with a?"
                        ]),
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
            if os.environ.get("UHOP_SMOKE_TRAINING") == "1":
                # Test original UHop training with prepared data and no GPU.
                repo = Path(__file__).resolve().parents[1]
                for model in ("HR_BiLSTM_plus", "ABWIM_plus"):
                    checkpoint = Path(tmp) / ("checkpoint_" + model)
                    checkpoint.mkdir()
                    command = [
                        sys.executable, "main.py", "--train",
                        "--framework", "UHop", "--model", model,
                        "--dataset", "kinship", "--data_dir", str(target),
                        "--path", str(checkpoint), "--device", "cpu",
                        "--num_workers", "0", "--emb_size", "50",
                        "--hidden_size", "8", "--epoch_num", "1",
                        "--dynamic", "recurrent",
                    ]
                    environment = os.environ.copy()
                    environment["OMP_NUM_THREADS"] = "1"
                    environment["MKL_NUM_THREADS"] = "1"
                    completed = subprocess.run(
                        command, cwd=repo / "src", env=environment,
                        capture_output=True, text=True, timeout=180
                    )
                    self.assertEqual(
                        completed.returncode, 0,
                        model + "\n" + completed.stdout[-3000:] +
                        "\n" + completed.stderr[-3000:]
                    )
                    self.assertTrue((checkpoint / "model.pth").is_file())
                    evaluation_path = Path(tmp) / ("navigation_" + model)
                    evaluation = subprocess.run(
                        [
                            sys.executable, "script/evaluate_theseus.py",
                            "--checkpoint", str(checkpoint),
                            "--offline-source", str(source),
                            "--split", "test", "--device", "cpu",
                            "--max-hops", "3", "--batch-size", "4",
                            "--output", str(evaluation_path)
                        ],
                        cwd=repo, env=environment,
                        capture_output=True, text=True, timeout=120
                    )
                    self.assertEqual(
                        evaluation.returncode, 0,
                        model + "\n" + evaluation.stdout[-3000:] +
                        "\n" + evaluation.stderr[-3000:]
                    )
                    self.assertTrue(evaluation_path.with_suffix(".summary.json").is_file())
                    summary = json.loads(
                        evaluation_path.with_suffix(".summary.json").read_text()
                    )
                    self.assertIn("hits_at_1", summary["all"])
                    self.assertIn("expected_hits_at_1_uniform", summary["all"])
                    self.assertIn("RED", summary["all"])
                    self.assertIn("F1_REL", summary["all"])
                    paraphrase_output = Path(tmp) / ("paraphrase_" + model)
                    paraphrase_test = subprocess.run(
                        [
                            sys.executable, "script/evaluate_theseus.py",
                            "--checkpoint", str(checkpoint),
                            "--offline-source", str(source),
                            "--split", "test", "--device", "cpu",
                            "--max-hops", "3", "--batch-size", "4",
                            "--question-variant", "paraphrased",
                            "--paraphrase-index", "all",
                            "--glove", str(glove),
                            "--output", str(paraphrase_output),
                        ],
                        cwd=repo, env=environment,
                        capture_output=True, text=True, timeout=120
                    )
                    self.assertEqual(
                        paraphrase_test.returncode, 0,
                        model + "\n" + paraphrase_test.stdout[-3000:] +
                        "\n" + paraphrase_test.stderr[-3000:]
                    )
                    para_summary = json.loads(
                        paraphrase_output.with_suffix(".summary.json").read_text()
                    )
                    self.assertEqual(para_summary["all"]["question_count"], 1)
                    self.assertEqual(para_summary["all"]["count"], 2)
                    self.assertEqual(
                        para_summary["question_variant"], "paraphrased"
                    )

    def test_huggingface_glove_vectors_and_local_fallback(self):
        from safetensors.numpy import save_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model.safetensors"
            vocab_path = root / "vocab.json"
            source_vectors = np.stack([
                np.full(300, 0.25, dtype=np.float32),
                np.full(300, -0.75, dtype=np.float32),
                np.full(300, 0.125, dtype=np.float32),
                np.full(300, 0.5, dtype=np.float32),
            ])
            save_file({"vectors": source_vectors}, str(model_path))
            vocab_path.write_text(json.dumps({
                "tokens": {"who": 0, "r": 1, "related": 2, "Who": 3}
            }), encoding="utf-8")

            def fake_download(*, repo_id, filename, revision):
                self.assertEqual(repo_id, "NeuML/glove-6B")
                self.assertEqual(revision, "test-revision")
                return str(root / filename)

            with patch("huggingface_hub.hf_hub_download", side_effect=fake_download):
                with patch("builtins.print") as prints:
                    make_embeddings(
                        root, ["Who Who WHO is related?"], {"r": "r"}, None, 300,
                        glove_source="huggingface",
                        glove_model="NeuML/glove-6B",
                        glove_revision="test-revision",
                    )
                messages = [call.args[0] for call in prints.call_args_list]
                self.assertIn("GloVe coverage: 4/5 unique tokens (300d)", messages)
                self.assertTrue(any(
                    message.startswith("GloVe occurrence coverage: 5/6 ")
                    for message in messages
                ))
                with self.assertRaisesRegex(ValueError, "dimension"):
                    make_embeddings(
                        root, ["Who is related?"], {"r": "r"}, None, 50,
                        glove_source="huggingface",
                        glove_model="NeuML/glove-6B",
                        glove_revision="test-revision",
                    )

            word2id = json.loads((root / "word2id.json").read_text())
            matrix = np.load(root / "word_emb.npy")
            self.assertEqual(matrix.shape[1], 300)
            # Exact mixed-case matches take priority over lowercase matches.
            np.testing.assert_array_equal(matrix[word2id["Who"]], source_vectors[3])
            np.testing.assert_array_equal(matrix[word2id["WHO"]], source_vectors[0])
            np.testing.assert_array_equal(matrix[word2id["related?"]], source_vectors[2])
            np.testing.assert_array_equal(matrix[word2id["r"]], source_vectors[1])
            np.testing.assert_array_equal(matrix[word2id["is"]], np.zeros(300))

    def test_local_glove_fallback_and_exact_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            glove = root / "glove.3d.txt"
            glove.write_text(
                "Who 0.4 0.4 0.4\n"
                "who 0.1 0.2 0.3\n"
                "related 0.3 0.2 0.1\n"
                "r 0.6 0.7 0.8\n",
                encoding="utf-8",
            )
            make_embeddings(
                root, ["Who WHO related?"], {"r": "r"}, glove, 3,
                glove_source="local",
            )
            word2id = json.loads((root / "word2id.json").read_text())
            matrix = np.load(root / "word_emb.npy")
            np.testing.assert_allclose(matrix[word2id["Who"]], [0.4, 0.4, 0.4])
            np.testing.assert_allclose(matrix[word2id["WHO"]], [0.1, 0.2, 0.3])
            np.testing.assert_allclose(matrix[word2id["related?"]], [0.3, 0.2, 0.1])
            np.testing.assert_allclose(matrix[word2id["r"]], [0.6, 0.7, 0.8])

    def test_missing_gold_relation_fails(self):
        with self.assertRaisesRegex(ValueError, "Gold relation"):
            candidates_and_frontier({"a": {"x": {"b"}}}, "a", ["not_in_graph"])


if __name__ == "__main__":
    unittest.main()
