"""Check free-running UHop evaluation against the original RC/TD batching."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "script"))

from Framework import Framework
import evaluate_theseus as navigation


WORDS = {"PADDING": 0, "<unk>": 1, "question": 2, "a": 3, "b": 4, "c": 5}
RELATIONS = {"PADDING": 0, "<unk>": 1, "r0": 2, "r1": 3, "r2": 4}
LABELS = {"r0": "a", "r1": "a_b", "r2": "a_b_c"}


class PaddingSensitiveModel:
    """Deterministic tensor model sensitive to every padded input."""

    def __call__(self, question, relation_text, relations,
                 previous_text, previous_relations):
        output = relations.float().sum(1) + relation_text.float().sum(1) / 10
        for index, previous in enumerate(previous_relations):
            output = output + previous.float().sum(1) / (index + 2)
        for index, previous in enumerate(previous_text):
            output = output + previous.float().sum(1) / (index + 4)
        return output


class JointTerminationTests(unittest.TestCase):
    def test_mixed_depth_scores_match_original_framework_across_chunks(self):
        """TD pads previous and next actions together, even across chunks."""
        actions = [("r0", []), ("r1", ["r0"]), ("r2", ["r0"])]
        model = PaddingSensitiveModel()

        for dynamic in ("recurrent", "flatten", "none"):
            with self.subTest(dynamic=dynamic):
                options = SimpleNamespace(
                    dynamic=dynamic, framework="UHop", device="cpu",
                    q_representation="lstm", margin=0.5
                )
                scores = navigation.score_actions(
                    model, "question", actions, WORDS, RELATIONS,
                    LABELS, options, batch_size=2
                )
                framework = Framework(options, WORDS, RELATIONS)
                def convert(relation, history, label):
                    return (
                        navigation.tokenize(relation, RELATIONS),
                        navigation.tokenize(LABELS[relation], WORDS, "._"),
                        [navigation.tokenize(r, RELATIONS) for r in history],
                        [navigation.tokenize(LABELS[r], WORDS, "._")
                         for r in history],
                        label
                    )
                # Framework places the positive next action before the
                # negative previous and alternate actions in one model call.
                _, _, framework_scores = framework._single_UHop_step(
                    model, [WORDS["question"]],
                    [convert("r1", ["r0"], 1)],
                    [convert("r0", [], 0), convert("r2", ["r0"], 0)]
                )
                expected = [framework_scores[1][0],
                            framework_scores[0][0], framework_scores[2][0]]
                self.assertEqual(len(scores), 3)
                for actual, target in zip(scores, expected):
                    self.assertAlmostEqual(actual, target, places=5)

    def test_real_hr_bilstm_joint_scores_match_training(self):
        """Catch padding differences in the actual LSTM, not only a spy."""
        import numpy as np
        from model.HR_BiLSTM_plus import Model

        torch.manual_seed(17)
        args = SimpleNamespace(
            word_embedding=np.random.default_rng(17).normal(
                size=(len(WORDS), 8)
            ).astype(np.float32),
            rela_vocab_size=len(RELATIONS), emb_size=8, hidden_size=5,
            q_representation="lstm", dropout_rate=0.0,
            reduce_method="dense", dynamic="recurrent",
            device="cpu", framework="UHop", margin=0.5,
        )
        model = Model(args).eval()
        with torch.no_grad():
            scores = navigation.score_actions(
                model, "question",
                [("r0", []), ("r1", ["r0"]), ("r2", ["r0"])],
                WORDS, RELATIONS, LABELS, args, batch_size=2
            )
            def convert(relation, history, label):
                return (
                    navigation.tokenize(relation, RELATIONS),
                    navigation.tokenize(LABELS[relation], WORDS, "._"),
                    [navigation.tokenize(r, RELATIONS) for r in history],
                    [navigation.tokenize(LABELS[r], WORDS, "._")
                     for r in history],
                    label
                )
            framework = Framework(args, WORDS, RELATIONS)
            _, _, reference = framework._single_UHop_step(
                model, [WORDS["question"]],
                [convert("r1", ["r0"], 1)],
                [convert("r0", [], 0), convert("r2", ["r0"], 0)]
            )
        for actual, expected in zip(
            scores, (reference[1][0], reference[0][0], reference[2][0])
        ):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_stop_decision_is_joint_and_ignore_mode_is_diagnostic(self):
        graph = {
            "source": {"r0": {"middle"}},
            "middle": {"r1": {"answer"}},
        }
        model_args = SimpleNamespace(device="cpu", dynamic="recurrent")
        learned_args = SimpleNamespace(
            max_hops=3, batch_size=2, termination_mode="learned"
        )
        calls = []

        def fake_scores(model, question, actions, *unused):
            calls.append(actions)
            if len(actions) == 2:
                # Previous action is preferable in joint TD scoring.
                return [10.0, 9.0]
            return [10.0]

        with patch.object(navigation, "score_actions", side_effect=fake_scores):
            predicted, terminal, reason = navigation.navigate(
                object(), "question", "source", graph, WORDS, RELATIONS,
                LABELS, model_args, learned_args
            )
        self.assertEqual(predicted, ["r0"])
        self.assertEqual(terminal, ["middle"])
        self.assertEqual(reason, "learned_termination")
        self.assertIn(
            [("r0", []), ("r1", ["r0"])], calls
        )

        ignored_args = SimpleNamespace(
            max_hops=3, batch_size=2, termination_mode="ignore"
        )
        with patch.object(navigation, "score_actions", side_effect=fake_scores):
            predicted, terminal, reason = navigation.navigate(
                object(), "question", "source", graph, WORDS, RELATIONS,
                LABELS, model_args, ignored_args
            )
        self.assertEqual(predicted, ["r0", "r1"])
        self.assertEqual(terminal, ["answer"])
        self.assertEqual(reason, "no_outgoing_relations")

        ignored_args.max_hops = 2
        with patch.object(navigation, "score_actions", side_effect=fake_scores):
            _, _, reason = navigation.navigate(
                object(), "question", "source", graph, WORDS, RELATIONS,
                LABELS, model_args, ignored_args
            )
        self.assertEqual(reason, "max_hops")


if __name__ == "__main__":
    unittest.main()
