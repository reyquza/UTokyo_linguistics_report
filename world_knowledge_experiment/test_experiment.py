"""標準ライブラリunittestだけで行う簡単な検査。"""

import unittest

import numpy as np

from world_knowledge_experiment import run_experiment as experiment
from world_knowledge_experiment import run_repeated_experiment as repeated


class ExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_dir = experiment.PROJECT_DIR / "data"
        cls.knowledge = experiment.load_json(data_dir / "world_knowledge.json")
        cls.labels = experiment.load_json(data_dir / "plausibility_labels.json")
        cls.folds = experiment.load_json(data_dir / "noun_folds.json")
        cls.word_vecs, cls.word_to_id = experiment.load_embeddings(
            experiment.REPO_ROOT / "cbow_params.pkl"
        )
        cls.all_nouns = experiment.fold_nouns(cls.folds)
        cls.verbs = list(cls.knowledge["verbs"])

    def test_each_noun_appears_in_exactly_one_test_fold(self):
        self.assertEqual(len(self.all_nouns), len(set(self.all_nouns)))
        self.assertEqual(set(self.all_nouns), set(self.knowledge["subjects"]))

    def test_knowledge_does_not_contain_pair_labels(self):
        self.assertNotIn("positive_verbs", self.knowledge)
        self.assertNotIn("label", self.knowledge)

    def test_data_and_cbow_vocabulary_are_valid(self):
        experiment.validate_data(
            self.knowledge,
            self.labels,
            self.folds,
            self.word_to_id,
        )

    def test_knowledge_feature_contains_elementwise_product(self):
        examples = experiment.make_examples(
            ["eagle"],
            self.verbs,
            self.labels["positive_verbs"],
        )
        x, _ = experiment.build_features(
            examples,
            self.knowledge,
            self.word_vecs,
            self.word_to_id,
            add_knowledge=True,
        )
        embedding_size = self.word_vecs.shape[1]
        attribute_size = len(self.knowledge["attributes"])
        knowledge_part = x[:, embedding_size * 2 :]
        subject = knowledge_part[:, :attribute_size]
        required = knowledge_part[:, attribute_size : attribute_size * 2]
        matched = knowledge_part[:, attribute_size * 2 :]
        np.testing.assert_array_equal(matched, subject * required)

    def test_shuffle_preserves_attribute_distribution_but_changes_nouns(self):
        shuffled = experiment.shuffled_subject_knowledge(
            self.knowledge,
            self.all_nouns,
            seed=123,
        )
        original_values = sorted(
            tuple(sorted(value))
            for value in self.knowledge["subjects"].values()
        )
        shuffled_values = sorted(
            tuple(sorted(value))
            for value in shuffled["subjects"].values()
        )
        self.assertEqual(original_values, shuffled_values)
        changed_count = sum(
            self.knowledge["subjects"][noun] != shuffled["subjects"][noun]
            for noun in self.all_nouns
        )
        self.assertGreater(changed_count, len(self.all_nouns) // 2)

    def test_knowledge_rule_is_not_the_answer_key(self):
        examples = experiment.make_examples(
            self.all_nouns,
            self.verbs,
            self.labels["positive_verbs"],
        )
        predictions = experiment.knowledge_rule_predict(
            examples, self.knowledge
        )
        labels = np.asarray(
            [example["label"] for example in examples],
            dtype=np.int64,
        )
        disagreement_count = int(np.sum(predictions != labels))
        self.assertGreaterEqual(disagreement_count, 10)

    def test_short_training_run_returns_finite_probabilities(self):
        train_nouns = self.all_nouns[:10]
        examples = experiment.make_examples(
            train_nouns,
            self.verbs,
            self.labels["positive_verbs"],
        )
        x, labels = experiment.build_features(
            examples,
            self.knowledge,
            self.word_vecs,
            self.word_to_id,
            add_knowledge=True,
        )
        model = experiment.train_model(
            x,
            labels,
            seed=0,
            hidden_size=8,
            epochs=2,
            batch_size=32,
            learning_rate=0.01,
            weight_decay=0.0001,
        )
        predictions, probabilities = experiment.predict(model, x)
        self.assertEqual(predictions.shape, labels.shape)
        self.assertTrue(np.isfinite(probabilities).all())

    def test_repeated_mean_and_sample_std(self):
        result = repeated.mean_std([1.0, 2.0, 3.0])
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["std"], 1.0)


if __name__ == "__main__":
    unittest.main()
