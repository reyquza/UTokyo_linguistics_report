"""Test whether world knowledge improves sentence plausibility judgments.
Train with NumPy and code based on Deep Learning from Scratch 2.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.optimizer import Adam  # noqa: E402

try:
    from .models import TwoLayerMLP  # type: ignore[attr-defined]  # noqa: E402
except ImportError:
    from models import TwoLayerMLP  # noqa: E402


MODEL_NAMES = (
    "majority",
    "embedding_mlp",
    "shuffled_knowledge_mlp",
    "knowledge_rule",
    "knowledge_mlp",
)

DISPLAY_NAMES = {
    "majority": "Majority",
    "embedding_mlp": "CBOW + MLP",
    "shuffled_knowledge_mlp": "CBOW + shuffled knowledge",
    "knowledge_rule": "Knowledge rule",
    "knowledge_mlp": "CBOW + knowledge",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_embeddings(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Load pretrained CBOW parameters from a pickle file."""
    with path.open("rb") as file:
        params = pickle.load(file)

    required_keys = {"word_vecs", "word_to_id"}
    missing = required_keys - set(params)
    # set(dict) returns the dictionary keys.
    if missing:
        raise ValueError(f"Missing required keys in the CBOW file: {sorted(missing)}")

    word_vecs = np.asarray(params["word_vecs"], dtype=np.float32)
    word_to_id = params["word_to_id"]
    return word_vecs, word_to_id


def fold_nouns(folds: dict[str, Any]) -> list[str]:
    """  Extract all nouns from noun_folds.json as a list. """
    return [
        noun
        for fold in folds["folds"]
        for noun in fold["test_nouns"]
    ]


def validate_data(
    knowledge: dict[str, Any],
    labels: dict[str, Any],
    folds: dict[str, Any],
    word_to_id: dict[str, int],
) -> None:
    """Check data consistency, noun-fold coverage, and CBOW vocabulary coverage."""
    attributes = knowledge["attributes"]
    subjects = knowledge["subjects"]
    verbs = knowledge["verbs"]
    all_nouns = fold_nouns(folds)

    if len(attributes) != len(set(attributes)):
        raise ValueError("attributes contains duplicates.")
    if len(folds["folds"]) < 2:
        raise ValueError("Noun evaluation requires at least two folds.")

    fold_names = [fold["name"] for fold in folds["folds"]]
    if len(fold_names) != len(set(fold_names)):
        raise ValueError("Fold names must be unique.")
    for fold in folds["folds"]:
        nouns = fold["test_nouns"]
        if not nouns:
            raise ValueError(f"{fold['name']}: test_nouns is empty.")
        if len(nouns) != len(set(nouns)):
            raise ValueError(f"{fold['name']}: duplicate nouns within this fold.")
    if len(all_nouns) != len(set(all_nouns)):
        raise ValueError("A noun occurs in more than one fold.")

    if set(all_nouns) != set(subjects):
        raise ValueError("Nouns do not match between noun_folds.json and world_knowledge.json.")
    if set(all_nouns) != set(labels["positive_verbs"]):
        raise ValueError(
            "Nouns do not match between noun_folds.json and plausibility_labels.json."
        )

    attribute_set = set(attributes)
    for noun, active_attributes in subjects.items():
        unknown = set(active_attributes) - attribute_set
        if unknown:
            raise ValueError(f"{noun}: unknown features: {sorted(unknown)}")

    verb_set = set(verbs)
    for verb, definition in verbs.items():
        required = definition["requires"]
        if not required:
            raise ValueError(f"{verb}: required features are empty.")
        unknown = set(required) - attribute_set
        if unknown:
            raise ValueError(f"{verb}: unknown required features: {sorted(unknown)}")

    for noun, positive_verbs in labels["positive_verbs"].items():
        unknown = set(positive_verbs) - verb_set
        if unknown:
            raise ValueError(f"{noun}: labels contain unknown verbs: {sorted(unknown)}")

    needed_words = set(all_nouns) | verb_set
    missing_words = sorted(needed_words - set(word_to_id))
    if missing_words:
        raise ValueError(
            "Words missing from the CBOW vocabulary; update the data or use other embeddings: "
            f"{missing_words}"
        )


def article(noun: str) -> str:
    return "An" if noun[0].lower() in "aeiou" else "A"


def make_examples(
    nouns: list[str],
    verbs: list[str],
    positive_verbs: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Combine each noun with every verb to create sentences and labels."""
    examples = []
    for noun in nouns:
        positive_set = set(positive_verbs[noun])
        for verb in verbs:
            examples.append(
                {
                    "noun": noun,
                    "verb": verb,
                    "sentence": f"{article(noun)} {noun} can {verb}.",
                    "label": int(verb in positive_set),
                }
            )
    return examples


def normalized_embedding(
    word: str,
    word_vecs: np.ndarray,
    word_to_id: dict[str, int],
) -> np.ndarray:
    vector = word_vecs[word_to_id[word]].astype(np.float32, copy=True)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def multi_hot(active: list[str], attribute_to_id: dict[str, int]) -> np.ndarray:
    """ Encode active features as a multi-hot vector. """
    vector = np.zeros(len(attribute_to_id), dtype=np.float32)
    for name in active:
        vector[attribute_to_id[name]] = 1.0
    return vector


def build_features(
    examples: list[dict[str, Any]],
    knowledge: dict[str, Any],
    word_vecs: np.ndarray,
    word_to_id: dict[str, int],
    add_knowledge: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """  Convert examples into numeric feature vectors for the MLP. """
    attribute_to_id = {
        name: index for index, name in enumerate(knowledge["attributes"])
    }
    rows = []
    labels = []

    for example in examples:
        noun = example["noun"]
        verb = example["verb"]
        noun_embedding = normalized_embedding(noun, word_vecs, word_to_id)
        verb_embedding = normalized_embedding(verb, word_vecs, word_to_id)
        parts = [noun_embedding, verb_embedding]

        if add_knowledge:
            subject_attributes = multi_hot(
                knowledge["subjects"][noun], attribute_to_id
            )
            verb_requirements = multi_hot(
                knowledge["verbs"][verb]["requires"], attribute_to_id
            )
            attribute_matches = subject_attributes * verb_requirements
            """ The product is 1 only for features required by the verb and present in the noun. """
            parts += [subject_attributes, verb_requirements, attribute_matches]

        rows.append(np.concatenate(parts))
        labels.append(example["label"])

    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def shuffled_subject_knowledge(
    knowledge: dict[str, Any],
    nouns: list[str],
    seed: int,
) -> dict[str, Any]:
    """Shuffle noun-feature assignments while preserving size and feature distribution."""
    rng = np.random.default_rng(seed)
    original_indices = np.arange(len(nouns))
    while True:
        shuffled_indices = rng.permutation(len(nouns))
        if np.all(shuffled_indices != original_indices):
            break

    shuffled_subjects = {
        noun: list(knowledge["subjects"][nouns[source_index]])
        for noun, source_index in zip(nouns, shuffled_indices)
    }
    return {
        "attributes": list(knowledge["attributes"]),
        "subjects": shuffled_subjects,
        "verbs": knowledge["verbs"],
    }


def knowledge_rule_predict(
    examples: list[dict[str, Any]],
    knowledge: dict[str, Any],
) -> np.ndarray:
    """Predict possible if the noun has every required feature; no learning."""
    predictions = []
    for example in examples:
        subject_attributes = set(knowledge["subjects"][example["noun"]])
        required_attributes = set(
            knowledge["verbs"][example["verb"]]["requires"]
        )
        predictions.append(int(required_attributes <= subject_attributes))
    return np.asarray(predictions, dtype=np.int64)


def train_model(
    x: np.ndarray,
    labels: np.ndarray,
    seed: int,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> TwoLayerMLP:
    np.random.seed(seed)
    model = TwoLayerMLP(x.shape[1], hidden_size)
    optimizer = Adam(lr=learning_rate)
    rng = np.random.default_rng(seed + 10_000)

    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            batch_ids = order[start : start + batch_size]
            model.forward(x[batch_ids], labels[batch_ids])
            model.backward()

            if weight_decay > 0.0:
                for param, grad in zip(model.params, model.grads):
                    if param.ndim == 2:
                        grad += weight_decay * param
            optimizer.update(model.params, model.grads)

    return model


def predict(model: TwoLayerMLP, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits = model.predict_logits(x)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities.argmax(axis=1), probabilities[:, 1]


def classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, Any]:
    """ Compare labels and predictions to calculate evaluation metrics. """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    tp = int(np.sum((true_labels == 1) & (predicted_labels == 1)))
    tn = int(np.sum((true_labels == 0) & (predicted_labels == 0)))
    fp = int(np.sum((true_labels == 0) & (predicted_labels == 1)))
    fn = int(np.sum((true_labels == 1) & (predicted_labels == 0)))

    def safe_divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    accuracy = safe_divide(tp + tn, len(true_labels))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def summarize_folds(
    fold_results: list[dict[str, Any]],
    pooled_labels: np.ndarray,
    pooled_predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    """ Summarize per-fold metrics and pooled predictions. """
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    summary: dict[str, Any] = {}

    for model_name in MODEL_NAMES:
        pooled = classification_metrics(
            pooled_labels, pooled_predictions[model_name]
        )
        fold_mean_std = {}
        for metric_name in metric_names:
            values = np.asarray(
                [
                    fold["metrics"][model_name][metric_name]
                    for fold in fold_results
                ],
                dtype=np.float64,
            )
            fold_mean_std[metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
        summary[model_name] = {
            "pooled": pooled,
            "fold_mean_std": fold_mean_std,
        }

    knowledge_score = summary["knowledge_mlp"]["pooled"]["balanced_accuracy"]
    summary["pooled_balanced_accuracy_differences"] = {
        "knowledge_minus_embedding": (
            knowledge_score
            - summary["embedding_mlp"]["pooled"]["balanced_accuracy"]
        ),
        "knowledge_minus_shuffled": (
            knowledge_score
            - summary["shuffled_knowledge_mlp"]["pooled"]["balanced_accuracy"]
        ),
        "knowledge_minus_rule": (
            knowledge_score
            - summary["knowledge_rule"]["pooled"]["balanced_accuracy"]
        ),
    }
    return summary


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "fold",
        "noun",
        "verb",
        "sentence",
        "gold",
        "majority_prediction",
        "embedding_prediction",
        "embedding_possible_probability",
        "shuffled_knowledge_prediction",
        "shuffled_knowledge_possible_probability",
        "knowledge_rule_prediction",
        "knowledge_prediction",
        "knowledge_possible_probability",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any], noun_count: int, pair_count: int) -> None:
    print(f"5-fold noun cross-validation: {noun_count} nouns / {pair_count} pairs")
    print("\nPooled predictions (each noun is tested once)")
    print("model                          accuracy   balanced accuracy   F1")
    for model_name in MODEL_NAMES:
        metrics = summary[model_name]["pooled"]
        print(
            f"{DISPLAY_NAMES[model_name]:<31}"
            f"{metrics['accuracy']:.3f}      "
            f"{metrics['balanced_accuracy']:.3f}               "
            f"{metrics['f1']:.3f}"
        )

    print("\nBalanced accuracy across five noun folds (mean +/- std)")
    for model_name in MODEL_NAMES:
        metric = summary[model_name]["fold_mean_std"]["balanced_accuracy"]
        print(
            f"{DISPLAY_NAMES[model_name]:<31}"
            f"{metric['mean']:.3f} +/- {metric['std']:.3f}"
        )

    differences = summary["pooled_balanced_accuracy_differences"]
    print("\nPooled balanced-accuracy differences")
    for name, value in differences.items():
        print(f"{name}: {value:+.3f}")


def main(args: argparse.Namespace) -> dict[str, Any]:
    
    data_dir = PROJECT_DIR / "data"
    knowledge = load_json(data_dir / "world_knowledge.json")
    labels = load_json(data_dir / "plausibility_labels.json")
    folds = load_json(data_dir / "noun_folds.json")
    word_vecs, word_to_id = load_embeddings(args.embeddings)
    validate_data(knowledge, labels, folds, word_to_id)

    all_nouns = fold_nouns(folds)
    verbs = list(knowledge["verbs"])
    positive_verbs = labels["positive_verbs"]
    pooled_labels: list[int] = []
    pooled_predictions: dict[str, list[int]] = {
        name: [] for name in MODEL_NAMES
    }
    prediction_rows: list[dict[str, Any]] = []
    fold_results = []

    for fold_index, fold in enumerate(folds["folds"]):
        fold_name = fold["name"]
        test_nouns = fold["test_nouns"]
        test_set = set(test_nouns)
        train_nouns = [noun for noun in all_nouns if noun not in test_set]
        train_examples = make_examples(train_nouns, verbs, positive_verbs)
        test_examples = make_examples(test_nouns, verbs, positive_verbs)

        x_train_embedding, y_train = build_features(
            train_examples,
            knowledge,
            word_vecs,
            word_to_id,
            add_knowledge=False,
        )
        x_test_embedding, y_test = build_features(
            test_examples,
            knowledge,
            word_vecs,
            word_to_id,
            add_knowledge=False,
        )
        x_train_knowledge, _ = build_features(
            train_examples,
            knowledge,
            word_vecs,
            word_to_id,
            add_knowledge=True,
        )
        x_test_knowledge, _ = build_features(
            test_examples,
            knowledge,
            word_vecs,
            word_to_id,
            add_knowledge=True,
        )

        shuffled = shuffled_subject_knowledge(
            knowledge,
            all_nouns,
            seed=args.shuffle_seed + fold_index,
        )
        x_train_shuffled, _ = build_features(
            train_examples,
            shuffled,
            word_vecs,
            word_to_id,
            add_knowledge=True,
        )
        x_test_shuffled, _ = build_features(
            test_examples,
            shuffled,
            word_vecs,
            word_to_id,
            add_knowledge=True,
        )

        model_seed = args.seed + fold_index
        embedding_model = train_model(
            x_train_embedding,
            y_train,
            model_seed,
            args.hidden_size,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
        )
        shuffled_model = train_model(
            x_train_shuffled,
            y_train,
            model_seed,
            args.hidden_size,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
        )
        knowledge_model = train_model(
            x_train_knowledge,
            y_train,
            model_seed,
            args.hidden_size,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
        )

        embedding_pred, embedding_prob = predict(
            embedding_model, x_test_embedding
        )
        shuffled_pred, shuffled_prob = predict(
            shuffled_model, x_test_shuffled
        )
        knowledge_pred, knowledge_prob = predict(
            knowledge_model, x_test_knowledge
        )
        rule_pred = knowledge_rule_predict(test_examples, knowledge)
        majority_label = int(float(y_train.mean()) >= 0.5)
        majority_pred = np.full_like(y_test, majority_label)

        fold_predictions = {
            "majority": majority_pred,
            "embedding_mlp": embedding_pred,
            "shuffled_knowledge_mlp": shuffled_pred,
            "knowledge_rule": rule_pred,
            "knowledge_mlp": knowledge_pred,
        }
        fold_metrics = {
            name: classification_metrics(y_test, predictions)
            for name, predictions in fold_predictions.items()
        }
        fold_results.append(
            {
                "name": fold_name,
                "train_nouns": train_nouns,
                "test_nouns": test_nouns,
                "train_pairs": len(train_examples),
                "test_pairs": len(test_examples),
                "test_positive_rate": float(y_test.mean()),
                "metrics": fold_metrics,
            }
        )

        pooled_labels.extend(y_test.tolist())
        for model_name, predictions in fold_predictions.items():
            pooled_predictions[model_name].extend(predictions.tolist())

        for index, example in enumerate(test_examples):
            prediction_rows.append(
                {
                    "fold": fold_name,
                    "noun": example["noun"],
                    "verb": example["verb"],
                    "sentence": example["sentence"],
                    "gold": example["label"],
                    "majority_prediction": int(majority_pred[index]),
                    "embedding_prediction": int(embedding_pred[index]),
                    "embedding_possible_probability": (
                        f"{embedding_prob[index]:.6f}"
                    ),
                    "shuffled_knowledge_prediction": int(
                        shuffled_pred[index]
                    ),
                    "shuffled_knowledge_possible_probability": (
                        f"{shuffled_prob[index]:.6f}"
                    ),
                    "knowledge_rule_prediction": int(rule_pred[index]),
                    "knowledge_prediction": int(knowledge_pred[index]),
                    "knowledge_possible_probability": (
                        f"{knowledge_prob[index]:.6f}"
                    ),
                }
            )

    pooled_labels_array = np.asarray(pooled_labels, dtype=np.int64)
    pooled_prediction_arrays = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in pooled_predictions.items()
    }
    summary = summarize_folds(
        fold_results,
        pooled_labels_array,
        pooled_prediction_arrays,
    )
    attribute_size = len(knowledge["attributes"])
    embedding_size = int(word_vecs.shape[1])
    results = {
        "config": {
            "embeddings": str(args.embeddings.resolve()),
            "embedding_shape": list(word_vecs.shape),
            "seed": args.seed,
            "shuffle_seed": args.shuffle_seed,
            "hidden_size": args.hidden_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "data": {
            "nouns": all_nouns,
            "verbs": verbs,
            "attributes": knowledge["attributes"],
            "noun_count": len(all_nouns),
            "pair_count": len(all_nouns) * len(verbs),
            "positive_rate": float(pooled_labels_array.mean()),
            "fold_count": len(folds["folds"]),
            "embedding_input_size": embedding_size * 2,
            "knowledge_input_size": embedding_size * 2 + attribute_size * 3,
        },
        "folds": fold_results,
        "summary": summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    write_predictions(
        args.output_dir / "cross_validation_predictions.csv",
        prediction_rows,
    )
    print_summary(summary, len(all_nouns), len(prediction_rows))
    print(f"\nSaved results to: {args.output_dir.resolve()}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=REPO_ROOT / "cbow_params.pkl",
        help="CBOW pickle file (default: locally trained vectors at the repository root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "results",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=20_260_730)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
