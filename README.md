# Can World Knowledge Improve Plausibility Judgments?

A small NumPy experiment developed alongside a linguistics course report at the University of Tokyo, after studying Word2vec with *Deep Learning from Scratch 2*.

The task is to predict whether a subject can perform an action in an ordinary, literal real-world setting: for example, **an eagle can fly**, but **a cat cannot fly**. I compare a classifier using CBOW word vectors with one that also receives simple facts such as whether the subject is an animal, has wings, or is an artifact.

## Experiment at a glance

- **Data:** 60 nouns × 12 verbs = 720 synthetic examples.
- **Evaluation:** five noun-based folds; each fold trains on 576 examples and tests on 144 examples involving different nouns.
- **Implementation:** frozen 100-dimensional CBOW vectors and a small two-layer MLP using NumPy and reusable book-based code. No PyTorch or Transformers.
- **Repetition:** five random seeds; results are mean ± sample standard deviation across seeds.

| Model | Balanced accuracy |
|---|---:|
| CBOW + MLP | 0.744 ± 0.007 |
| CBOW + shuffled knowledge | 0.697 ± 0.021 |
| Knowledge rule | 0.832 ± 0.000 |
| CBOW + knowledge | **0.963 ± 0.003** |

Balanced accuracy averages the fraction of possible examples classified correctly and the fraction of impossible examples classified correctly. The knowledge features helped on this dataset, including for nouns excluded from classifier training. These nouns already have pretrained CBOW vectors; this is not a test of entirely unknown vocabulary.

Read the [experiment report](world_knowledge_experiment/README.md) for the data format, model comparisons, full results, and limitations.

## Run

From the repository root, with Python 3.11 or later and uv installed:

```sh
uv run --frozen python world_knowledge_experiment/run_repeated_experiment.py --jobs 5
```

This runs five seeds for 100 epochs each. Use `--jobs 1` for sequential execution. A previous five-process run took about 24 seconds on the development machine; runtime depends on the environment. The experiment runs on the CPU, although the repository's dependency list also includes CuPy for other learning exercises.

```sh
uv run --frozen python -m unittest world_knowledge_experiment/test_experiment.py
```

Results are saved under [`world_knowledge_experiment/results/`](world_knowledge_experiment/results/), including per-seed metrics and predictions for every example.

## Repository layout

- `world_knowledge_experiment/`: experiment code, data, report, and saved results.
- `cbow.py`, `cbow_train.py`, `cbow_params.pkl`: CBOW implementation, training script, and pretrained vectors.
- `common/`, `toolkits/`, `dataset/`: reusable layers, training utilities, and dataset helpers based on the book.
- `deep-learning-from-scratch-2-master/`: downloaded reference code, retained unchanged.

## Scope and AI assistance

This is an exploratory student project using a small, synthetic dataset. Labels and knowledge features were developed together and are not independently validated. The results do not establish the effectiveness of world knowledge in general or a solution to language-model hallucinations.

I used OpenAI Codex to create the classification dataset, generate and revise new experiment code, and edit the README. I questioned and revised the proposed design and code and made manual changes.
