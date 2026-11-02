### Plausibility Judgments with World Knowledge

#### Overview

This experiment asks whether adding simple facts about the world to CBOW word vectors improves judgments about what a subject can realistically do.

```text
An eagle can fly.  -> possible
A cat can fly.    -> impossible
```

The classifier is tested on nouns excluded from its supervised training examples. On this synthetic dataset, adding knowledge improved balanced accuracy from **0.744 ± 0.007** to **0.963 ± 0.003** across five random seeds.

#### Data and inputs

CBOW was trained on the Penn Treebank (PTB) dataset used in *Deep Learning from Scratch 2*. Its 100-dimensional word vectors remain fixed during classifier training.

OpenAI Codex was used to create the classification dataset, with instructions to avoid overly obscure words. It contains all combinations of 60 nouns and 12 verbs: **720 examples** in the form `A/An noun can verb`. The classifier receives vectors representing the noun and verb, rather than processing the full sentence.

The verbs are `fly`, `walk`, `eat`, `speak`, `float`, `grow`, `sleep`, `think`, `read`, `write`, `cook`, and `hunt`. Labels describe ordinary capabilities in typical situations, excluding metaphors, accidents, and exceptional modifications.

[`data/world_knowledge.json`](data/world_knowledge.json) stores general facts about each noun and one basic condition associated with each verb. For example:

```text
"cat": ["physical", "mobile", "animate", "living", "animal", "has_legs"]
"fly": {"requires": ["mobile"]}
```

There are 19 features, including `animal`, `artifact`, `has_wings`, and `has_legs`. Each noun is represented by 19 values: 1 if the feature applies and 0 otherwise. The verb's condition is encoded in the same way. These are deliberately broad conditions: being mobile alone does not mean something can fly.

[`data/plausibility_labels.json`](data/plausibility_labels.json) separately lists the verbs labeled possible for each noun:

```text
"cat": ["walk", "eat", "grow", "sleep", "think", "hunt"]
```

For example, `A cat can sleep` is labeled possible and `A cat can cook` is labeled impossible. Verbs absent from a noun's list are labeled impossible.

#### Models

| Model | How it makes a prediction |
|---|---|
| Majority | Always predicts the most common training label. |
| CBOW + MLP | Concatenates the noun and verb vectors and feeds them into a small neural-network classifier. |
| CBOW + shuffled knowledge | Uses the same inputs as the knowledge model, but randomly assigns each noun another noun's feature list. This checks whether adding more input dimensions alone explains the improvement. |
| Knowledge rule | Does not learn. Predicts possible if the noun has every feature listed as a condition for the verb, and impossible otherwise. |
| CBOW + knowledge | Gives the classifier the word vectors, the noun's features, the verb's conditions, and the features present in both lists. |

For example, `walk` has the condition `has_legs`. The rule therefore predicts cat–walk as possible. It also predicts cat–speak as possible because the condition for `speak` is `animate`, which the cat has. This is an error under the dataset's labels. The learned model can use the additional inputs and training examples to go beyond this simple rule.

The baseline input has 200 values: 100 for the noun and 100 for the verb. The knowledge model has 257 values: the same 200 plus three 19-value feature vectors. The third vector is the elementwise product of the noun and verb feature vectors, so an entry is 1 only when both vectors contain 1 at that position.

Both classifiers use a two-layer MLP with 32 hidden units and a sigmoid activation, trained for 100 epochs. The implementation uses NumPy and the book-based `Affine`, `Sigmoid`, `Softmax_with_Loss`, and `Adam` implementations in this repository. It does not use PyTorch or Transformers.

#### Evaluation

Nouns are divided into five fixed groups of 12. Each fold trains on **48 nouns × 12 verbs = 576 examples** and tests on **12 different nouns × 12 verbs = 144 examples**. Every noun is tested once per seed. This avoids placing different sentences about the same noun in both the training and test sets of a fold.

The test nouns are unseen only during classifier training: their pretrained CBOW vectors and supplied knowledge features are available.

The entire five-fold experiment is repeated with seeds 0–4. For each seed, predictions from all five folds are pooled into one score over 720 examples. The reported mean and sample standard deviation are calculated across these five scores, not across 25 independent experiments. Initialization and minibatch order vary; the shuffled-knowledge assignment also varies. The noun groups and pretrained embeddings stay fixed.

### Results

Values are mean ± sample standard deviation across five seeds. Balanced accuracy is the average of the correctly classified fractions within the possible and impossible classes, giving both classes equal weight.

| Model | Accuracy | Balanced accuracy | F1 |
|---|---:|---:|---:|
| Majority | 0.683 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 |
| CBOW + MLP | 0.798 ± 0.010 | 0.744 ± 0.007 | 0.651 ± 0.011 |
| CBOW + shuffled knowledge | 0.749 ± 0.026 | 0.697 ± 0.021 | 0.583 ± 0.032 |
| Knowledge rule | 0.785 ± 0.000 | 0.832 ± 0.000 | 0.739 ± 0.000 |
| CBOW + knowledge | **0.972 ± 0.002** | **0.963 ± 0.003** | **0.956 ± 0.003** |

Adding knowledge increased balanced accuracy by 0.219 on average over CBOW alone and by 0.266 over shuffled knowledge. The shuffled comparison supports the idea that meaningful noun–feature associations helped, rather than input size alone.

The rule made 155 errors out of 720 examples. The learned knowledge model exceeded it by 0.131 in balanced accuracy, so it did not simply reproduce that rule. This comparison alone does not establish how much the learned model relies on word vectors versus general features; a features-only learned baseline would be needed to separate those contributions.

### Avoiding an answer key disguised as knowledge

If the supplied facts directly encode the labels, high performance would say little about learning. To reduce this problem:

- No features such as `can_fly` or `can_speak` are included.
- Each verb has one broad condition. For example, `animate` is associated with `speak`, but not every animate subject can speak.
- The rule baseline checks how far simply matching these conditions gets. Its 155 errors show that this particular rule is not the answer key; they do not prove that labels and features are independent.
- The shuffled-knowledge model checks whether the correct association between nouns and features matters.

The labels and features were developed within the same project, so their independence cannot be assumed. The coarse verb conditions also make this a limited rule baseline, not the strongest possible hand-written system.

### Use of generative AI

I used OpenAI Codex to create the classification dataset, generate and revise new code, and edit the README. I repeatedly questioned the proposed design and code, requested improvements, and made manual changes. AI assistance was part of the development process rather than an independent source of ground-truth labels.

### Running the experiment

From the repository root:

```sh
uv run --frozen python world_knowledge_experiment/run_repeated_experiment.py --jobs 5
```

This runs five seeds for 100 epochs each with up to five worker processes. Use `--jobs 1` for sequential execution. A previous run took about 24 seconds on the development machine; runtime varies by environment.

Outputs under `world_knowledge_experiment/`:

- `results/repeated_results.json`: per-seed metrics, means, and standard deviations.
- `results/repeated_cross_validation_predictions.csv`: predictions for all 5 × 720 examples.
- `results/seed_runs/seed_*/`: detailed results for each seed.

Run the tests with:

```sh
uv run --frozen python -m unittest world_knowledge_experiment/test_experiment.py
```

### Limitations

This is an exploratory student experiment on a small, AI-assisted synthetic dataset. It demonstrates an improvement under the chosen vocabulary, labels, features, and noun splits. It does not establish the general effectiveness of world knowledge or a solution to language-model hallucinations.

The standard deviations reflect randomness in classifier training and, for the shuffled control, feature assignment. They do not measure uncertainty from selecting different vocabulary, annotators, noun splits, or CBOW pretraining runs. Independent human labeling and broader evaluation would be needed for stronger conclusions.
