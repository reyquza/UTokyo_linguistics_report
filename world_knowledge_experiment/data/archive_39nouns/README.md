# Archived 39-noun dataset

This folder preserves the dataset used before the expansion to 60 nouns and 12 verbs.

The earlier version contained 39 nouns, 8 verbs, and 312 pairs. Pooled balanced accuracy was 0.941 for Knowledge rule and 0.978 for CBOW + knowledge. The verb conditions aligned too closely with the labels, motivating an expanded vocabulary and a revised design with one coarse condition per verb. Results from the two datasets are not directly comparable.

The current experiment does not read this folder. It uses the 60-noun dataset in the parent `data/` directory.
