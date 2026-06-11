# API reference

Generated from the source docstrings. Every public aggregate is an
[`Estimate`](#holdout.stats.Estimate); there is no public path to a bare float
metric.

## holdout — core types

The top-level package re-exports the types you touch most.

::: holdout
    options:
      members:
        - Case
        - Eval
        - Run
        - CaseResult
        - Completion
        - Target
        - Score
        - Scorer
        - run
        - arun

## holdout.stats

::: holdout.stats
    options:
      members:
        - Estimate
        - TestResult
        - bootstrap_ci
        - paired_diffs
        - paired_bootstrap_test
        - mcnemar_test
        - permutation_test
        - benjamini_hochberg
        - holm_bonferroni
        - PowerAnalysis
        - minimum_detectable_effect
        - required_sample_size
        - paired_binary_sd
        - sd_diff_from_scores

## holdout.regression

::: holdout.regression.compare.compare
    options:
      show_root_full_path: false

::: holdout.regression
    options:
      show_root_heading: false
      members:
        - RunComparison
        - MetricComparison

## holdout.leakage

::: holdout.leakage
    options:
      members:
        - check_contamination
        - check_contamination_embeddings
        - ContaminationReport
        - ContaminationFinding
        - find_near_duplicates
        - DuplicatePair
        - HoldoutLedger
        - DisciplineReport

## holdout.store

::: holdout.store
    options:
      members:
        - RunStore
        - StoredRunInfo

## holdout.testing

::: holdout.testing
    options:
      members:
        - assert_no_regression
        - assert_significant_improvement
        - assert_adequately_powered
        - assert_no_leakage
        - llm_eval

## holdout.providers

::: holdout.providers
    options:
      members:
        - ModelProvider
        - Anthropic
        - OpenAI
        - Ollama
        - MLX
        - StaticTarget
        - OllamaEmbeddings
        - OpenAIEmbeddings

## holdout.scorers

::: holdout.scorers
    options:
      members:
        - ExactMatch
        - RegexMatch
        - EmbeddingSimilarity
        - EmbeddingBackend
        - cosine_similarity
