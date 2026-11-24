# Loop Breaker

Detecting AI debugging loops in vibe-coding sessions before they waste users' credits, and before those users churn.

A data science case study by **Keerthi Medha Anand** (MSc Advanced Data Science & AI, University of Liverpool).

## Why

Public reviews of Lovable keep returning to one pain point: credits spent while the AI goes round in circles fixing its own mistakes. I measured that with real review data, then built and evaluated a detector on simulated build sessions and designed an experiment to test a fix.

## 1. The problem (real data)

Source: the 200 most recent public Trustpilot reviews of lovable.dev (4 Oct – 24 Nov 2026), plus recent 1★ and 2★ reviews (135 negative reviews after de-duplication). Themes were tagged with keyword rules, and one review can have several themes.

| Finding | Value |
|---|---|
| Recent reviews mentioning credits or cost | 19.5% |
| Average rating with vs without a credits mention | 2.23★ vs 4.73★ |
| 1–2★ reviews mentioning credits | 74.1% |
| 1–2★ reviews mentioning credits **and** bugs/loops | 28.9% |
| 1–2★ reviewers who still praise ease or speed | 30.4% |

Details are in `review_findings.json`.

## 2. The model (simulated data)

`simulate_and_model.py` simulates 6,000 users and about 140k prompts, with a fixed seed. Each prompt carries only backward-looking features:

- the fix-prompt streak
- prompt similarity
- whether the same error repeated
- failed previews
- reverts
- credits spent since the last working preview

Results on a held-out 30% of users:

| Method | Precision | Recall |
|---|---|---|
| Rule: 3 fix prompts in a row | 0.72 | 0.62 |
| Gradient boosting | 0.86 | 0.72 |

The gradient boosting model reaches ROC-AUC 0.98 and PR-AUC 0.88, and it flags 64% of loops within their first 2 prompts.

## 3. The cost (simulated)

- 13.4% of credits are spent inside loops.
- 30-day churn is 38.6% for users who hit a loop, against 25.0% for users who don't.
- Churn rises with the number of credits wasted. This relationship is an assumption built into the simulation, informed by the review evidence.

## 4. The experiment: Loop Rescue

When the model flags a loop, offer a revert or a fresh chat and refund the credits spent in the loop.

- **Population:** users with a detected loop, randomised 50/50.
- **Primary metric:** 30-day churn.
- **Guardrails:** credits refunded, time to first published app, support tickets.
- **Sample size:** detecting a 3-point drop from 38.6% churn (α = 0.05, power 0.8) needs about 4,100 users per arm.

## Limitations

- The review samples are the most recent ones, because Trustpilot shows only 10 pages per filter without login.
- Keyword tagging is approximate.
- All product and session data is simulated. Real event data is needed to measure the true effects.

## Run it

```
pip install numpy pandas scipy scikit-learn
python simulate_and_model.py
```

The interactive report was built in Lovable.
