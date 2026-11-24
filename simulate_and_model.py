"""Loop Breaker: simulated build sessions, loop detection model, churn link, experiment sizing.

All session data here is SIMULATED (seed 42). The review analysis that motivates it uses
real public Trustpilot reviews (see review_findings.json).
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
N_USERS = 6000

# ---- 1. Simulate users and prompt-by-prompt build sessions ----
personas = ["Founder", "Small business", "Designer", "Developer", "Student"]
persona_p = [0.30, 0.25, 0.15, 0.15, 0.15]
loop_propensity = {"Founder": 0.30, "Small business": 0.34, "Designer": 0.28, "Developer": 0.15, "Student": 0.32}

prompt_rows, user_rows = [], []
for u in range(N_USERS):
    persona = rng.choice(personas, p=persona_p)
    complexity = rng.choice(["Simple", "Medium", "Complex"], p=[0.45, 0.35, 0.20])
    cx = {"Simple": 0.6, "Medium": 1.0, "Complex": 1.7}[complexity]
    n_prompts = int(rng.integers(8, 40))
    in_loop, loop_len, credits_in_loops, loops = False, 0, 0.0, 0
    err_code = None
    for i in range(n_prompts):
        if not in_loop and i >= 2 and rng.random() < loop_propensity[persona] * cx * 0.12:
            in_loop, loop_len, loops = True, 0, loops + 1
            err_code = int(rng.integers(1, 30))
        loop_flag = int(in_loop)
        if in_loop:
            loop_len += 1
            kind = "fix"
            similarity = np.clip(rng.normal(0.60, 0.17), 0, 1)      # fix prompts often repeat each other
            same_error = rng.random() < 0.55
            error_shown = err_code if same_error else int(rng.integers(1, 30))
            reverted = rng.random() < 0.18
            preview_ok = rng.random() < 0.30
            credits = float(rng.choice([1.0, 1.0, 1.5, 2.0]))
            credits_in_loops += credits
            if preview_ok and rng.random() < 0.6 or loop_len > rng.integers(4, 12):
                in_loop = False
        else:
            kind = rng.choice(["feature", "style", "fix"], p=[0.50, 0.22, 0.28])
            similarity = np.clip(rng.normal(0.50 if kind == "fix" else 0.35, 0.17), 0, 1)
            error_shown = int(rng.integers(1, 30)) if (kind == "fix" or rng.random() < 0.1) else None
            reverted = rng.random() < 0.04
            preview_ok = rng.random() < (0.65 if kind == "fix" else 0.85)
            credits = float(rng.choice([0.5, 1.0, 1.0, 1.5]))
        prompt_rows.append(dict(user=u, step=i, kind=kind, similarity=similarity, error=error_shown,
                                reverted=reverted, preview_ok=preview_ok, credits=credits,
                                in_loop=loop_flag))
    # churn within 30 days rises with credits wasted in loops (simulated assumption)
    base = {"Developer": 0.18, "Founder": 0.24, "Small business": 0.27, "Designer": 0.25, "Student": 0.35}[persona]
    logit = np.log(base / (1 - base)) + 0.09 * credits_in_loops
    churn = rng.random() < 1 / (1 + np.exp(-logit))
    user_rows.append(dict(user=u, persona=persona, complexity=complexity, n_prompts=n_prompts, loops=loops,
                          credits_in_loops=credits_in_loops, churn30=int(churn)))

P = pd.DataFrame(prompt_rows)
U = pd.DataFrame(user_rows)
P["total_credits"] = P.groupby("user")["credits"].transform("sum")

# ---- 2. Features available at prompt time (only looking backwards) ----
g = P.groupby("user")
P["is_fix"] = (P.kind == "fix").astype(int)
P["fix_streak"] = g["is_fix"].transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumsum() * s)
P["sim_mean3"] = g["similarity"].transform(lambda s: s.rolling(3, min_periods=1).mean())
P["prev_error"] = g["error"].shift(1)
P["same_error_as_prev"] = ((P.error.notna()) & (P.error == P.prev_error)).astype(int)
P["same_error_last3"] = g["same_error_as_prev"].transform(lambda s: s.rolling(3, min_periods=1).sum())
P["reverts_last5"] = g["reverted"].transform(lambda s: s.astype(int).rolling(5, min_periods=1).sum())
P["fails_last3"] = g["preview_ok"].transform(lambda s: (~s).astype(int).rolling(3, min_periods=1).sum())
P["credits_since_ok"] = g.apply(lambda d: d.credits.groupby(d.preview_ok.cumsum()).cumsum()).reset_index(level=0, drop=True)

feats = ["fix_streak", "sim_mean3", "same_error_last3", "reverts_last5", "fails_last3", "credits_since_ok", "step"]
y = P.in_loop
users_tr, users_te = train_test_split(U.user, test_size=0.3, random_state=42)
tr, te = P[P.user.isin(users_tr)], P[P.user.isin(users_te)]

# Baseline: a simple rule ("3 fix prompts in a row")
rule = (te.fix_streak >= 3).astype(int)
gb = GradientBoostingClassifier(random_state=42).fit(tr[feats], tr.in_loop)
p = gb.predict_proba(te[feats])[:, 1]
thr = 0.5
pred = (p >= thr).astype(int)

# Early detection: loops caught by the 2nd prompt of the loop
te2 = te.assign(p=p)
te2["loop_id"] = (te2.in_loop.diff().fillna(0) == 1).groupby(te2.user).cumsum()
loop_starts = te2[te2.in_loop == 1].groupby(["user", "loop_id"]).head(2)
caught_early = (loop_starts.groupby(["user", "loop_id"]).p.max() >= thr).mean()

model = dict(
    n_users=N_USERS, n_prompts=int(len(P)), loop_prompt_share=round(float(P.in_loop.mean()) * 100, 1),
    rule_precision=round(precision_score(te.in_loop, rule), 3), rule_recall=round(recall_score(te.in_loop, rule), 3),
    gb_auc=round(roc_auc_score(te.in_loop, p), 3), gb_pr_auc=round(average_precision_score(te.in_loop, p), 3),
    gb_precision=round(precision_score(te.in_loop, pred), 3), gb_recall=round(recall_score(te.in_loop, pred), 3),
    loops_flagged_within_2_prompts=round(float(caught_early) * 100, 1),
    feature_importance={f: round(float(v), 3) for f, v in sorted(zip(feats, gb.feature_importances_), key=lambda x: -x[1])},
)

# ---- 3. Credits wasted and churn link ----
share_credits_in_loops = P.loc[P.in_loop == 1, "credits"].sum() / P.credits.sum()
U["waste_band"] = pd.cut(U.credits_in_loops, [-0.1, 0, 5, 10, 1000], labels=["0", "1-5", "6-10", "10+"])
churn_by_band = U.groupby("waste_band", observed=True).churn30.agg(["mean", "count"]).reset_index()
lr = LogisticRegression().fit(U[["credits_in_loops"]], U.churn30)
odds_per_credit = float(np.exp(lr.coef_[0][0]))
loop_by_persona = U.groupby("persona").apply(lambda d: (d.loops > 0).mean()).round(3).to_dict()
loop_by_complexity = U.groupby("complexity").apply(lambda d: (d.loops > 0).mean()).round(3).to_dict()

# ---- 4. Experiment: "Loop Rescue" ----
# After the model flags a loop: suggest revert / fresh chat and refund the loop's credits.
p0 = float(U.churn30.mean())
looped = U[U.loops > 0]
p0_looped = float(looped.churn30.mean())
mde_abs = 0.03  # want to detect a 3-point drop in 30-day churn among users who hit a loop
alpha, power = 0.05, 0.8
z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
p1 = p0_looped - mde_abs
n_per_arm = int(np.ceil(((z_a * np.sqrt(2 * p0_looped * (1 - p0_looped)) + z_b * np.sqrt(p0_looped * (1 - p0_looped) + p1 * (1 - p1))) ** 2) / mde_abs ** 2))

out = dict(
    model=model,
    credits=dict(share_of_credits_spent_in_loops=round(float(share_credits_in_loops) * 100, 1),
                 avg_loop_credits_per_looping_user=round(float(looped.credits_in_loops.mean()), 1)),
    churn=dict(overall=round(p0 * 100, 1), looped_users=round(p0_looped * 100, 1),
               no_loop_users=round(float(U[U.loops == 0].churn30.mean()) * 100, 1),
               by_waste_band=[dict(band=str(r.waste_band), churn=round(r["mean"] * 100, 1), users=int(r["count"])) for _, r in churn_by_band.iterrows()],
               odds_multiplier_per_wasted_credit=round(odds_per_credit, 3)),
    loop_rate_by_persona=loop_by_persona, loop_rate_by_complexity=loop_by_complexity,
    experiment=dict(name="Loop Rescue", baseline_churn_looped=round(p0_looped * 100, 1), mde_points=3,
                    alpha=alpha, power=power, users_per_arm=n_per_arm,
                    primary_metric="30-day churn among users who hit a detected loop",
                    guardrails=["credits refunded per user", "time to first published app", "support tickets"]),
)
json.dump(out, open("model_results.json", "w"), indent=2)
print(json.dumps(out, indent=2))
