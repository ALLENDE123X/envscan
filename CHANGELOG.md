# Changelog

## 0.5.0

57 new public rule ids, most older rules widened, and a second kind of
off switch. The work list came from a scan of 2,991 paper notes and 626
competitor checks, merged into 1,103 items. 98 of them became scanner code.

* New rule ids by module, all at `review`: Harbor task files (10), judge
  prompts and rubrics (11), answer matching (6), test quality (7), datasets
  (6), tools and environment state (9), grader logic (4), the agent image (1)
  and three in `rules.py`. `the internal notes` lists every one
  with its default, its Hub hits and what the independent readers found.
* Independent check before anything shipped on. A Hub scan with every new
  rule on gave 558 new hits on 368 environments. Five readers who did not
  write the rules judged 200 of them, at most 10 per rule. 94 were real, 45
  real but harmless, 57 not real and 4 unclear. So fewer than half of the new
  hits were worth showing, and the defaults below follow the readers, not the
  rule authors (`the internal notes`,
  `the internal notes` to `check_5.md`).
* 28 rule ids are off by default: the two from 0.4.0, 19 that the readers
  found wrong, harmless or low value too often, and 7 with no Hub hit that
  their own authors asked to ship off. `envscan --rules` marks them. Turn one
  on with `--enable RULE` or `ENVSCAN_ENABLE`.
* New: one branch of a rule can be off. A finding can carry a `variant`, and
  `rule-id:variant` strings in `OFF_VARIANTS` are dropped by default. Six
  v0.5 widenings of older rules failed the read and are off this way:
  `keyword-reward:similarity` (0 of 10 real), `aggregation-without-gate:v05`
  (2 of 10), `verdict-parse-semantics:v05` (0 of 4), `grader-fails-open:v05`
  (0 of 1), `format-heavy-reward:v05` (1 of 2) and
  `first-match-label-regex:first-fenced-block` (4 of 5 harmless). Turn one on
  with `--enable rule-id:variant`. The 0.4.0 behaviour of those rules is
  unchanged. Turning the branches off also hides 4 hits the readers called
  real.
* Fixes made from the readers' verdicts, one change per rule:
  `module-state-shared-across-rollouts`, `grader-failure-scored-zero`,
  `exact-compare-no-normalization`, `numeric-tolerance-too-loose`,
  `duplicate-prompts-conflicting-gold`, `judge-template-static-review`,
  `verifier-deserializes-agent-artifact`, `open-fetch-tool` and the no-clip
  branch of `kernel-perf-grader`. Checked again on the judged hits: all 36
  hits the readers called real on these rules still fire, and 13 of the 23
  they called not real or harmless are gone. That check is in sample, because
  the fixes were written from the same verdicts.
* New rule `judge-verdict-substring-match` (hack, high). It was a branch of
  the hygiene rule `first-match-label-regex`. A grader that reads the judge's
  verdict with `"correct" in reply` also matches "incorrect", so the judge's
  no is paid as a yes. Both Hub hits the reader saw pay wrong answers in
  full.
* `writable-reward-channel` has a new branch for Harbor tasks whose test
  script runs agent-written code in the agent's box and then writes only
  `reward.txt`. Harbor 0.26.0 and the verifiers Harbor taskset both clear the
  reward files, run the script, then read `reward.json` before `reward.txt`.
  The branch fires on 26 of 89 Terminal-Bench 2.0 tasks, and an independent
  read of 20 of them called 17 real by reading. This mechanism is already
  public (SWE-bench Pro issue 133, Terminal-Bench issue 2086). The scanner
  detects it. It did not discover it. Nothing was run.
* Labeled set: verdict scores are the same as 0.4.0 (`likely` 0.97 precision
  and 0.66 recall, `likely` or `review` 0.90 and 0.81).
* Published labels: on Terminal-Bench 2.0 the task-specific rules now flag 30
  tasks and 7 of the 18 with a published hack (0.4.0: 14 and 4), so precision
  is 0.23 and recall 0.39 (0.4.0: 0.29 and 0.22). SWE-Bench Pro V2 is
  unchanged at 1 of 8. `results/groundtruth/score_v0.5.txt`.
* Hub verdicts with the final defaults: 215 likely, 369 review, 1,322 clean
  of 1,906 scanned (0.4.0: 216, 324, 1,366). 45 environments moved from clean
  to review. Most of those come from two rules, state shared across rollouts
  and a grader failure scored as zero. One environment that was `likely` is
  clean because its author published a new version between the two scans.
  With every new rule on, before the fix wave, the same scan gave 215, 437
  and 1,254. No rule raised an error on any environment.
* 26 of the new rule ids have no Hub hit, so no reader has judged them out
  of sample. 18 of those are on by default on the strength of their authors'
  checks on the Harbor task sets alone.
* `.patch`, `.diff` and `.dfy` files are now read as text. 1,079 tests pass.

## 0.4.0

Six judge and grader rules (55 in total), four fixes to existing rules, and
rules that are off by default.

* New rules, all at `review`: `cot-reaches-reward`, `judge-config-hygiene`,
  `label-free-reward`, `judge-same-family-or-policy`, `kernel-perf-grader`,
  `spec-oracle-gap`. They come from rows 15, 18 to 21 and 23 of the
  literature review's product table. Parts of three rows were left out because
  no rollout that gains or loses reward could be named (`the internal notes`).
* Two rules are off by default. `spec-oracle-gap`, run once, unchanged, on
  Terminal-Bench 2.0, found 0 of the 19 tasks with a published instruction
  and test mismatch label and flagged 2 other tasks, both wrong.
  `judge-config-hygiene` was 0 of 4 in the independent check. Turn a rule on
  with `--enable RULE` or `ENVSCAN_ENABLE`. `--rules` marks rules that are
  off.
* `grader-fails-open`: follows a judge fallback that is logged as a metric
  into a paid reward that reads it from `state`. Skips a pairwise or
  multiple-choice fallback at or below chance, and a fallback below the lowest
  reward any valid answer gets.
* `lenient-answer-match`: skips `x in completion` when `completion` is the
  raw chat message list, where a string can never match.
* Labeled set unchanged (one new `review` finding on an env already labeled
  positive). Hub verdicts: 216 likely, 324 review, 1,366 clean. With every
  new rule on, they fire on 11 Hub envs. All 13 hits confirmed by verify runs
  still fire. The published-label scores on the four Harbor datasets are
  unchanged. 324 tests pass. Scan time grew about 5%.
* Independent check of all 11 Hub hits: `cot-reaches-reward` 3 of 5 real,
  `judge-same-family-or-policy` 1 of 1 set up as described,
  `kernel-perf-grader` 0 of 1 (the timed path is never paid),
  `judge-config-hygiene` 0 of 4. `label-free-reward` has never fired
  (`the internal notes`).

## 0.3.1

Fixes from the independent check of the 0.3.0 Hub hits.

* `reward-trusts-agent-claim`: the policy's reply stays a claim only through
  parse steps (string methods, regex, `json.loads`, `parse*` and local
  helpers that only parse, including recursive text flatteners). A value
  that passed through any other call, such as an HTTP classifier or the env's
  own analyzer, is treated as computed. This removes two
  misreads on Hub environments.
* All four reward rules skip `.py` files that the wheel build config leaves
  out (hatch `packages`, `only-include` and `include`, setuptools `packages`
  and `py-modules`). This removes a stale `src_orig/`
  copy in one Hub package.
* `aggregation-without-gate`: the message now says what the rule saw (an
  ungated term that divides by a length, turn, token or cost count) and flags
  weights that were set by a variable and read as verifiers' default 1.0.
* The 4 real hits, all confirmed by verify runs, still fire. Hub verdicts: 216
  likely, 323 review, 1,367 clean. 252 tests pass. Calibration unchanged.

## 0.3.0

Four new rules (49 in total), dataset-wide notes, and the first test on
public benchmark labels.

* Reward wiring rules, all at `review`: `no-held-out-tests`,
  `reward-trusts-agent-claim`, `aggregation-without-gate`,
  `grader-mutates-live-env`. They add no finding on the 90 labeled envs and
  fire on 8 of 1,906 Hub envs.
* An independent read of all 8 Hub hits: `reward-trusts-agent-claim` 4 of 7
  can really raise reward, 2 are misreads and 1 is in a file the package does
  not ship. The one `aggregation-without-gate` hit pays as flagged but is a
  hack the author planted on purpose, and the rule's message gets the
  mechanism wrong. Known gaps for the next batch: tell a score in an HTTP
  response or from the env's own code apart from a score the policy states,
  scan only files the package ships, and fix that message.
* Dataset mode: on 5 or more tasks, a rule that fires on at least 90% of the
  tasks is printed once as a `dataset:` line, and one more line names the
  tasks that have findings of their own.
* First out-of-sample test on published labels (Terminal-Bench 2.0 and 2.1,
  Z.ai's verified fork, SWE-Bench Pro V2). All 89 Terminal-Bench tasks get
  `likely` for two dataset-wide reasons, so the verdict does not rank tasks.
  Task-specific rules reach precision 0.29 and recall 0.22 against published
  hacks, and none of 44 maintainer fixes changes a finding. No rule was tuned
  on these sets. Details in the README.

## 0.2.0

Packaging release plus 13 new rules (45 in total).

Rules:

* Agent image and reward channel: `git-history-or-target-in-image`,
  `answers-or-grader-readable`, `writable-reward-channel`, `verify-time-fetch`,
  `egress-and-shared-state`.
* How tests are run and read: `grader-import-surface`,
  `verdict-parse-semantics`, `existence-or-constant-check`,
  `equality-on-model-objects`.
* Grader logic: `grader-fails-open`, `formal-checker-escape-hatch`,
  `lenient-answer-match`, and the hygiene note `judge-forgeable-delimiter`.
* All new rules report at `review`, except one `verify-time-fetch` pattern
  that reports at `likely`.
* `harbor-shared-grading`: a `[verifier.environment]` table now counts as a
  separate verifier, and a task with one shared step is still flagged. A plain
  Harbor task (no verifiers environment code around it) reports at `review`,
  because the shared verifier is Harbor's default. Dataset mode prints one
  `dataset:` line for it. Verifiers environments that wrap Harbor keep
  `likely`.
* `lean-no-axiom-check` matches `lake` and `lean` case-sensitively, so
  "Frozen Lake env" no longer triggers it.
* Files named `Dockerfile.<name>` and `Containerfile.<name>` are loaded as
  text.
* An independent read of random Hub hits found `grader-fails-open` and
  `lenient-answer-match` noisy (2 of 8 and 4 of 8 real). Both were tightened
  and a second read of fresh hits gave 4 of 8 and 6 of 8. See the README.
* On the 90 labeled environments no v0.1.2 finding changed. `likely` stays at
  precision 0.97 and recall 0.66. `likely` or `review` is 0.90 and 0.81.

CLI and packaging:

* `--sarif FILE`: SARIF 2.1.0 log for GitHub code scanning (one run, all rules in
  `tool.driver.rules`, one result per hack or broken finding, `error` for
  likely and `warning` for review). `--sarif-base DIR` makes paths relative to
  a repo root.
* `.envscan.toml` (found at the root of a scanned local folder, or `--config`):
  `ignore-rule`, `ignore-path` and `suppress` entries, each with a required
  reason. Suppressed findings leave the verdict and appear under `suppressed`
  in JSON. Configs inside archives and Hub packages are not trusted.
* `--write-baseline FILE` and `--baseline FILE`: match on rule, file and a hash
  of the evidence line text, not the line number.
* Harbor dataset mode: a folder of two or more task folders is scanned task by
  task, one verdict per task plus a summary line; JSON is a list; the exit
  code is the worst task.
* `action.yml` composite GitHub Action (inputs `path`, `fail-on`, `sarif`,
  `upload-sarif`, `config`, `baseline`, `category`) and `.pre-commit-hooks.yaml`.
* README: private repos (local paths and archives never touch the network,
  private Hub envs use `PRIME_API_KEY`), SARIF, config, baselines, datasets.
* JSON results gain `suppressed` (always) and `baselined` (when a baseline is
  used). A bad config or baseline file exits 3.
* No change to what the existing rules see or report: scans of the 90 labeled
  environments give identical findings before and after the packaging changes.

## 0.1.2

Calibrated 32 rule set (see README, "How it was calibrated").
