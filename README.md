# envscan

Static checks that flag likely reward hacks and broken graders in
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments, the
format used by the Prime Intellect Environments Hub.

envscan reads source. It never imports, installs or runs the environment, so it
is safe to point at untrusted code and cheap enough to run on every publish.
Plain Python 3.11+, standard library only.

## Install and run

```bash
pip install git+https://github.com/ALLENDE123X/envscan   # or, from a clone: pip install .
envscan path/to/my-env           # a local env directory
envscan dist/my-env-0.1.0.tar.gz # a source archive
envscan owner/name               # fetch the latest Hub version in memory and scan it
envscan owner/name@0.2.1         # a specific Hub version
envscan --rules                  # list every rule
envscan my-env --json            # machine readable output
envscan my-env -v                # also show hygiene notes
envscan my-env --sarif out.sarif # also write a SARIF 2.1.0 log for GitHub code scanning
envscan my-env --baseline b.json # only findings that are not in the baseline
envscan tasks/                   # a folder of Harbor tasks: one verdict per task
```

Hub fetches use the same public endpoint as `prime env pull` and stream the
package archive into memory; nothing is written to disk. Public envs need no
key. If `PRIME_API_KEY` is set it is sent, as the prime CLI would.

### Private repos and private envs

* A local path (a folder or a `.tar.gz`) is read from disk and never touches the
  network. envscan reads source only: it does not import, install or run
  anything. This is the right mode for private code, in CI or on a laptop.
* Private Hub envs work with `PRIME_API_KEY` set in the environment. The key is
  sent only to the Hub, the archive is streamed into memory and nothing is
  written to disk.
* Nothing is uploaded anywhere by envscan. The GitHub Action uploads SARIF to
  your own repository's code scanning only if you turn on `upload-sarif`.

## Verdicts

Each finding has a rule id, a kind, a severity hint, a confidence and
`file:line` evidence.

* kind `hack`: a policy can earn reward without solving the task.
* kind `broken`: correct work is scored wrong, or the env cannot load or score.
* kind `hygiene`: a risk worth knowing (judge prompt injection surface, legacy
  API with an unbounded pin). Never counted as a defect.
* confidence `likely`: a pattern that matched hand verified defects with high
  precision during calibration.
* confidence `review`: a real signal that needs a person to confirm it.

The env verdict is `likely` if any hack or broken finding is `likely`,
`review` if there are only `review` findings, else `clean`.

Exit codes: 0 clean, 1 review, 2 likely, 3 scan error (also a bad config or baseline file). With `--fail-on likely`
(or `--fail-on review`) the exit code is 1 only at or above that level, which
is what you want for a publish gate.

## Use it as a publish gate in CI

### GitHub Action

The repo ships a composite action (`action.yml` at the scanner root). It runs
envscan, writes a SARIF log, optionally uploads it to GitHub code scanning
(`github/codeql-action/upload-sarif`), and then fails the job according to
`fail-on`. It needs Python 3.11+ on the runner (`ubuntu-latest` has it) and no
`pip install`.

Inputs: `path` (default `.`), `fail-on` (`likely` default, `review` or `never`),
`sarif` (output path, default `envscan.sarif`), `upload-sarif` (`true` or
`false`, default `false`), plus `config`, `baseline` and `category`. Output:
`exit-code` (0 clean, 1 review, 2 likely, 3 scan error).

Used straight from this repo (pin a tag):

```yaml
name: envscan
on: [push, pull_request]
permissions:
  contents: read
  security-events: write     # only needed with upload-sarif
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ALLENDE123X/envscan@v0.5.0
        with:
          path: environments/my_env
          fail-on: likely
          upload-sarif: "true"
```

Used when vendored (this repo copied into yours as an `envscan/` folder):

```yaml
      - uses: actions/checkout@v4
      - uses: ./envscan
        with:
          path: .
          fail-on: likely
          sarif: envscan.sarif
          upload-sarif: "true"
```

SARIF paths are written relative to the repository root, so code scanning
alerts land on the right file. Findings show as `error` (likely) or `warning`
(review). On pull requests from forks the token cannot write code scanning
results, so leave `upload-sarif` off there and rely on the job status.

### pre-commit

`.pre-commit-hooks.yaml` defines an `envscan` hook (runs `envscan --fail-on
likely .` on the repo root, no filenames):

```yaml
repos:
  - repo: https://github.com/ALLENDE123X/envscan     # the repo that holds .pre-commit-hooks.yaml
    rev: v0.5.0
    hooks:
      - id: envscan
        args: ["--fail-on", "likely", "environments/my_env"]
```

### Plain GitHub Actions without the action

```yaml
name: envscan
on: [push, pull_request]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install git+https://github.com/ALLENDE123X/envscan
      - run: envscan environments/my_env --fail-on likely
      - run: envscan environments/my_env --json > envscan.json
        if: always()
      - uses: actions/upload-artifact@v4
        if: always()
        with: { name: envscan, path: envscan.json }
```

A pre push hook works the same way:

```bash
envscan . --fail-on likely && prime env push
```

To recheck what is already live (for example nightly), scan the Hub id instead
of the directory: `envscan owner/name --fail-on likely`.

## SARIF output

`--sarif FILE` writes one SARIF 2.1.0 log: one run, tool `envscan`, every rule in
`tool.driver.rules` (with a `security-severity` that GitHub shows as the
alert's severity), and one result per `hack` or `broken` finding. Hygiene notes
are not results. Level is `error` for `likely` and `warning` for `review`. The
first evidence entry is the result's location (`file` and start line), the rest
are `relatedLocations`. A finding without file evidence is placed on
`pyproject.toml` or `task.toml` when present. Paths are relative to the scanned
folder; `--sarif-base DIR` makes them relative to `DIR` (the action passes the
workspace). In dataset mode every path starts with the task folder. The log
is written even when the exit code is non zero.

## Config file

`.envscan.toml` at the root of a scanned local folder (or `--config FILE`)
can silence known findings. Every entry needs a `reason`, and an entry with no
reason, an unknown key or an unknown rule id is a scan error (exit 3).

```toml
[[ignore-rule]]                 # drop a rule everywhere
rule = "keyword-reward"
reason = "the metric is a keyword check on purpose"

[[ignore-path]]                 # ignore files; * stays in a folder, ** crosses folders
glob = "tests/fixtures/**"
reason = "fixtures are deliberately broken"

[[suppress]]                    # one rule in one place
rule = "exit-code-verdict"
path = "tests/test.sh"
reason = "reviewed 2026-11, the verifier also writes reward.txt"
```

Evidence entries on ignored paths are dropped from a finding. A finding left
with no evidence is suppressed: it is out of the verdict and out of SARIF, and
JSON output lists it under `"suppressed"` with `suppressed_by` and `reason`.
Text output says how many were suppressed. Hygiene notes are never suppressed.
In dataset mode a path may be written relative to the task or as
`task-name/path`.

For safety the config is only auto-found in a local folder you point envscan
at. A `.envscan.toml` shipped inside a `.tar.gz` or a Hub package is ignored,
so an environment cannot switch off its own findings; pass `--config` to use
one on purpose.

## Baselines

```bash
envscan my-env --write-baseline envscan-baseline.json   # save today's findings, exit 0
envscan my-env --baseline envscan-baseline.json         # report only what is new
```

An entry is the rule, the file and a hash of the evidence line's text (spaces
trimmed and collapsed), not the line number, so edits above a finding do not
bring it back, while changing the flagged line itself does. A finding counts as
baselined when all its evidence entries are. Hygiene notes are not baselined.
The count shows as `baselined` in JSON and as "N in baseline" in text.
The baseline is applied after the config file. Commit the file, and refresh it
deliberately when you accept new findings.

## Harbor datasets

When a target folder has no `task.toml` itself but holds two or more task
folders (each with a `task.toml`) directly inside, envscan scans each task as
its own package and prints one verdict per task plus a summary line:

```
$ envscan tasks/
hello-world: LIKELY (high)  harbor-shared-grading, verify-time-fetch
hello-user: REVIEW (critical)  harbor-shared-grading
separate-verifier-environment: CLEAN
...
dataset: 26 of 32 tasks grade in the agent's container (harbor-shared-grading, Harbor's default). One change fixes all of them: a separate verifier.
32 tasks: 7 likely, 19 review, 6 clean
```

On a dataset of 5 or more tasks, any other rule that fires on at least 90% of
the tasks is also said once, and one more line names the tasks that have
findings of their own. That last line is the one to read first. On
Terminal-Bench 2.0:

```
dataset: 89 of 89 tasks grade in the agent's container (harbor-shared-grading, Harbor's default). One change fixes all of them: a separate verifier.
dataset: 89 of 89 tasks share verify-time-fetch. It comes from the dataset's task template, so fix it once there.
dataset: 14 of 89 tasks have findings of their own beyond the dataset-wide ones: adaptive-rejection-sampler, ... and 6 more
89 tasks: 89 likely, 0 review, 0 clean
```

Harbor's default is a shared verifier: `test.sh` runs after the agent, in the
agent's container. That is true of every task that does not ask for a separate
verifier, so a plain Harbor task gets `harbor-shared-grading` at `review`, and
the dataset line says it once. A task goes to `likely` when something specific
makes the shared box easy to use, for example a `test.sh` that pipes `curl`
into a shell or runs `uvx` at verify time (`verify-time-fetch`). A verifiers
environment that wraps Harbor tasks still gets `likely`, as in v0.1.

`-v` prints the full findings per task. `--json` prints a list with one result
per task. The exit code is the worst task. Config and baseline apply per task
(baseline entries carry the task folder name). A folder with a single task, or
tasks nested deeper than one level, is scanned as one package as before.

## What it checks

Run `envscan --rules` for the full list. In short:

| family | rules |
|---|---|
| Grader runs in the agent's box | `grader-in-agent-sandbox`, `harbor-shared-grading`, `keep-sandbox-for-scoring`, `partial-test-restore`, `answer-key-staged-in-runtime` |
| Exit code is the verdict | `exit-code-verdict` |
| Answer blind rewards | `keyword-reward`, `format-heavy-reward` |
| Answer key within reach | `model-named-files-into-test-dir`, `answer-in-prompt`, `reference-shipped-in-workspace`, `open-fetch-tool`, `verifier-feedback-leak` |
| Model text executed by the grader | `eval-model-text` |
| Reward never wired up | `judge-not-awaited`, `rubric-funcs-ignored`, `rubric-not-a-rubric`, `prompt-keyed-lookup`, `shipped-evals-constant-reward` |
| Load breakers | `removed-verifiers-symbol`, `missing-import`, `file-not-shipped`, `no-load-environment`, `verifiers-fork-dependency`, `unpinned-remote-fetch`, `swallowed-dataset-load` |
| Answer key wrong | `placeholder-answer-key` |
| Lean | `lean-no-axiom-check` |
| Agent image and reward channel (v0.2) | `git-history-or-target-in-image`, `answers-or-grader-readable`, `writable-reward-channel`, `verify-time-fetch`, `egress-and-shared-state` |
| How tests are run and read (v0.2) | `grader-import-surface`, `verdict-parse-semantics`, `existence-or-constant-check`, `equality-on-model-objects` |
| Grader logic (v0.2) | `grader-fails-open`, `formal-checker-escape-hatch`, `lenient-answer-match` |
| Reward wiring (v0.3) | `no-held-out-tests`, `reward-trusts-agent-claim`, `aggregation-without-gate`, `grader-mutates-live-env` |
| Judges and graders (v0.4) | `cot-reaches-reward`, `judge-config-hygiene`, `label-free-reward`, `judge-same-family-or-policy`, `kernel-perf-grader`, and off by default `judge-config-hygiene` and `spec-oracle-gap` |
| Harbor task files and images (v0.5) | `docker-privilege-or-socket`, `verifier-applies-reference-patch`, `shell-or-true-fail-open`, `compose-writable-bind-mounts`, `separate-verifier-reads-undeclared-path`, `verifier-runs-submission-privileged`, and off by default `task-toml-unknown-keys`, `agent-runs-as-root`, `root-ops-on-agent-writable-paths` |
| Answer matching and grader logic (v0.5) | `judge-verdict-substring-match`, `grader-failure-scored-zero`, `set-answer-recall-no-precision`, `numeric-tolerance-too-loose`, `formal-statement-vacuity`, `exact-compare-no-normalization`, `extractor-prompt-format-mismatch`, and off by default `grader-or-chain`, `count-reward-no-dedup`, `tool-call-parser-lenient`, `empty-answer-from-truncation` |
| Judge prompts and rubrics (v0.5) | `rubric-no-veto-on-critical-criteria`, `judge-grades-code-without-execution`, and off by default `judge-prompt-lacks-reference` |
| Test quality (v0.5) | `any-of-n-pass-condition`, `no-pass-to-pass-regression-tests`, and off by default `multi-answer-exact-diff-no-checker` |
| Datasets (v0.5) | `duplicate-prompts-conflicting-gold`, `train-eval-split-overlap`, `question-reference-numeric-mismatch`, `items-incompatible-with-exact-answer`, and off by default `mcq-gold-position-skew`, `eval-harness-template-bug` |
| Tools and environment state (v0.5) | `module-state-shared-across-rollouts`, `state-diff-ignored-fields`, `readonly-sql-guard-false-reject`, and off by default `tool-call-count-or-name-reward`, `env-init-state-mismatch` |
| Model text and artifacts (v0.5) | `verifier-deserializes-agent-artifact`, `shell-injection-from-model-text`, and off by default `solution-in-task-text` |
| Hygiene (v0.5, not counted) | `judge-no-length-control`, `judge-prompt-carries-bias-cues`, `judge-template-static-review`, `tests-call-network-or-clock`, `partial-credit-pass-rate-reward`, `tool-count-above-api-cap`, and off by default `oracle-solution-echoes-answer`, `secrets-in-agent-env`, `rubric-presence-only-no-penalty`, `judge-many-criteria-one-call`, `judge-prompt-missing-guard-instructions`, `unpinned-judge-model-alias`, `rubric-static-review`, `thin-test-suite`, `test-inputs-below-constraint-range`, `tool-contract-conformance`, `observation-truncation-silent`, `prompt-wording-raises-cheating` |
| Hygiene (not counted) | `judge-prompt-undelimited`, `first-match-label-regex`, `legacy-api-unbounded-pin`, `judge-forgeable-delimiter` (v0.2) |

The 13 v0.2 rules come from a literature review of reward hacking and grader
flaws. All of them report at `review`, except one `verify-time-fetch` pattern
(a `curl` to shell installer, `pip install` or `uvx` in a test script that
runs in the agent's box), which reports at `likely`.

The 4 v0.3 rules come from the same review. All four report at `review`.

The 6 v0.4 rules come from the same review and all report at `review`.
Two are off by default because they failed their tests: `spec-oracle-gap`
(see "Out of sample" below) and `judge-config-hygiene` (0 of 4 Hub hits
real). Turn one on with `--enable RULE` or `ENVSCAN_ENABLE=rule1,rule2`.
`envscan --rules` marks rules that are off.

The 57 public v0.5 rule ids come from a wider pass over the literature (2,991 paper
notes, merged into a ranked work list). All report at `review`. After an
independent read of their Hub hits, 26 of them are off by default, and six
v0.5 branches of older rules are off as well. A branch is switched with
`--enable rule-id:variant`, for example `--enable keyword-reward:similarity`.
`envscan --rules` lists both kinds. A table of every
new rule with its default, its Hub hits and what the readers found is kept
with the internal build notes.

Some rules encode knowledge of the `verifiers` library itself, for example
that the v1 Harbor taskset grades in the agent's box unless a task declares a
separate verifier, that `RemoteHybridMathRubric` runs `score.py` inside the
agent's sandbox, and which symbols were removed in which release. Those tables
live at the top of the relevant rules in `envscan/rules.py`.

## How it was calibrated

The 90 hand audited environments were scanned and the verdict was compared
with the final audit labels (positive = medium or worse, or a broken grader):
precision and recall overall, by sample and per rule, plus every miss and
false alarm. The labels and the calibration script are not in this repo yet,
because they name environments whose owners are still being told. Rules were
tuned to keep false alarms low, then checked by hand on Hub envs outside the 90.

v0.5.0 numbers (112 public rules, 28 off by default, 6 branches off):

* On the 90 labeled envs the verdict scores did not move: `likely` precision
  0.97 (35 of 36), recall 0.66, and `likely` or `review` precision 0.90 (43 of
  48), recall 0.81.
* On the Hub (1,906 envs) with the shipped defaults: 215 likely, 369 review, 1,322
  clean. With every new rule and branch on, before any fix, the same scan gave
  215, 437 and 1,254.
* Independent check: that all-on scan had 558 hits that 0.4.0 did not have, on
  368 envs. Five readers who did not write the rules judged 200 of them, at
  most 10 per rule: 94 real, 45 real but harmless, 57 not real, 4 unclear. So
  about half of the new hits were not worth showing. The defaults follow the
  readers. Rules they found wrong, harmless or low value too often are off,
  and nine rules were fixed with the one change the reader asked for.
* After those fixes, on the same judged hits: all 36 that the readers called
  real on the fixed rules still fire, and 13 of the 23 they called not real or
  harmless are gone. This is an in-sample check, because the fixes were
  written from those verdicts. No new reader has judged the fixed rules.
* 26 of the new rule ids never fired on the Hub, so nobody has judged them
  out of sample. 18 of those are on by default, checked only by their authors
  on the Harbor task sets and the labeled envs.
* The clearest new results the readers confirmed: state shared across
  concurrent rollouts so one rollout can take another's score (6 envs), a
  judge verdict read with `"correct" in reply` that pays "incorrect" as a
  pass (2 envs), plain string equality on math answers that scores a right
  answer in another form as 0 (4 envs), and an open web search tool on
  benchmark questions whose answers are on the web (4 envs).

v0.4.0 numbers (55 rules):

* On the 90 labeled envs nothing changed: `likely` precision 0.97 (35 of 36),
  recall 0.66, and `likely` or `review` precision 0.90 (43 of 48), recall 0.81.
  The only new finding is `kernel-perf-grader` on `proximal/frontier-swe`,
  which was already labeled positive.
* On the Hub (1,906 envs): 216 likely, 324 review, 1,366 clean. With every
  new rule on, they fire on 11 envs: `cot-reaches-reward` 5,
  `judge-config-hygiene` 4, `judge-same-family-or-policy` 1 and
  `kernel-perf-grader` 1.
  `label-free-reward` has never fired, so its precision is unknown. The rule
  author's 150-env Hub sample had only 1 hit, so most hits were first read by
  the independent reviewer.
* Independent check: a reviewer who did not write the rules read all 11 Hub
  hits, 9 of them never read by the rule author. `cot-reaches-reward`: 3 of
  5 real (2 code bases, where the raw reply with its reasoning reaches the
  judge and can lift a wrong answer), 1 real but harmless, 1 misread (the
  judge sees only a tool argument). `judge-same-family-or-policy`: 1 of 1 set
  up as described (the judge runs on the policy's own server), but reading
  cannot show that self-preference pays more. `kernel-perf-grader`: 0 of 1,
  the timer has no sync but the timed path is never paid. That is the same
  env as its one labeled-set hit, which the rule author had called real.
  `judge-config-hygiene`: 0 of 4, all misreads, so it is now off by default.
  These counts are tiny. Treat the judge rules' hits as leads to read
  (internal notes, not in this repo).
* Four fixes to existing rules, each from an earlier independent check or
  hand check: `grader-fails-open` now follows a judge fallback into a paid
  reward through `state` (one Hub environment gains a finding),
  and skips a pairwise fallback at or below chance and a fallback below the
  lowest reward any valid answer gets (two Hub environments lose theirs).
  `lenient-answer-match` skips `x in completion` when `completion` is the raw
  chat message list, where it can never match (two packages from one owner
  lose theirs). All 13 hits confirmed by verify runs still fire.
* Scan time grew about 5%.

v0.3.0 numbers (49 rules):

* On the 90 labeled envs nothing changed: `likely` precision 0.97 (35 of 36),
  recall 0.66, and `likely` or `review` precision 0.90 (43 of 48), recall 0.81.
  None of the 4 new rules fires on a labeled env.
* On the Hub (1,906 envs): 216 likely, 325 review, 1,365 clean. The new rules
  fire on 8 envs: `reward-trusts-agent-claim` 7 and `aggregation-without-gate`
  1. `no-held-out-tests` and `grader-mutates-live-env` have never fired on
  real code, so their precision is unknown.
* Independent check: a reviewer who did not write the rules read all 8 hits.
  `reward-trusts-agent-claim`: 4 of 7 can really raise reward. 1 is a correct
  read of a stale file the package does not ship. 2 are misreads: one score
  comes from an external classifier's HTTP response and one from the env's own
  linter, not from the policy's claim. The `aggregation-without-gate` hit pays
  as flagged, but the env's README says the hack is planted on purpose for a
  study, and the rule's message describes the mechanism wrongly. These rules
  were not tuned afterwards, because no unread hits are left to measure a
  change on (internal notes, not in this repo).
* Verify runs (11/24): the 4 real `reward-trusts-agent-claim` hits and the
  `aggregation-without-gate` hit were run in a container through each env's
  own reward code, and all 5 paid the crafted reply
  (internal notes, not in this repo).
* 0.3.1 fixes the three wrong hits from that check. Text read from the reply
  now stays "the policy's claim" only through parse steps (string methods,
  regex, `json.loads`, local helpers that only parse), so a score from an HTTP
  classifier or from the env's own linter no longer counts. The reward rules
  also skip files the wheel does not ship. On the 8 hits: the 4 real ones
  still fire and the 3 wrong ones are gone. Hub verdicts after the fix: 216
  likely, 323 review, 1,367 clean. These fixes were made on hits the
  reviewer had already read, so they are not a new precision measurement.
* On public Harbor datasets with published labels the rules say little about
  single tasks. See "Out of sample: published benchmark labels" below.

v0.2.0 numbers (45 rules):

* On the 90 labeled envs (53 positive), `likely`: precision 0.97 (35 of 36),
  recall 0.66, the same as v0.1.2. `likely` or `review`: precision 0.90
  (43 of 48), recall 0.81. The new rules changed no v0.1.2 finding on these
  envs and added findings on 10 of them, all labeled positive. One former miss is
  now caught at `review`.
* The new rules were also run on Harbor's example tasks and adapters and on
  150 Hub envs outside the 90, and every hit was read by hand: test-runner
  rules 15 of 15 real, `grader-fails-open` 3 of 3, `lenient-answer-match`
  4 of 4, `verify-time-fetch` 24 of 24, `answers-or-grader-readable` 6 of 6,
  `writable-reward-channel` 2 of 2, `git-history-or-target-in-image` 2 of 4
  with 2 unclear, `egress-and-shared-state` 1 of 2 with 1 unclear.
  `formal-checker-escape-hatch` has one hit that no run has confirmed, and
  `existence-or-constant-check` has not fired yet. These are small counts read
  by the rule authors, not an independent test.
* Independent check: a reviewer who did not write the rules read random Hub
  hits from outside every set the rules were tuned or checked on, and judged
  whether each can really raise reward. Round one (8 envs per rule):
  `verdict-parse-semantics` 7 of 8, `lenient-answer-match` 4 of 8,
  `grader-fails-open` 2 of 8. The last two were then tightened, which cut
  their Hub hits from 31 to 23 and from 46 to 15. Round two, on 8 fresh envs
  per rule: `lenient-answer-match` 6 of 8, `grader-fails-open` 4 of 8. In
  both rounds the flagged pattern was really in the code for 7 of 8 hits. The
  misses were fallbacks that pay no more than a valid answer, or checks that
  never fire. With 8 hits per rule these numbers are rough. Treat `review`
  hits from these rules as leads to read, not as findings
  (internal notes, not in this repo).
* Scan time grew about 7% with the new rules.

v0.1.2 numbers:

* On the 90 labeled envs (53 positive), `likely`: precision 0.97 (35 of 36),
  recall 0.66. `likely` or `review`: precision 0.89, recall 0.79. The one
  `likely` false alarm has the same unshipped `config.yaml` as a sibling env
  that the audit did label broken.
  These numbers are in sample: the rules were tuned on these envs.
* Out of sample (20 random `likely` envs outside the 90, judged by reading the
  flagged code): about 15 of 20 were real, so expect roughly 75% precision on
  new envs. Most misses come from `keyword-reward`. Treat `likely` as "worth a
  verify run", not as a confirmed finding.

## Out of sample: published benchmark labels

On 11/24 envscan was run, unchanged, on four public Harbor datasets it was
never tuned on. The results were compared with task labels that papers,
maintainers and issue reports have published The label files and the scoring script are not
in this repo yet.

| dataset | tasks | verdicts | published labels used |
|---|--:|---|---|
| Terminal-Bench 2.0 | 89 | 89 likely | 18 tasks with a published hack |
| Terminal-Bench 2.1 | 89 | 89 likely | 28 tasks fixed since 2.0 |
| Z.ai Terminal-Bench 2.0 Verified | 89 | 89 likely | 16 tasks fixed |
| SWE-Bench Pro V2 | 642 | 642 review | 8 tasks with a published hack |

What this shows:

* The verdict does not rank tasks on these datasets. Every Terminal-Bench task
  gets `likely`, because every `test.sh` fetches or installs tools at grading
  time in the agent's container (`verify-time-fetch`): 82 of 89 pipe the `uv`
  installer from `curl`, and the other 7 run `pip install`. This sits on top of Harbor's
  shared verifier. Both are true of the whole dataset, and they match the
  maintainers' own list of verification-time hacks
  (harbor-framework/terminal-bench issue 2086). Read them as one dataset
  finding. They do not say which task is worse. Dataset mode prints them once.
* Task-specific rules are weak here. On Terminal-Bench 2.0, 14 tasks have a
  finding of their own, and 4 of those are among the 18 tasks with a published
  hack: precision 0.29 against a base rate of 0.20, recall 0.22. Unlabeled
  tasks are not confirmed clean, so 0.29 is a floor, but it is close to
  chance.
* Fixes are invisible. None of the 28 tasks repaired in Terminal-Bench 2.1 and
  none of the 16 repaired in Z.ai's fork changed a single finding. Most of
  those repairs align the instruction with the tests or change resources,
  which a static rule cannot see. The two that removed a leak or a false pass
  were not seen either.
* SWE-Bench Pro V2: `verdict-parse-semantics` fires on 71 tasks through 7
  shared output parsers that fall back to searching for the word PASSED. One
  of the 71 is the task with a proven parser exploit (issue 125). A second
  task with forgeable PASSED lines (issue 133) uses a different parser and is
  missed, and the 6 tasks with weak tests get no finding of their own. The
  dataset-wide `harbor-shared-grading` finding agrees with issue 133's claim
  that the shared `test.sh` leaves the reward file writable during grading.

* v0.4 added `spec-oracle-gap` for instruction and test mismatches: a name the
  hidden tests need that only the oracle solution gives. It was written from
  that definition, frozen, then run once on Terminal-Bench 2.0. It found none
  of the 19 tasks with a published mismatch label and flagged 2 other tasks,
  both wrong (git's own flags, and paths the setup script already creates).
  Most labeled mismatches are not about names. The rule is off by default.
  With it off, every number above is the same under 0.4.0
  (internal notes, not in this repo).
* v0.5 added a branch of `writable-reward-channel` for a test script that
  runs agent-written code in the agent's box and then writes only
  `reward.txt`. Harbor reads `reward.json` first, so a file left there during
  the test run would be the score. The branch fires on 26 Terminal-Bench 2.0
  tasks. With it, 30 tasks have a finding of their own and 7 of those are
  among the 18 with a published hack: precision 0.23, recall 0.39
  (internal notes, not in this repo). An independent read of 20 of the 26
  called 17 real by reading the task files. Nothing was run. The mechanism is
  already public (SWE-bench Pro issue 133, Terminal-Bench issue 2086), so
  this is detection of a known problem. One of the 28 tasks repaired in
  Terminal-Bench 2.1 now changes its findings between the two versions. The
  SWE-Bench Pro V2 numbers did not move.

So on Harbor datasets envscan answers the dataset-wide isolation question and
does little to tell tasks apart. Weak tests, instruction and test mismatches
and hollow solutions need a run: null, forged and shortcut submissions sent
through each task's own verifier.

## Limits

* Static only. It finds patterns, not proofs. A `likely` flag is a strong
  lead; it is not a reproduced exploit. Every public claim should be confirmed
  by running the env's own reward code on crafted inputs.
* Recall is limited for semantic bugs: rewards scored at the wrong moment,
  look ahead data, judge prompts that see the wrong field, verifier side
  feedback leaks, IFEval style format checks. Those need a run.
* Flaws that live only in a container image, a remote Harbor dataset or a
  pinned upstream package are invisible unless the package's own code shows
  them (some library level ones are encoded as rules).

## Hub wide scan

The Hub numbers in this file come from a resume safe driver that fetches each
published package, scans it in memory and writes one line per environment
(findings and metadata, no source content). The driver and its results are not
in this repo, because the results name environments whose owners have not been
told yet. To scan a single Hub environment, see Usage above.
