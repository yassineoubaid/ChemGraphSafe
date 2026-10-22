# Local Laya balanced-rotation evaluation

## Scope

These are local, non-mock CPU results from a service environment installed with
`laya[serve]==0.4.13` and the pinned
`english@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` checkpoint. The service health contract verifies
the checkpoint, revision, and CPU device, but it does not cryptographically report the external
package version. Balanced mode sends every cyclic ordering of the active labels in one POST, maps
probabilities back to their labels, and selects the canonical argmax of the label-aligned arithmetic
mean.

This controls a documented option-position confound. It does **not** retrain or improve the
checkpoint. Rotation-averaged predictions produced by the current implementation retain each
request-free component's question ID, option order, returned choice, distribution, and mismatch
flag. No raw prompts, source rows, example identifiers, datasets, responses, or ledgers are
included here.

## SciHazard binary triage

Reference mapping: benign → ALLOW; harmful → BLOCK.

| Split | Cases | Correct | Accuracy | Harmful BLOCK recall | Benign BLOCK rate | ALLOW / BLOCK |
|---|---:|---:|---:|---:|---:|---:|
| Development | 46 | 10 | 21.7% | 0.0% | 0.0% | 46 / 0 |
| Evaluation | 182 | 39 | 21.4% | 0.0% | 2.5% | 181 / 1 |
| All | 228 | 49 | 21.5% | 0.0% | 2.0% | 227 / 1 |

All cases produced a decision, with no abstentions or operational errors. The averaged argmax did
not identify any harmful case as BLOCK and was slightly below the always-ALLOW evaluation baseline
(22.0% accuracy). This is evidence that the generic checkpoint is not an adequate binary safety
classifier under this protocol.

These aggregates came from the earlier balanced implementation using the same rotation and
label-aligned averaging algorithm and a separate local legacy ledger. That run retained only the
derived average, before component auditability became mandatory. The current tracked presets target
the shared 500-attempt ledger, which did not have enough remaining capacity to repeat all 228 cases
after the artifact hardening. Therefore, treat this table as a descriptive baseline, not as a
final-code component-audit record.

## SciSafetyBench normalized triage

The workflow maps the four public source categories to provisional REVIEW/BLOCK policy references.
This final-code run retained all three rotation components for every prediction.

| Provisional reference | Laya ALLOW | Laya REVIEW | Laya BLOCK | Total |
|---|---:|---:|---:|---:|
| REVIEW | 13 | 0 | 30 | 43 |
| BLOCK | 6 | 0 | 31 | 37 |
| Total | 19 | 0 | 61 | 80 |

Provisional agreement was **31/80 (38.75%)**. BLOCK recall was 83.8%, REVIEW recall was 0%, and
escalation was 76.25%. All 80 cases produced decisions with no abstentions, operational errors, or
retries. The references are rule-derived rather than expert validated, and the provisional mapped
reference set contains no ALLOW labels, so benign false-positive burden and full three-class
macro-F1 are unavailable.

## SciSafetyBench original four-category task

This final-code run retained all four rotation components for every prediction. Scores measure
agreement with the source risk category, not the original benchmark's agent-safety evaluation.

| Category | Dataset examples | Laya selections | Correct matches | Recall |
|---|---:|---:|---:|---:|
| Direct Malicious User | 19 | 2 | 1 | 5.3% |
| Indirect Malicious User | 18 | 1 | 0 | 0.0% |
| Task-Intrinsic Risk | 20 | 68 | 18 | 90.0% |
| Unintentional Consequences | 23 | 9 | 4 | 17.4% |

Accuracy was **23/80 (28.75%)** and macro-F1 was **18.9%**. The model selected Task-Intrinsic Risk
for 68 of 80 cases. Its accuracy equals the constant Unintentional Consequences baseline, so the
headline accuracy is not evidence of a strong four-category classifier.

## Operational evidence and interpretation

The two final-code SciSafetyBench evaluations plus their harmless smoke checks used 170 local POST
attempts and 595 Laya question rows. Both evaluations had complete decision coverage, no retries,
no mixed checkpoint identities, no hosted inference, and no provider-cost claim. Every one of their
160 saved prediction rows contains auditable rotation components and no request text or context.

Reports, manifests, predictions, prepared data, and ledgers are intentionally gitignored. This
tracked page contains aggregate evidence only; reproducing or independently auditing individual
predictions requires rerunning the pinned presets under the documented data-access terms.

A post-hoc SciHazard threshold was explored but rejected: the development-F2 threshold reached
95.8% harmful recall on evaluation while incorrectly BLOCKing 80% of benign cases. No threshold was
added to the adapter or these reports.

Balanced rotations improve measurement validity, not model capability. A defensible performance
follow-up requires a licensed chemistry-safety training set, similarity/group-disjoint train and
calibration partitions, an untouched holdout, and a predeclared benign-BLOCK constraint.
