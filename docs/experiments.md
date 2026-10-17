# Experiment ledger

Every number below is a measured EleutherAI `lm-eval` result. Validation is a 500-row holdout carved from the official GSM8K train split (seed 1234). The test set is the official 1,319-question split. It was scored twice, on two checkpoints, each chosen before its test run. Nothing in this file was selected on the test set.

Protocol unless a row says otherwise: LoRA rank 16, alpha 32, dropout 0, all linear modules, one epoch, warmup 0.05, context 4,096, chat template, thinking off, 5-shot, greedy, max 1,024 generated tokens. Targets are the official worked answers, verbatim, including `<<...>>` and `#### N`.

Score = strict-match accuracy (%) − parameters (B). Official counts include every stored tensor. The text-only count is called out where it was used.

## Submitted checkpoint

| | |
| --- | --- |
| Base | `Qwen/Qwen3.5-4B` (Overmind trained from `unsloth/Qwen3.5-4B`; the weight shards match) |
| Job | `0dc73da5-93dd-4926-8e40-403758df4b3e` |
| Recipe | LoRA r16 α32, LR 2e-5, one epoch, QA wrapper `Question: {q}\nAnswer:` |
| Train rows | Official train minus the 500-row holdout |
| Validation, before the test | 91.4% strict, 91.4% flexible (457/500) |
| Test | **1,204 / 1,319 = 91.2813%** strict, 91.21% flexible |
| Official params | 4,681,098,752 = base 4,659,865,088 + adapter 21,233,664 |
| Official score | **86.60** |
| Text-only params | 4,205,751,296 (vision tower 333,514,240 and MTP head 120,599,552 removed, LoRA merged) |
| Text-only score, the leaderboard entry | **87.07** |
| Training loss | 0.922 → 0.540 over 26 steps. Step-1 loss 0.92, which is a normal base forward pass |

The text-only score uses the same 1,204 generations. It is an accounting change, not a new evaluation. The fp32 merge matches base + adapter. A bf16 merge rounds away most of this particular adapter, because a learning rate of 2e-5 is small next to bf16.

An earlier checkpoint, LoRA at learning rate 1e-4, was the first test run: **1,077 / 1,319 = 81.65%**, official score **76.97**. It was superseded. It was not used to choose the second checkpoint.

## Qwen3.5-4B validation

Morning trainer, step-1 loss about 0.92. Evening trainer, step-1 loss about 0.50. The same nominal learning rate is a larger update in the evening, so the two blocks are not one learning-rate curve.

### Morning

| Run | Val strict | Val flexible | Official score |
| --- | ---: | ---: | ---: |
| LoRA 5e-6 | 91.0 | 90.8 | 86.32 |
| LoRA 2e-5 | 91.4 | 91.4 | 86.72 |
| LoRA 4e-5 | **91.6** | 91.6 | 86.92 |
| LoRA 2e-5, QA format (submitted) | 91.4 | 91.4 | 86.72 |
| LoRA 1e-4 (first run) | 82.6 | 82.6 | 77.92 |
| Full fine-tune, LR 2e-6 | 89.2 | 89.2 | 84.54 |
| Full fine-tune, LR 5e-6 | 82.0 | 82.2 | 77.34 |

LR 4e-5 had the best morning validation and was not tested. The submitted checkpoint was chosen before the test by a rule registered on validation (QA format, LR 2e-5), not by opening the test set and picking the winner.

### Evening, same base, after the step-1 shift

Control is the morning recipe (QA format, LR 2e-5, full pool) retrained on the evening trainer: **87.2 / 87.4** flexible. A shuffled-order copy scored 87.4 (paired z = 0.33 against the control), which is noise.

| Run | Steps | Val strict | Val flexible | vs control |
| --- | ---: | ---: | ---: | --- |
| LR 5e-6, full pool | 26 | **91.0** | 90.8 | +3.8, z = 3.04 |
| LR 1e-5, full pool | 26 | 90.2 | 90.2 | +3.0 |
| Hard examples only (526 misses), LR 5e-6, batch 16 | 33 | 90.6 | 90.4 | +3.4 |
| Full pool plus 4 extra copies of the 526 misses, LR 5e-6, batch 16 | 29 | 90.2 | 90.4 | +3.0, and −4 questions vs the 91.0 run |
| LR 6e-5 | default batch | 80.2 | 80.4 | worse |
| LR 8e-5 | default batch | 78.6 | 78.6 | worse |
| Batch size 4, LR 2e-5 | 82 | 81.2 | 81.4 | −6.0 |
| Batch size 1, LR 2e-5 | 328 | 84.0 | 84.0 | −3.2 |
| Long answers, top 25% by characters, LR 2e-5 | 40 | 84.0 | 84.4 | −3.2 |
| Long answers, LR 5e-5, 3 epochs | 120 | 84.4 | 84.4 | −2.8 |
| Most `<<>>` steps, top 25%, LR 5e-5, 3 epochs | 117 | 81.4 | 81.6 | −5.8 |
| Five-shot packed conversations, LR 2e-5 | 498 | 83.6 | 83.6 | −3.6 |
| Grouped six-turn conversations, LR 2e-5 | 540 | 81.6 | 81.8 | −5.6 |
| Grouped six-turn, LR 5e-5 | 540 | 83.2 | 83.4 | −4.0 |
| Decimal-money rows, LR 2e-5, 1 epoch | 394 | 80.8 | 80.8 | −6.4 |
| Decimal-money rows, LR 2e-5, 2 epochs | 788 | 80.0 | 80.0 | −7.2 |
| Decimal-money rows, LR 4e-5, 1 epoch | 394 | 81.8 | 81.8 | −5.4 |
| Decimal-money rows, LR 4e-5, 2 epochs | 788 | 82.8 | 82.8 | −4.4 |

The 91.0 evening run is the best result from that trainer. It does not beat the morning validation of 91.4, and it was not given a test run. A third test would have needed a validation result clearly above the submitted model. None was.

Hard-example upsampling is the clean comparison: the 91.0 model got 455/500, the upsampled run got 451/500 (+3 / −7). Repeating the misses inside a gentle epoch moved the wrong questions.

One ordering experiment (misses placed first, so they sit at the peak of the cosine) never started. The launch returned HTTP 504 and no job was created.

## Other base models

All of these are 500-question validation after one epoch of LoRA at LR 2e-5, except Qwen3.5-2B, where several learning rates were tried and the best is shown.

| Model | Val strict | Val flexible | Stored params (B) | Val score |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B, LR 2e-5 | 92.0 | 94.6 | 9.682 | 82.32 |
| Qwen3-8B | 83.8 | 88.4 | 8.234 | 75.57 |
| Gemma-4-E4B | 76.2 | 76.6 | 8.032 | 68.17 |
| Gemma-4-E2B | 62.0 | 62.4 | 5.142 | 56.86 |
| Llama-3.2-3B | 80.0 | 83.4 | 3.237 | 76.76 |
| Qwen3.5-2B, best LR 1e-5 | 65.6 | 68.6 | 2.290 | 63.31 |
| Qwen3.5-2B, LR 2e-5 | 63.6 | 63.6 | 2.290 | 61.31 |
| Qwen3.5-2B, LR 8e-5 | 61.2 | 61.4 | 2.290 | 58.91 |
| Qwen3.5-2B, LR 4e-5 | 60.2 | 60.2 | 2.290 | 57.91 |
| Llama-3.2-1B | 52.8 | 55.4 | 1.247 | 51.55 |
| LFM2.5-1.2B | 75.2 | 76.0 | 1.183 | 74.02 |
| Qwen3.5-0.8B | 41.2 | 41.2 | 0.882 | 40.32 |
| LFM2.5-350M | 29.4 | 31.6 | 0.361 | 29.04 |
| LFM2.5-230M | 32.6 | 32.6 | 0.233 | 32.37 |

Qwen3.5-9B is the highest raw accuracy in the project and a worse score than the 4B, because 9.7B costs more points than the extra correct answers return. Twenty of its 500 validation answers have no `####` line, which is why flexible-extract is 94.6 against strict 92.0.

A 20-question smoke of Antares-1B was 55% strict / 75% flexible. That was not promoted to a full validation.

## The tied-Qwen3 loader

Qwen3-4B, Qwen3-1.7B, and Qwen3-0.6B are tied-embedding checkpoints. On the first trainer they started at a step-1 loss of about 6 to 10, with token accuracy near 0.14. A healthy base forward pass on this data is around 0.5 to 1.0, which is what Qwen3.5-4B and the untied Qwen3-8B showed. More epochs cannot repair a broken load: the loss was already wrong before the adapter moved.

Qwen3-4B under that bug scored 66.2% strict and 86.8% flexible on the 500-question holdout. The served model behaved like the untuned base. Raw 5-shot on 50 questions was 88% strict. The published 4-shot chain-of-thought number is 87.79. At 4.022B, even 88% accuracy is a score of about 84, under the submitted 4B.

After the platform fix, a new Qwen3-1.7B job started at step-1 loss **1.56** and token accuracy **0.80**, against 10.34 and 0.14 the night before. The loader was fixed. The run itself was not a good fine-tune: batch size 16 was applied as 16 rows per step, so one epoch took **405 steps** at LR 5e-6. Loss fell from 1.56 to 0.45. Validation was **34.4% strict / 62.4% flexible**. Of 328 misses, 271 had no `####` line. The flexible score is also below the published base (75.44) and below a 50-question raw smoke of the untuned model (70%). The test set was not opened.

## Thinking, under the 1,024-token cap

Two 20-question probes with thinking enabled, on morning checkpoints that score about 91% with thinking off:

| Checkpoint | Strict | Flexible | n |
| --- | ---: | ---: | ---: |
| LR 4e-5 | 55 | 35 | 20 |
| LR 5e-6 | 35 | 30 | 20 |

Most of the failures hit the 1,024-token cap inside the thought, before an answer. Strict-match can also fire on a `####` line inside the thought. Thinking stayed off for every scored run.

## Test-set misses of the submitted model

1,204 correct. The other 115:

| Kind | Count | What it is |
| --- | ---: | --- |
| Wrong number | 84 | A `####` line whose value is not the gold value |
| Gold elsewhere | 15 | The gold integer appears in the trace, and the `####` line is a different number |
| No `####` | 8 | The strict regex does not match |
| `N.00` vs `N` | 8 | `#### 75.00` against gold `#### 75`. Strict-match strips a trailing period, not a trailing `.00` |

The eight decimal misses are real strict-match failures, and they are 0.6 points of accuracy. Every gold `####` line in the training split is an integer. Training on the decimal-money subset, which is the legal way to emphasize that convention, did not fix it: those runs scored 80.0 to 82.8 on validation and raised the wrong-answer count.

Keyword buckets on the wrong answers overlap (time, ratio, money). They are not one bug.

## Final leaderboard

After the deadline on 7 October 2026, 17:00 PDT. The page showed 28 teams. Top 10:

| Rank | Team | Score |
| ---: | --- | ---: |
| 1 | Aman | 89.95 |
| 2 | Cursive | 88.91 |
| 3 | Daniel | 88.09 |
| 4 | Martin | 87.84 |
| 5 | Chinese | 87.34 |
| 6 | Nodolas | 87.20 |
| 7 | Shiven Tripathi | 87.07 |
| 8 | Kartik | 86.67 |
| 9 | Seif | 86.45 |
| 10 | Ayush Nigade | 86.24 |

## What was not allowed, and was not done

Training used the official GSM8K train split only. Inputs were questions. Targets were the official worked answers, copied verbatim. Repeating, reordering, dropping, and wrapping those pairs was in bounds. Paraphrased questions, new word problems, and model-written answers were not used. No tools, retrieval, ensembles, or repeated sampling at evaluation time. No checkpoint that was already fine-tuned on GSM8K.
