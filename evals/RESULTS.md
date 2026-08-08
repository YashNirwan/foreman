# Eval results

49 hand-labelled windows, 10 containing a hazard (10 labelled events). Perception raised 53 candidates.

Perception `nvidia/nemotron-nano-12b-v2-vl` | verifier VLM `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | verifier LLM `nvidia/nemotron-3-super-120b-a12b`

| arm | precision | recall | F1 | false-alarm windows |
|---|---|---|---|---|
| `vlm_only` | 0.15 | 0.80 | 0.25 | 30/39 (77%) |
| `vlm_conf_60` | 0.18 | 0.80 | 0.29 | 28/39 (72%) |
| `vlm_conf_80` | 0.18 | 0.80 | 0.30 | 28/39 (72%) |
| `scene_gate` | 0.15 | 0.80 | 0.25 | 30/39 (77%) |
| `llm_verifier` | 0.14 | 0.30 | 0.19 | 12/39 (31%) |
| `vlm_verifier` | 0.35 | 0.60 | 0.44 | 10/39 (26%) |
| `vlm_verifier_nogate` | 0.28 | 0.50 | 0.36 | 12/39 (31%) |

## False positives by frame content

| arm | footage | mixed | slide | graphic | talking_head |
|---|---|---|---|---|---|
| `vlm_only` | 34 | 11 | 0 | 0 | 0 |
| `vlm_conf_60` | 26 | 11 | 0 | 0 | 0 |
| `vlm_conf_80` | 25 | 11 | 0 | 0 | 0 |
| `scene_gate` | 34 | 11 | 0 | 0 | 0 |
| `llm_verifier` | 13 | 5 | 0 | 0 | 0 |
| `vlm_verifier` | 8 | 3 | 0 | 0 | 0 |
| `vlm_verifier_nogate` | 10 | 3 | 0 | 0 | 0 |
