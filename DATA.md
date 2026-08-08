# Data, provenance and licensing

## What is in this repo

Text only:

- `evals/ground_truth.json` — hand-written labels keyed on chunk id
- `evals/eval_set.json` — which windows of which videos the eval covers
- `evals/cache/perceptions.json` — model output (descriptions and candidates)
- `evals/results.json`, `evals/variance.json`, `evals/RESULTS.md` — scores
- `data/runs/*/` — per-video analysis output
- `assets/*.png` — screenshots of the review console

## What is deliberately not in this repo

No video, and no extracted frames. The source clips are public safety training
material under standard YouTube licence, which does not grant redistribution. Copying
them into a public repo would be a licensing problem, and shipping the frames would
reproduce the footage a keyframe at a time, which is the same problem wearing a hat.

So the repo ships the *analysis*, which is my own work product, and a script that
rebuilds the footage locally:

```bash
python scripts/fetch_data.py
```

Chunk ids are stable (`{video_id}_{index:04d}`), so labels and model output rejoin the
rebuilt footage exactly.

## Sources

| video id | title | role in the eval |
|---|---|---|
| `E0wdFL4WsGU` | Warehouse Pedestrian Safety - Crossing in Front of a Forklift | staged pedestrian crossing; also the title-card false positive |
| `N4OQFi4UrR4` | Forklift Accidents Caused by Pedestrians Standing Too Close | real CCTV incidents |
| `pLjVBBW_g7c` | Forklift near miss, pedestrian not watching | overhead CCTV near miss |
| `nK6UGe1FwCI` | View from a Forklift Driver's Perspective | operator POV, multiple pedestrians |
| `MqvOjo62BHQ` | Forklift safety, pedestrians | staged hazards plus PPE demonstrations |
| `fz9Q7-7l-40` | Forklift and Pedestrian Safety | long training video; supplies most of the negatives |
| `4p2vg_IbmKY` | Fields of Vision: Pedestrian Safety around Forklifts | older mill footage, mostly interviews |

### Excluded after review

`h5cMg8SdE0s` ("Pedestrians vs Forklifts") was sampled at 12 windows and then dropped:
it is AI-generated CGI, not camera footage. Scoring a perception model on synthetic
renders would not say anything about how it reads a real warehouse, and leaving it in
would have inflated the numbers on the easiest possible imagery. The exclusion is
recorded here rather than quietly applied.

## Labelling method

One annotator (the author). A window is labelled with a hazard class only when the
evidence that class requires — as written in
[`src/foreman/hazards.py`](src/foreman/hazards.py) — is visible in that window's
frames. Two rules did most of the work:

1. **Judge the footage, not the narration.** These are training videos; a presenter
   saying "this is dangerous" over a shot of a parked forklift is not a hazard, and
   on-screen captions are never evidence.
2. **Ambiguous is negative.** A pedestrian near a stationary truck, a raised load at a
   rack face, poor lighting that might be missing PPE — all labelled clean. This
   biases the ground truth toward fewer positives and makes the recall numbers
   conservative rather than flattering.

Windows are also tagged with `scene_type` (`footage`, `mixed`, `slide`, `graphic`,
`talking_head`) so false positives can be attributed to frame content. That tagging is
what surfaced the title-card failure mode.

Labels reflect one person's reading of ambiguous footage. A second annotator would
move the numbers, and the honest summary of the eval is "directional result on one
labelled set", not "benchmark".

## Ethical scope

Public training footage, used for analysis, not redistributed. Nothing here identifies
individuals, and no attempt is made to. Deployed for real, a system like this needs
worker consultation, a retention policy, and a rule that it flags conditions rather
than scoring people — the alert should read "pedestrian in truck path at bay 4", never
"this worker keeps doing X".
