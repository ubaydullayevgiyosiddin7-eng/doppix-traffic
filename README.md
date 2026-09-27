# Do'ppiX — Traffic Event Detection & Accident Anticipation

WIUT Hackathon 2026 · Computer Vision track (Toyota) · team **Do'ppiX**

A fixed road camera → time segments of traffic events (**Part A**) and a causal
per-frame accident-risk score (**Part B**).

```
video ──► YOLO26m (fine-tuned, 8 classes) ──► ByteTrack ──► tracks ─┬─► scene rules ──► events   (Part A)
          ~10 fps, 1920 px frame, imgsz 1280                        │   (configs/scene.json,
                                                                    │    auto-aligned per video)
frame stream ──► same detector+tracker @ ~5 fps ──► pairwise TTC ───┴─► calibrated risk ──► P(accident ≤ 5 s)  (Part B)
```

## Quick start

```bash
pip install -r requirements.txt
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
```

* Weights are in the repo: `weights/detector.pt` (YOLO26m fine-tune, ~44 MB; val mAP@50 0.978, mAP@50-95 0.885). No internet needed at run time.
* GPU is used automatically when available (fp16), CPU works but is slow.
* `ultralytics` is pinned to the exact version we tested (8.4.163).
* `run_submission.py` and `evaluate.py` are the unmodified organizer files.
* `predictions_samples.json` — our output on the 4 sample videos.

## Repository layout

```
solution.py            CLASSES, detect_events(), RiskEstimator — the organizer interface
run_submission.py      organizer harness (unchanged)
evaluate.py            organizer metric (unchanged)
requirements.txt
weights/detector.pt    fine-tuned YOLO26m
configs/
  scene.json           scene layout drawn once: carriageway, crosswalks, stop line, traffic light, islands
  scene_ref.png        reference background of that frame (for camera-shift alignment)
  bytetrack.yaml       tracker settings
src/doppix/
  config.py            every constant in one place (paths, seed, classes, frame size, stride)
  perception.py        video → per-frame tracked detections (sequential decode, grab() for skipped frames)
  align.py             camera-shift correction: median background ↔ reference, ORB + RANSAC similarity
  scene.py             scene polygons/lines and geometry helpers
  tracks.py            per-object tracks (foot point, speed) and the traffic-light state over time
  rules.py             one function per event class → time intervals; thresholds in the `P` dict
  segments.py          mask → intervals, merging, clipping, same-class overlap removal
  risk.py              Part B: pairwise time-to-collision risk + calibration + causal smoothing
  zones.py             renders the scene zones (used by our labeling tool and the website)
tools/                 development scripts (perception cache, dev evaluation, analyses)
```

## Approach

### What is learned and what is rule-based

| Component | Type | Notes |
|---|---|---|
| Object detector (person, bicycle, car, motorcycle, bus, truck) | **learned** | YOLO26m (COCO-pretrained, Ultralytics) fine-tuned on frames from the sample videos that we labeled ourselves |
| Traffic-light colour (`light_red` / `light_not_red`) | **learned** | same detector, two extra classes; in daylight the lamp colour is hard to see, so we trained it instead of using colour thresholds |
| Tracking | algorithmic | ByteTrack (Ultralytics implementation) |
| Scene layout | manual, once | polygons drawn on one frame in our labeling tool; automatically re-aligned for every video |
| Events (Part A) | **rule-based** | interpretable rules on tracks + scene + light state; thresholds tuned on our own labels of the samples |
| Risk (Part B) | **rule-based** | time-to-closest-approach between road users, calibrated on normal traffic |

The organizers gave only ~18 minutes of footage with no event labels, and several
classes are rare or absent. That is far too little to train an event classifier. So
we put the learning where there is enough data (objects: thousands of boxes) and
wrote the event logic as transparent rules on top of the tracks.

### Perception (`perception.py`)

* 4K 10-bit 4:2:2 H.264 (~147 Mb/s) is decoded **sequentially**. Seeking is slow and inaccurate for long-GOP video, so frames we skip are only `grab()`-ed, never converted.
* Frame sampling: Part A processes every 3rd frame (~10 fps, `PART_A_STRIDE_SEC = 0.1`). Each processed frame is resized to 1920 px and the detector runs at `imgsz = 1280`.
* ByteTrack runs with a low detection threshold (`DET_CONF = 0.1`), so its second association stage can use weak boxes. Tracks are reset per video.
* About 25 frames spread over the video give a median background image, which is used for alignment.

### Camera-shift alignment (`align.py`)

The camera is "fixed", but the sample videos differ by up to ~70 px of shift and
~1° of rotation. The scene is drawn once. For every video we match ORB features
between its median background and `configs/scene_ref.png`, estimate a similarity
transform with RANSAC, and move all scene polygons accordingly. The residual error
is 1–2.5 px. Sanity checks (scale, translation, inlier count) fall back to the
identity transform if the match is unreliable.

### Event rules (`rules.py`)

A vehicle or person position is its box's bottom-centre ("foot") point. Pixel
thresholds are for the 1920-px frame.

| Class | Rule |
|---|---|
| `red_light` | a vehicle's foot crosses the stop line while the governing light is red. The first 1 s after a phase change is ignored because the light state lags. |
| `stop_line` | a vehicle stands still on/over the stop line for ≥ 3 s |
| `stopped_vehicle` | a single vehicle is stationary for ≥ 10 s on the carriageway, outside a queue |
| `congestion` | ≥ 3 stationary vehicles in the intersection approach at once, for ≥ 8 s |
| `jaywalking` | a moving pedestrian on the carriageway for ≥ 1 s who is more than 60 px from any crosswalk and not on an island. Riders on bikes or motorcycles are excluded, as are tiny far-away people (box height < 45 px). Walking across an island does not break a crossing into two events, but walking only on an island is not an event. |
| `failure_to_yield` | a moving vehicle is inside a crosswalk (0.5–8 s) while a pedestrian on that crosswalk is **in its path**: ahead of it along its velocity and laterally within its width |

The traffic light that is visible and faces the camera (light "C", on the right
island) governs the left approach and its stop line. On the samples, 97–98 % of
stop-line crossings happen on not-red (`tools/light_analysis.py`). The light state
is the detector's `light_red`/`light_not_red` inside the light's polygon, then
forward-filled and median-smoothed.

We do not predict `accident`, `near_miss`, `wrong_way`, `illegal_u_turn`,
`illegal_turn`, `solid_line_crossing`, `road_obstacle` or `fire_smoke` as
segments. They do not occur in the samples, and false positives on an absent class
are costly under macro-F1. Part B covers the accident case.

### Accident anticipation (`risk.py`, Part B)

* The detector and tracker run causally on every 6th frame (~5 fps). Between those frames the last score is returned.
* For every pair of road users with ≥ 3 observations in the last 1 s, we extrapolate their centres linearly and compute the time to closest approach `t*` and the miss distance `d*`. The pair is risky if the two are closing at ≥ 150 px/s, `t* ≤ 3 s`, and `d*` is below 60 % of their combined size. The raw pair score is `(1 − t*/3)(1 − d*/reach)`.
* **Calibration:** the 99.9th percentile of the raw score on normal traffic (all sample videos) maps to 0.45. Ordinary near-passes therefore stay below the alarm threshold θ = 0.5, and only unusual approaches exceed it. The score stays continuous, which matters for AP.
* A pair must stay risky for 3 consecutive steps, and the output is EMA-smoothed (α = 0.5). Everything is causal: only past frames are used.
* On the 4 sample videos (no accidents) this gives **0 false alarms**.

## Results on the sample videos

We labeled the samples ourselves in our labeling tool. Our development metric is
the official `evaluate_part_a` run on our labels (`tools/dev_eval.py`).

| Video | Status | Score A |
|---|---|---|
| C3896 (5:40) | fully reviewed | 0.80 (`predictions_samples.json` vs our labels) |

⚠️ These labels were made by reviewing the pipeline's own suggestions
(accept / fix boundaries / reject / add missed), so the score is optimistic. The
remaining videos are still being reviewed. The per-class breakdown is in
`tools/dev_eval.py -v`.

Weakest classes: `failure_to_yield` and `jaywalking` (short, ambiguous events;
most errors come from tracking switches at crowded crosswalks).

## Determinism

* `SEED = 42` (`src/doppix/config.py`) is set for `random`, `numpy` and `torch` before every video.
* The detector runs in inference mode, and ByteTrack is deterministic for identical inputs.
* No test-time augmentation and no sampling. All thresholds are constants in `config.py` / `rules.P` / `risk.py`.

## Runtime

The budget is 3× the video duration for Part A + Part B together. Measured with
the unmodified harness on an RTX 3060 + i9-12900F, all 4 sample videos:

| Video | Length | Part A | Part B | Total | × duration |
|---|---|---|---|---|---|
| C3896 | 340 s | 268 s | 350 s | 617 s | 1.81 |
| C3897 | 318 s | 264 s | 310 s | 575 s | 1.81 |
| C3902 | 318 s | 276 s | 335 s | 612 s | 1.92 |
| C3905 | 128 s | 103 s | 130 s | 233 s | 1.83 |

Most of the time goes to decoding the 4K 10-bit 4:2:2 frames on the CPU. The
harness decodes every frame again for Part B.

**Time-budget guard.** A video over budget scores zero, and the test machine may
be slower than ours, so both parts watch their own speed:

* Part A: every 10 processed frames it projects the finish time. If the projection
  exceeds 1.3× the duration, the frame stride doubles, up to 8×.
* Part B: the deadline is counted from the start of Part A. If the projection
  exceeds 90 % of the 3× budget, the detector stride doubles. As a last resort the
  detector switches off and the last score is returned.

## Datasets and licences

| Data | Use | Licence |
|---|---|---|
| Organizer sample videos (4 × ~5 min) | detector fine-tuning frames (self-labeled), rule tuning | provided for the hackathon |
| COCO (via Ultralytics YOLO26m pretrained weights) | detector initialisation | CC BY 4.0 (annotations); weights AGPL-3.0 (Ultralytics) |

No other footage from this camera was collected, and no external event datasets
were used.

## Tooling we built

* **Labeling tool** (`06_labeler` in our working tree): FastAPI + React.
  * Event segments on a timeline with keyboard shortcuts, and multi-user editing with conflict detection.
  * The pipeline's suggestions can be accepted or rejected in one click.
  * A scene editor for polygons and lines.
  * A bounding-box labeler with model pre-labels and YOLO/COCO export.
* **Website**: sample-video visualisations, EDA, and a live demo (upload an MP4) — https://giyos1212-doppix-traffic.static.hf.space

## Team

| Member | Role |
|---|---|
| Eldor Musayev | Team captain |
| Soibnazar Berdimurodov | Team member |
| Shohruh Gulmirodov | Team member |

Website: https://giyos1212-doppix-traffic.static.hf.space ·
Repository: https://github.com/ubaydullayevgiyosiddin7-eng/doppix-traffic

## Acknowledgements

Ultralytics (YOLO26, ByteTrack implementation) — AGPL-3.0. OpenCV — Apache-2.0.
