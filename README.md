# Computer Vision-Based Traffic Load Estimation for Ambulance Route Optimization

An academic prototype that connects local road-scene images, pretrained YOLO vehicle detection, a transparent image-based load proxy, and graph-based route search. It is designed to demonstrate the complete processing chain—not to provide live traffic monitoring, a real geographic route network, ambulance dispatch, or validated travel-time estimates.

> **Interpretation boundary:** vehicle count from a single image is not physical traffic density, speed, queue length, congestion, or travel time. Graph costs are abstract scenario units unless separately calibrated against real travel-time data.

## Implementation status

| Area | Status |
| --- | --- |
| Local image discovery, image validation, and status reporting | Implemented; requires local dataset files for dataset-specific use |
| Annotation adapters | Implemented for BDD100K detection JSON, COCO JSON, Pascal VOC XML, and YOLO TXT with an explicit class-name file; schemas are detected/reported per local files |
| Pretrained Ultralytics YOLO inference | Implemented with runtime model-name inspection and actionable download/path errors; **not executed successfully in this environment** because the pretrained-weight host failed TLS certificate verification |
| Image-based load proxy and density class | Implemented; the continuous `[0, 1]` proxy feeds routing edge costs, and a fixed-threshold Low/Medium/High class (≤5 / 6–15 / ≥16 vehicles) is displayed alongside it. Both parameter sets are documented design choices, not measured thresholds |
| Editable graph, two traffic scenarios, Dijkstra and A* | Implemented; A* uses the admissible zero heuristic because this illustrative graph has no calibrated geometric coordinates |
| Matched-image evaluation and CSV export | Implemented for per-image precision/recall/F1 and supported-class vehicle-count absolute error; only runs when a local supported annotation matches the selected image |
| Fine-tuning, video, dataset-level mAP/comparison, live traffic, real GIS, dispatch | Not implemented / outside the initial baseline |
| BDD100K and IDD data availability in this checkout | **Not available.** No dataset archives or image samples were downloaded or inspected here |

No dataset-specific training, detection accuracy, mAP, dataset sizes, route-improvement percentages, or real-image inference results are claimed.

## Architecture

```text
app.py (Streamlit)
  ├── src/data/          discovery, Pillow image loading, BDD100K/IDD annotation adapters
  ├── src/detection/     lazy Ultralytics YOLO inference, verified name-to-class mapping, OpenCV overlays
  ├── src/traffic/       bounded count-based load proxy + display-only density class (Low/Medium/High)
  ├── src/routing/       validated graph, traffic scenarios, Dijkstra, A*
  ├── src/evaluation/    matched-image, class-aware IoU evaluation
  └── src/exports/       detection/evaluation CSV serialization
```

Local datasets, checkpoints, caches, and generated outputs are excluded from Git. The dashboard does not upload dataset images or annotations to a remote service. The first use of a model name may contact Ultralytics' pretrained-weight host; a local checkpoint can be selected instead.

## Environment setup

Python 3.10 or newer is recommended. The implementation environment used Python 3.11.2.

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_dependencies.sh
```

The script installs `requirements.txt` and checks OpenCV. Ultralytics declares the GUI-enabled `opencv-python` package as a dependency; on headless hosts without `libGL.so.1`, the script removes that wheel and reinstalls the compatible `opencv-python-headless` wheel. If you have the system OpenGL libraries and prefer the GUI-enabled OpenCV package, `python -m pip install -r requirements.txt` is also available.

Run the app and tests:

```bash
streamlit run app.py
python -m pytest -q
python -m compileall -q app.py src tests
```

For a proxied preview/container, bind Streamlit to all interfaces:

```bash
streamlit run app.py --server.address 0.0.0.0
```

The test suite uses temporary fixtures and never downloads datasets or model weights. A smoke test renders the dashboard without starting inference.

## Datasets: local acquisition and layout

The app does not download datasets automatically. Obtain them from the specified Kaggle pages, review their license/usage conditions, extract locally, then set each directory in **Dataset status** (or through the environment variables below):

- [BDD100K Kaggle source](https://www.kaggle.com/datasets/alvaromalfaro/bdd100k)
- [IDD Kaggle source](https://www.kaggle.com/datasets/mitanshuchakrawarty/new-idd-dataset)

### Windows PowerShell (recommended for this repository)

This repository includes `scripts/download_datasets.ps1`. It downloads the selected full Kaggle mirror into the ignored local `data/` directory. It does **not** upload dataset files to GitHub.

```powershell
# From E:\\ambulance-route-optimization
Set-ExecutionPolicy -Scope Process Bypass
.\\scripts\\download_datasets.ps1 -Dataset bdd100k

# Optional, only if you also need IDD and have enough disk space:
.\\scripts\\download_datasets.ps1 -Dataset idd
```

Before running it, review the [BDD100K Kaggle source](https://www.kaggle.com/datasets/alvaromalfaro/bdd100k) and [IDD Kaggle source](https://www.kaggle.com/datasets/mitanshuchakrawarty/new-idd-dataset), and accept their applicable terms. The script installs the Kaggle CLI if needed and expects a Kaggle API token at `%USERPROFILE%\\.kaggle\\kaggle.json` or the `KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables. Never commit that token.

**Storage note:** these commands download complete dataset mirrors, not a small subset. Check free disk space before starting. Given the size of road-scene datasets, do not download both unless you have sufficient storage. If disk space is limited, use a small, explicitly selected evaluation subset prepared separately.

Keep Kaggle credentials outside this repository. Do not commit a Kaggle token, `kaggle.json`, or private credentials. Manual download/extraction is also supported. The app recursively discovers supported images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`) and candidate `.json`, `.jsonl`, `.xml`, and `.txt` annotation files; it does not assume an archive layout or annotation schema.

Default local directories:

```text
data/bdd100k/
data/idd/
```

Optional path overrides:

```bash
export AMBULANCE_BDD100K_DIR=/path/to/bdd100k
export AMBULANCE_IDD_DIR=/path/to/idd
```

The **Dataset status** panel reports the configured path, discovered image and candidate-annotation counts, candidate extensions, recognized parser schemas, image matches, unmatched annotation identifiers, images with malformed labels, unmapped categories, unsupported files, and parser issues. A selected image is decoded/validated before inference; an on-demand button can decode every discovered image and report invalid files (this may take time for full datasets). An absent dataset or absent annotations is reported explicitly; an uploaded or unrelated image is never labeled as BDD100K/IDD.

### Annotation support and matching

Supported structures are parser capabilities, **not confirmation of what the Kaggle mirrors contain**. The actual archive formats must be confirmed by running the local scan after download.

| Reader | Recognized structure | Box conversion |
| --- | --- | --- |
| BDD100K JSON | Per-image `name` + `labels`; labels use `category` and `box2d` with `x1,y1,x2,y2`; JSON arrays, JSONL, and a single per-image object are handled | Stored as xyxy |
| COCO JSON | `images`, `annotations`, optional `categories`; annotation `bbox` is `[x,y,width,height]` | Converted to xyxy |
| Pascal VOC XML | `annotation` / `filename` / `object` / `name` / `bndbox` | Top-left x/y are converted from the VOC 1-based convention |
| YOLO TXT | Per-image normalized `class_id x_center y_center width height` text labels | Converted using the paired image's actual pixel dimensions |

COCO, VOC, and YOLO adapters are also available for repackaged local archives. YOLO numeric IDs are mapped only when a `classes.txt`, `classes.names`, `obj.names`, or `labels.names` file supplies names in ID order. Without such a map, boxes are reported with their raw `class_id:N` label and excluded from four-class evaluation. Malformed records, unsupported schemas, unmatched filenames, duplicate/ambiguous image basenames, invalid boxes, and unmapped categories are not silently treated as valid labels.

Image matching first tries a relative path (including suffix matches for extra archive prefixes), then a unique basename. Ambiguous basenames are reported rather than selected arbitrarily. Matching is local and based on identifiers/path names; there is no camera geolocation or image-to-real-road association.

### Vehicle categories

The app supports canonical `car`, `truck`, `bus`, and `motorcycle` categories. Model class IDs are read from the actual loaded checkpoint's `names` mapping and then mapped by recognized names/aliases; IDs are not assumed to match annotation IDs. BDD-style `motor` maps to `motorcycle`. Unmapped annotation categories are excluded from supported-class evaluation and reported by the scan. The pretrained model's actual mapping cannot be verified until its checkpoint loads.

## Pretrained model and inference

- Default checkpoint name: `yolo11n.pt` (Ultralytics YOLO nano baseline).
- Set `AMBULANCE_YOLO_MODEL` or use the dashboard field to select another model/checkpoint.
- A local `.pt` checkpoint avoids the automatic first-run download; it must expose class names matching one or more supported vehicle classes.
- If the default asset is not cached, Ultralytics attempts to download it on first inference. Download, PyTorch, checkpoint, image, and inference errors are shown with next steps.
- Ultralytics settings are directed to the ignored local `.cache/Ultralytics` directory by default; `YOLO_CONFIG_DIR` may be set explicitly. Checkpoint files and the generated `weights/` folder are Git-ignored and are never committed.
- Detection confidence defaults to 0.25 and is configurable from 0 to 1. Only runtime-verified model class IDs whose names map to the supported vehicle classes are passed to inference.
- Each CSV detection row represents one actual returned box and contains image/model/threshold, class ID/name, confidence, and xyxy coordinates. A zero-detection result exports headers and no invented rows.

## Traffic-load proxy and routing costs

### Image-based load proxy

The default **project design choice** is:

```text
load_fraction = min(supported_vehicle_count / saturation_count, 1)
saturation_count = 20 (configurable in the dashboard)
```

The score is deterministic and bounded to `[0, 1]`; an empty set of detections yields zero. The saturation value is a heuristic parameter, not calibrated against road capacity, camera field-of-view, traffic speed, or travel-time observations. Vehicle count and the proxy must not be described as physical density or measured congestion.

### Illustrative density class (display only)

The same supported-vehicle count is also labeled with fixed illustrative thresholds (`src/traffic/density_estimator.py`):

| Supported-vehicle count | Class |
| --- | --- |
| 0–5 | Low |
| 6–15 | Medium |
| 16 or more | High |

Like the saturation count, these thresholds are prototype design choices, not calibrated density limits. The categorical label is shown next to the proxy in the dashboard for readability only: edge costs and routing always use the continuous `[0, 1]` proxy fraction, and the class never influences the computed route. The two traffic modules are complementary views of one count; there is no second, competing load implementation.

### Edge cost

```text
adjusted_edge_cost = base_edge_cost × (1 + α × edge_traffic_load)
```

The default `α` is 1.0 and can be adjusted from 0 to 3 in the dashboard. Base edge costs are nonnegative abstract units; edge load must be within `[0, 1]`. Both Dijkstra and A* receive the same constructed graph and adjusted costs.

### Editable demo graph and scenarios

The default undirected graph has these illustrative base edges:

| Edge | Base cost |
| --- | ---: |
| Ambulance Base — Junction A | 2.0 |
| Junction A — Emergency Site | 6.0 |
| Ambulance Base — Junction B | 4.0 |
| Junction B — Emergency Site | 5.0 |
| Junction A — Junction B | 2.5 |

Two explicit synthetic scenarios are available: (1) all edges load 0.0; (2) Junction A–Emergency Site load 1.0 with other edges at 0.0. At α=1, the computed baseline route is Ambulance Base → Junction A → Emergency Site (cost 8); in the second scenario that corridor's cost becomes 12 and the computed route changes to Ambulance Base → Junction B → Emergency Site (cost 9). These are deterministic outcomes of the **illustrative graph**, not empirical route improvements. Users may edit edges, base costs, and loads, select endpoints, or manually apply the latest image proxy to one chosen edge. That manual assignment is explicitly not a camera-to-road mapping.

A* uses `h(n)=0`, an admissible heuristic for nonnegative edge costs. Because no defensible geometric lower bound is available, this makes A* Dijkstra-style uniform-cost search; no speedup is claimed. Expanded nodes are counted as each non-stale priority-queue node settled, including a reached destination; a source-equals-destination query counts one.

## Evaluation and exports

Evaluation is offered only when a supported annotation record successfully matches the inference image. For that one image:

- Predictions are filtered at the confidence threshold used for inference.
- Matching is greedy, confidence-descending, class-aware, one-to-one, at configurable IoU (default 0.50).
- The dashboard can compute TP/FP/FN, precision, recall, F1, and absolute error between predicted and ground-truth counts of the four mapped vehicle classes.
- The reported evaluation sample size is 1 image. A single-image result is not dataset-level accuracy or mAP. Dataset-level mAP and a cross-dataset comparison are not implemented in this baseline.
- Missing labels, malformed labels, unsupported category IDs, or unmatched identifiers make the relevant metrics unavailable for that image; they are not replaced by zero or guessed values. A matched image with malformed boxes is explicitly withheld from evaluation rather than treated as an empty ground truth.

CSV downloads:

1. **Detection boxes:** one row per actual predicted box, with a header-only file if no boxes were detected.
2. **Evaluation:** one row per actually evaluated image, with threshold, IoU, TP/FP/FN, precision/recall/F1, mapped-class counts/MAE, matching rule, and limitations.

No training/validation/test experiment is performed. If fine-tuning is added later, splits must be kept separate and baseline weights preserved.

## Tests and verification

```bash
python -m pytest -q
python -m compileall -q app.py src tests
```

The tests cover discovery/missing directories, BDD-style JSON/JSONL and COCO/VOC/YOLO parser fixtures, matching/errors/category mapping, image decoding/full-image validation, proxy bounds/reproducibility and image-proxy-to-edge routing, traffic-adjusted costs, scenario-driven route changes, Dijkstra/A* optimal-cost agreement, unreachable/source-equals-destination cases, CSV schemas/values, and a Streamlit `AppTest` smoke run where supported. `tests/test_cv_to_routing_integration.py` additionally wires a synthetic `DetectionRun` test double through the traffic proxy and density class, the edge-load override, and a single constructed graph where Dijkstra and A* must agree on path, cost, and expanded-node count (with h(n)=0 that equality is expected; measured runtimes are asserted nonnegative, never an A* speedup). Density-classifier boundaries, invalid vehicle counts, and out-of-range/missing/ambiguous edge-load overrides are validated. Test annotations/images are generated temporary fixtures; they are not described as real BDD100K or IDD samples.

## Current environment verification record

At implementation time, the checkout contained no dataset archives, no model weights, and no road-scene image; the Kaggle CLI was also not installed. Python 3.11.2 and pip were available, and PyPI package installation succeeded. The system environment has no `libGL.so.1`, so `scripts/install_dependencies.sh` selects headless OpenCV. Both Kaggle URL probes failed with `URLError: TLS/SSL connection has been closed (EOF)`. Loading the default YOLO model attempted `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt` and failed with `SSLCertVerificationError: unable to get local issuer certificate`. Therefore no actual BDD100K/IDD annotation schema was inspected, no pretrained checkpoint was obtained, no real-image inference was run, and no dataset detection metric was computed. Verification completed locally: **29 pytest tests passed** at that time, Python compilation passed, and the installed stack imported as Streamlit 1.65.0, Ultralytics 8.4.172, OpenCV 4.14.0, Pillow 12.3.0, NumPy 2.4.6, pandas 2.3.3, pytest 8.4.2, and PyTorch 2.14.1 (`torch.cuda.is_available()` was false). The deterministic synthetic graph tests produced baseline path Base → Junction A → Emergency Site (cost 8.0) and congestion path Base → Junction B → Emergency Site (cost 9.0); these are computed demo outputs, not empirical results. The interface, parsers, routing, tests, and setup instructions are ready for local data/checkpoint input; model/data network access remains to be verified in the user's environment.

**Integration verification update (density-class wiring):** after connecting `src/traffic/density_estimator.py` into the package API and dashboard, the suite was re-run in this checkout with `python -m pytest -q` (Python 3.11.2; ultralytics/torch not installed or exercised, matching the suite's lazy-import design): **47 tests passed**, alongside `python -m compileall -q app.py src tests` and a Streamlit `AppTest` render with no exceptions. The 18 added integration tests use synthetic `DetectionRun` test doubles and verify module wiring and deterministic arithmetic only — not detector accuracy, real-image results, or ambulance travel-time claims.

## Troubleshooting

- **Dataset folder missing:** set the correct local path in the Dataset status panel or `AMBULANCE_BDD100K_DIR` / `AMBULANCE_IDD_DIR`, then select “Refresh local file discovery.”
- **No labels / unsupported schema:** inspect the panel's detected extensions, unsupported files, and parser issues. Add an adapter only after inspecting the actual annotation format; do not rename or relabel unknown data to make it appear supported.
- **YOLO download fails:** use a trusted, compatible local Ultralytics `.pt` checkpoint in the model field or repair the host's TLS/network trust configuration. Do not disable TLS verification to fetch weights.
- **`libGL.so.1` / OpenCV import error:** run `./scripts/install_dependencies.sh` in the active Python environment; it installs the headless OpenCV build for this server.
- **No evaluation values:** evaluation requires an annotation file using a supported schema and an unambiguous match for the selected image.
- **No route:** check source/destination and edge connectivity; negative/non-finite costs are rejected because Dijkstra/A* require nonnegative finite edge costs.
