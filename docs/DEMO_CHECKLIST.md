# Monday demo checklist

## Before the demo

- [ ] Use the pinned session branch/repository state and run `git status --short --branch`.
- [ ] Keep BDD100K/IDD archives, images, annotations, `.pt` weights, credentials, and evaluation outputs outside Git. The repository `.gitignore` protects `data/`, `outputs/`, model weights, and Kaggle secrets.
- [ ] Confirm the local BDD100K folder is the already-extracted dataset; do not start another full download.
- [ ] Confirm `streamlit`, `Pillow`, `opencv`, `numpy`, `pandas`, and `ultralytics` can import. Confirm the expected checkpoint is cached or select a trusted local checkpoint. Do not disable TLS verification to fetch model files.
- [ ] Run the test suite and compile check from the README. If a GPU is needed for a larger batch run, confirm it is actually visible before starting.
- [ ] If the BDD100K label scan or weights are unavailable, use the route demo and automated tests as the verified fallback; do not invent detection overlays or dataset metrics.

## Local app launch (Windows PowerShell)

```powershell
Set-Location E:\ambulance-route-optimization
$env:AMBULANCE_BDD100K_DIR = "E:\ambulance-route-optimization\data\bdd100k"
streamlit run app.py
```

A Linux/Colab-style launch is:

```bash
streamlit run app.py --server.address 0.0.0.0
```

## Presentation flow

1. **Overview and limitations** — identify this as an academic prototype: no live traffic feed, GPS road network, dispatch, or validated minute-based ETA.
2. **Dataset status** — show the configured directory and local image/annotation discovery counts. The full parser scan is on demand and can take time across 100,000 files; do not start it during the demo unless it has already been completed and checked.
3. **Detection & traffic proxy** — select a real local road image or upload one. Run YOLO only after the model/checkpoint is known to load. Show the original and actual returned boxes, verified model class IDs, count, and inference time. State that a single frame is not calibrated density.
4. **Explain the load formula** — `min(vehicle_count / saturation_count, 1)`, default saturation count 20. The Low/Medium/High labels (0–5 / 6–15 / 16+) are display-only design thresholds.
5. **Road graph & routing** — show the edge formula `base_cost × (1 + α × edge_load)`. The image proxy can be manually assigned to one edge, but this is only an illustrative scenario input—not an inferred camera-to-road match.
6. **Compare routes** — baseline example: Base → Junction A → Emergency Site, cost 8. With the synthetic Junction A–Emergency Site load at 1 and α=1, that edge cost is 12 and the computed alternate is Base → Junction B → Emergency Site, cost 9. These are abstract graph costs, not minutes or real-world improvements.
7. **Compare algorithms accurately** — both use identical adjusted weights. A* currently has `h(n)=0`, so it is Dijkstra-style uniform-cost search; equal expansions are expected and no speedup is claimed. Runtime readings are machine-dependent.
8. **Evaluation & export** — show one-image metrics only when that exact image has a valid matched annotation, or show a saved subset run's CSV/JSON artifacts with its seed/model/threshold provenance. Label subset size and the historical 100-image pilot separately; neither is full-dataset mAP.
9. **Quality evidence** — run `python -m pytest -q` and `python -m compileall -q app.py src scripts tests`. Explain that synthetic integration tests validate arithmetic and wiring, not detector accuracy.
10. **Close with limitations** — no training, video tracking, live traffic, calibrated road costs, real dispatch, or validated travel-time estimates.

## Optional Colab subset evaluation

1. Verify the clone, branch, runtime and whether the dataset is actually mounted. A Git clone does not contain ignored data:

   ```bash
   !git status --short --branch
   !free -h
   !df -h
   !python scripts/evaluate_bdd100k.py --data-root /content/drive/MyDrive/bdd100k --preflight-only
   ```

2. If the existing dataset is stored in Google Drive, mount that Drive first and pass its extracted BDD100K directory. If it is not present in Colab/Drive, stop and arrange access to the existing local files—do not silently download another 5+ GB copy.
3. Start with `--sample-size 10`, `--seed 42`, `--iou 0.50`, and the desired repeated `--confidence` values. Inspect `summary.json`, `manifest.json`, and `per_image_metrics.csv` before increasing the sample.
4. Outputs are written to a new ignored `outputs/` subdirectory. Copy only the small result artifacts—not raw data, credentials, or model weights—to approved persistent storage if they must survive a runtime reset.
