# Rail Vehicle Condition Monitoring — App

A single Streamlit dashboard covering all four subsystems (Item 3 of the
Deliverables): choose a subsystem, review its task and expected input,
upload data file(s), preview the input, monitor inference progress, review
results, and download the exact submission-schema CSV.

## Structure

```
app/
  app.py                       # Streamlit entry point — run this
  requirements.txt
  ml/
    features.py                 # shared feature extraction (Rail Corrugation)
  subsystems/
    __init__.py                 # plugin registry + the contract every subsystem follows
    rail_corrugation.py         # WORKING — wraps the two-stage model
    door.py                     # schema-tolerant segmentation baseline
    acv.py                      # schema-tolerant car-localisation baseline
    shm.py                      # stress-derived damage baseline
  models/
    rail_corrugation/            # put stage1_model.json, stage2_model.json, config.json here
    door/  acv/  shm/            # put each subsystem's trained model artifact(s) here
```

## 1. Setup

```bash
pip install -r requirements.txt
```

Copy Rail Corrugation's trained artifacts (from `train.py`'s `--output_dir`)
into `models/rail_corrugation/`:
```
models/rail_corrugation/
  stage1_model.json
  stage2_model.json
  config.json
```

## 2. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Select a subsystem, upload the relevant
file(s), click **Run prediction**, review the generated results, then
**Download** the resulting CSV — this is exactly the CSV to place in
`predictions.zip` for that subsystem.

## 2.1 Deploy publicly with Streamlit Community Cloud

This repository is configured for Streamlit Community Cloud. The app entrypoint
is `rail_app/app.py`, and its dependency file is `rail_app/requirements.txt`.

1. Push the repository to GitHub, including the `rail_app/models/` artifacts.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with the
   GitHub account that can access this repository.
3. Create an app with:
   - **Repository:** `tedydevmac/goongoongobeyond`
   - **Branch:** `main`
   - **Main file path:** `rail_app/app.py`
4. Click **Deploy**. Streamlit Cloud installs the dependencies from
   `rail_app/requirements.txt` and serves the app at a public `streamlit.app`
   URL.

The app does not require secrets or a database. Uploaded files are processed
in memory during the prediction request and are removed when that request
finishes. To publish a newer model or UI version, push to `main`; Streamlit
Community Cloud automatically rebuilds and restarts the app.

For a live demo, open the deployed URL in an incognito window and verify that
the subsystem selector, upload control, prediction button, results table, and
CSV download all work before recording the demo.

If a subsystem's model isn't in place yet, the app shows a warning and
disables the Run button for it rather than crashing — the other subsystems
stay usable.

## 3. Door / ACV / SHM baselines

The three modules include runnable, deterministic baselines because the
training datasets and trained artifacts are not part of this repository.
`Door` detects contiguous opening/closing activity and flags robust current
outliers, `ACV` ranks cars by cross-car telemetry anomaly, and `SHM` computes
a non-negative turning-point stress damage proxy. Each returns the exact
submission schema and reports progress to the app. Replace the internal
baseline with a trained artifact when the labelled datasets are available;
the public plugin contract and upload flow do not need to change.

Nothing else needs to change — `app.py` and the registry are already
generic over `INPUT_MODE` ("multi_file" vs "single_stream") and read
`OUTPUT_FILENAME`/`FILE_HINT` from each module automatically.

## Notes for the demo video / submission

- The app already produces the exact `file_id, prediction` schema Rail
  Corrugation's submission needs — for the official `rail_predictions.csv`,
  upload all 68 Test files at once (multi-select in the file uploader),
  run, and download.
- Batch feature extraction is single-threaded within the app for
  reliability (no subprocess pool inside Streamlit's rerun model) — expect
  roughly 3-4 minutes for the full 68-file Test set, matching the timing
  from `predict.py`.
- For the demo video, a single- or few-file upload is more practical to
  show end-to-end within the 3-minute limit; the full 68-file run is what
  you'd do once, off-camera or sped up, to generate the actual submission
  CSV.
