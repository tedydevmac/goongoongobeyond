# Rail Vehicle Condition Monitoring — App

A single Streamlit dashboard covering the three implemented subsystems (Item 3 of the
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
    door.py                     # trained Keras cycle classifier
    shm.py                      # trained Keras damage regression model
  models/
    rail_corrugation/            # put stage1_model.json, stage2_model.json, config.json here
    door/ shm/                   # put each subsystem's trained model artifact(s) here
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
The root `runtime.txt` pins the Cloud runtime to Python 3.12 because the SHM
TensorFlow dependency does not provide wheels for every newer Python release.

1. Push the repository to GitHub, including the `rail_app/models/` artifacts.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with the
   GitHub account that can access this repository.
3. Create an app with:
   - **Repository:** `tedydevmac/goongoongobeyond`
   - **Branch:** `main`
   - **Main file path:** `rail_app/app.py`
   - **Python version:** `3.12` in **Advanced settings**
4. Click **Deploy**. Streamlit Cloud installs the dependencies from
   `rail_app/requirements.txt` and serves the app at a public `streamlit.app`
   URL.

If the existing app was created with another Python version, open its
**Settings**, choose **Python 3.12** under the Python version setting, save,
and rebuild the app. The checked-in `runtime.txt` documents the required
version, but the dashboard setting takes precedence on Community Cloud.

The app does not require secrets or a database. Uploaded files are processed
in memory during the prediction request and are removed when that request
finishes. To publish a newer model or UI version, push to `main`; Streamlit
Community Cloud automatically rebuilds and restarts the app.

For a live demo, open the deployed URL in an incognito window and verify that
the subsystem selector, upload control, prediction button, results table, and
CSV download all work before recording the demo.

## 2.2 Deploy publicly on Google Cloud Run

The repository includes a Dockerfile for Cloud Run. Cloud Run runs the app as
a managed public HTTPS service; no local process or VM needs to stay running.
The container uses Python 3.12 and installs the pinned TensorFlow runtime
required by the Door and SHM models.

Install and authenticate the Google Cloud CLI, then run these commands from
the repository root in PowerShell:

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
gcloud run deploy rail-cdm `
  --source . `
  --region asia-southeast1 `
  --platform managed `
  --allow-unauthenticated `
  --memory 4Gi `
  --cpu 2 `
  --timeout 900 `
  --min 0 `
  --max 3
```

Replace `YOUR_PROJECT_ID` and choose a region close to your users. Cloud Build
builds the root `Dockerfile`, pushes the image, deploys it to Cloud Run, and
prints the public HTTPS service URL. The `--allow-unauthenticated` flag is
required for a publicly accessible demo.

After deployment, verify the service before sharing the URL:

```powershell
$url = gcloud run services describe rail-cdm `
  --region asia-southeast1 `
  --format="value(status.url)"
Invoke-WebRequest "$url/_stcore/health"
```

The health request should return `ok`. Open `$url` in a browser and test one
input file for each model whose artifacts are installed. To publish changes,
run the same `gcloud run deploy` command again; Cloud Run creates a new
revision and keeps the previous revision available for rollback.

If a subsystem's model isn't in place yet, the app shows a warning and
disables the Run button for it rather than crashing — the other subsystems
stay usable.

## 3. Door / SHM implementations

Door loads `models/door/status_model.keras` and
`models/door/status_model_scaling.npz`, segments the stream at 200 ms
timestamp gaps, normalizes its numeric features, and classifies each segment.
SHM loads `models/shm/regression_model.keras`, flattens
the numeric CSV readings, resamples them to the model's required input length,
and returns the model's numeric regression output. TensorFlow is loaded only
when SHM inference is requested, and the model is cached for the session.

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
