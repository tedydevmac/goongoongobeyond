"""
Subsystem plugin registry.

Each subsystem module (rail_corrugation.py, door.py, acv.py, shm.py) must
expose this contract:

  NAME              - display name shown in the UI dropdown
  OUTPUT_FILENAME   - exact filename required by the submission schema,
                       e.g. "rail_predictions.csv"
  INPUT_MODE        - "multi_file"    : independent files, batch-predicted
                                         one row per file (Rail, ACV, SHM)
                       "single_stream": ONE continuous unlabeled recording,
                                         segments extracted from it (Door)
  FILE_HINT         - one-line string shown to the user describing what to
                       upload
  MODEL_DIR         - where this subsystem's trained model artifact(s) live
                       (under ../models/<subsystem>/)

  is_ready() -> bool
      Whether model artifacts are present and loadable. The app shows a
      friendly "not configured yet" message instead of crashing when False.

  run(file_paths: list[str], progress_callback=None) -> pandas.DataFrame
      Run inference and return a DataFrame with EXACTLY the column names
      required by that subsystem's submission schema (see
      01_Problem_Statement_3_Specifications.md, Deliverables table):
        Rail Corrugation : file_id, prediction
        ACV              : file_id, ranked_cars
        SHM              : file_id, prediction
        Door             : start_time, end_time, prediction   (no file_id —
                            one row per predicted segment in the stream)
      progress_callback, if given, is a callable taking a float in [0, 1]
      to drive a progress bar.
"""
import importlib

SUBSYSTEM_MODULES = {
    "Rail Corrugation": "subsystems.rail_corrugation",
    "Door": "subsystems.door",
    "ACV": "subsystems.acv",
    "SHM": "subsystems.shm",
}


def get_subsystem(name):
    return importlib.import_module(SUBSYSTEM_MODULES[name])


def list_subsystems():
    return list(SUBSYSTEM_MODULES.keys())
