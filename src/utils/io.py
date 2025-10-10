
import os
import pickle
import json
from datetime import datetime
from tqdm import tqdm



def save_results(
    results,
    filename,
    verbose=False,
    timestamp=False,  # Kept for compatibility; not used.
    format="pickle",
    run_name=None,
):
    """
    Save results to a pickle or JSON file inside a timestamped or custom-named folder.

    Parameters
    ----------
    results : dict
        Dictionary containing the results to save.
    filename : str
        Path to the file for saving (base name, without folder).
    verbose : bool, optional
        If True, prints confirmation messages (default: False).
    timestamp : bool, optional
        Ignored. Timestamping is handled by creating a timestamped folder.
    format : str, optional
        Save format: "pickle" (default) or "json".
    run_name : str or None, optional
        Custom name for the results folder. If None, creates a folder named run_YYYYMMDD_HHMMSS.
    """
    try:
        # Determine run folder name
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = run_name if run_name is not None else f"run_{now_str}"

        # Determine base directory from filename and create run folder inside it
        base_dir = os.path.dirname(filename)
        if not base_dir:
            base_dir = "."
        full_run_path = os.path.join(base_dir, run_folder)
        os.makedirs(full_run_path, exist_ok=True)

        # Get base filename without directory and extension
        base_filename = os.path.basename(filename)
        name_only, _ = os.path.splitext(base_filename)

        # Save results in chosen format
        if format == "pickle":
            results_filename = os.path.join(full_run_path, f"{name_only}.pkl")
            with tqdm(total=1, desc="Saving pickle file...") as pbar:
                with open(results_filename, "wb") as f:
                    pickle.dump(results, f)
                pbar.update(1)
        elif format == "json":
            def safe_serialize(obj):
                """
                Try to serialize an object to JSON, otherwise return its string representation.
                """
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, OverflowError):
                    return str(obj)

            safe_results = {}
            with tqdm(total=len(results), desc="Serializing JSON data") as pbar:
                for k, v in results.items():
                    safe_results[k] = safe_serialize(v)
                    pbar.update(1)
            results_filename = os.path.join(full_run_path, f"{name_only}.json")
            with open(results_filename, "w", encoding="utf-8") as f:
                json.dump(safe_results, f, ensure_ascii=False, indent=4)
        else:
            raise ValueError(f"[IO] Unsupported format: {format}")

        if verbose:
            print(f"[IO] Results saved to {results_filename} inside folder {full_run_path}")
        print("[IO] Save completed.")

    except Exception as e:
        print(f"[IO] Error saving results to {filename}: {e}")

