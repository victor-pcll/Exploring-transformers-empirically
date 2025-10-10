import os
import pickle
import json
from datetime import datetime
from tqdm import tqdm

def save_results(results, filename, verbose=False, timestamp=False, format="pickle", run_name=None):
    """
    Sauvegarde les résultats dans un fichier pickle ou json à l'intérieur d'un dossier spécifique à l'exécution.
    
    Parameters
    ----------
    results : dict
        Dictionnaire contenant les résultats à sauvegarder.
    filename : str
        Chemin du fichier où sauvegarder les résultats (nom de base, sans dossier).
    verbose : bool, optional
        Si True, affiche un message de confirmation (par défaut False).
    timestamp : bool, optional
        Ignoré, la gestion du timestamp est remplacée par la création d'un dossier horodaté.
    format : str, optional
        Format de sauvegarde, "pickle" (par défaut) ou "json".
    run_name : str or None, optional
        Nom personnalisé du dossier de sauvegarde. Si None, un dossier nommé run_YYYYMMDD_HHMMSS est créé.
    """

    try:
        # Determine run folder name
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name is None:
            run_folder = f"run_{now_str}"
        else:
            run_folder = run_name

        # Determine base directory from filename and create run folder inside it
        base_dir = os.path.dirname(filename)
        if base_dir == "":
            base_dir = "."
        full_run_path = os.path.join(base_dir, run_folder)
        os.makedirs(full_run_path, exist_ok=True)

        # Determine the base filename without directory and extension
        base_filename = os.path.basename(filename)
        name_only, ext = os.path.splitext(base_filename)

        # Construct full path for results file inside run folder
        if format == "pickle":
            results_filename = os.path.join(full_run_path, f"{name_only}.pkl")
            with tqdm(total=1, desc="Saving pickle file...") as pbar:
                with open(results_filename, "wb") as f:
                    pickle.dump(results, f)
                pbar.update(1)
        elif format == "json":
            def safe_serialize(obj):
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
