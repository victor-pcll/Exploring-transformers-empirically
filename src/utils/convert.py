def convert_numeric_config(config, verbose=False):
    """
    Convertit les valeurs numériques en int ou float dans un dictionnaire de configuration.

    Parameters
    ----------
    config : dict
        Dictionnaire de configuration avec des valeurs sous forme de chaînes.
    verbose : bool, optional
        Si True, affiche le dictionnaire converti (par défaut False).
        
    Returns
    -------
    dict
        Dictionnaire de configuration avec les valeurs converties.
    """

    for k, v in config.items():
        if isinstance(v, str):
            try:
                config[k] = int(v)  # Essayer de convertir en int
            except ValueError:
                try:
                    config[k] = float(v)  # Essayer de convertir en float
                except ValueError:
                    pass  # Laisser tel quel si ce n’est ni int ni float
        elif isinstance(v, dict):
            return convert_numeric_config(v)

    if verbose:
        print(f"[Config] Converted config: {config}")

    return config
