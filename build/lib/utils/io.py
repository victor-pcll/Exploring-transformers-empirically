def save_results(results, filename):
    import pickle
    with open(filename, "wb") as f:
        pickle.dump(results, f)
