def S_MSE(S_stud, S_teach, D):
    return float(((S_stud - S_teach) ** 2).sum() / D)