import numpy as np

def compute_S_from_W(W, R, D):
    return (W.T @ W) / np.sqrt(R * D)  

def S_MSE(W_student, W_teacher, R, R_star, D):
    S_stud   = compute_S_from_W(W_student, R, D)
    S_teach  = compute_S_from_W(W_teacher, R_star, D)
    return float(((S_stud - S_teach) ** 2).sum() / D)