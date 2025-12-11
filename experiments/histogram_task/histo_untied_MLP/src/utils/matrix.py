import numpy as np

def compute_S_from_W(W, R, D):
    """
    Compute S matrix from weight matrix W.
    Args:
        W: weight matrix of shape (D, R)
        R: rank parameter
        D: input dimension
    Returns:
        S matrix of shape (R, R)
    """
    return (W.T @ W) / np.sqrt(R * D)

def S_MSE(W_student, W_teacher, R, R_star, D):
    """
    Compute the Mean Squared Error between student and teacher S matrices.
     Args:
        W_student: weight matrix of the student (D, R)
        W_teacher: weight matrix of the teacher (D, R_star)
        R: rank parameter for student
        R_star: rank parameter for teacher
        D: input dimension
    Returns:
        MSE value as a float
    """
    S_stud = compute_S_from_W(W_student, R, D)
    S_teach = compute_S_from_W(W_teacher, R_star, D)
    return float(((S_stud - S_teach)**2).sum() / D)