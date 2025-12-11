import torch
from src.models import Net

def generate_teacher_student_data(D, L, R, R_star, N, Delta_in=0.0, beta=1.0):
    teacher = Net(D, R_star, L, norm=1.0, beta=beta)
    W_teacher = teacher.fc1.weight.detach().cpu().numpy()
    x_train = torch.normal(0, 1, (N, L, D))
    with torch.no_grad():
        y_train = teacher(x_train, delta_in=Delta_in)
    return teacher, W_teacher, x_train, y_train