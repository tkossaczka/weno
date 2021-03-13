import torch


def compute_error(u, u_ex):
    u_last = u
    u_ex_last = u_ex
    err = torch.mean((u_ex_last - u_last) ** 2)
    return err

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
    return loss

def exact_loss(u, u_ex):
    error = compute_error(u, u_ex)
    loss = 10e1*error # PME boxes
    # loss = 10e4*error # PME Barenblatt
    # loss = error
    return loss

def exact_loss_2d(u, u_ex):
    error = torch.mean((u_ex - u) ** 2)
    loss = 10e4*error
    return loss