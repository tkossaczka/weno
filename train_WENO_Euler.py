from define_WENO_Network import WENONetwork
from define_WENO_Euler import WENONetwork_Euler
import torch
from torch import optim
from define_Euler_system import Euler_system

torch.set_default_dtype(torch.float64)

# TRAIN NETWORK
train_model = WENONetwork_Euler()

# DROP PROBLEM FOR TRAINING
#params = None
problem_class = Euler_system

def monotonicity_loss(u):
    monotonicity = torch.sum(torch.max(u[:-1]-u[1:], torch.Tensor([0.0])))
    loss = monotonicity
    return loss

def exact_loss(u, u_ex):
    error = train_model.compute_error(u, u_ex)
    loss = error
    return loss

#optimizer = optim.SGD(train_model.parameters(), lr=0.1)
optimizer = optim.Adam(train_model.parameters())

for k in range(1):
    # Forward path
    params = None
    problem_main = problem_class(space_steps=64*2, time_steps=None, params=params)
    print(problem_main.time_steps)
    gamma = problem_main.params['gamma']
    q_0, q_1, q_2, lamb, nn = train_model.init_Euler(problem_main, vectorized=True, just_one_time_step=False)
    _, x, t = problem_main.transformation(q_0)
    x_ex = torch.linspace(0, 1, 64*2 + 1)
    p_ex = torch.zeros((x_ex.shape[0], t.shape[0]))
    rho_ex = torch.zeros((x_ex.shape[0], t.shape[0]))
    u_ex = torch.zeros((x_ex.shape[0], t.shape[0]))
    rho_ex[:,0] = q_0
    u_ex[:,0] = q_1/q_0
    p_ex[:,0] = q_2
    q_0_train = q_0
    q_1_train = q_1
    q_2_train = q_2
    for k in range(nn):
        q_0_train, q_1_train, q_2_train, lamb = train_model.forward(problem_main, q_0_train, q_1_train, q_2_train, lamb, k)
        rho = q_0_train
        u = q_1_train / rho
        E = q_2_train
        p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
        p_ex[:, k+1], rho_ex[:, k+1], u_ex[:, k+1], _, _ = problem_main.exact(x_ex, t[k+1])
        # Train model:
        optimizer.zero_grad()  # Clear gradients
        # Calculate loss
        loss_0 = exact_loss(rho, rho_ex[:,k+1])
        loss_1 = exact_loss(u, u_ex[:, k + 1])
        loss_2 = exact_loss(p, p_ex[:, k + 1])
        loss = loss_0 + loss_1 + loss_2
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        print(k, loss)
        q_0_train = q_0_train.detach()
        q_1_train = q_1_train.detach()
        q_2_train = q_2_train.detach()
        lamb = lamb.detach()
        #print(params)

#plt.plot(S, V_train.detach().numpy())
#print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
# g=train_model.parameters()
# g.__next__()

torch.save(train_model, "model4")