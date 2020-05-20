from define_BS_WENO import WENONetwork
import torch
from torch import optim

train_model=WENONetwork()
#V=train_model.forward()

def monotonicity_loss(x):
    return torch.sum(torch.max(x[:-1]-x[1:], torch.Tensor([0.0])))

#optimizer = optim.SGD(train_model.parameters(), lr=0.001)
optimizer = optim.Adam(train_model.parameters())

for k in range(1500):
    # Forward path
    V_train = train_model.forward()
    # Train model:
    optimizer.zero_grad()  # Clear gradients
    loss = monotonicity_loss(V_train[:,1]) # Calculate loss
    loss.backward()  # Backward pass
    optimizer.step()  # Optimize weights
    print(k, loss)

S,tt = train_model.return_S_tt()
#plt.plot(S, V_train.detach().numpy())
print("number of parameters:", sum(p.numel() for p in train_model.parameters()))
train_model.compare_wenos()
#_,_,_,params = train_model.full_WENO()
#g=train_model.parameters()
#g.__next__()

torch.save(train_model, "model")