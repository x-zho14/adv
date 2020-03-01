import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import shutil

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

class Simple(torch.nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# shutil.rmtree('./runs')
torch.manual_seed(1)
np.random.seed(0)
size = 200
labeled_size = 50
unlabeled_size = 150
x, y = make_classification(n_samples=size, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=2, flip_y=0)
x_labeled = x[:labeled_size]
y_labeled = y[:labeled_size]
x_unlabeled = x[labeled_size:]
y_unlabeled = y[labeled_size:]

plt.scatter(x_labeled[:, 0], x_labeled[:, 1], marker='o', c=y_labeled, s=25, edgecolor='k')

plt.scatter(x_unlabeled[:, 0], x_unlabeled[:, 1], marker='o', c=y_unlabeled, s=25, edgecolor='r')

x_labeled = torch.from_numpy(x_labeled).float()
y_labeled = torch.from_numpy(y_labeled).reshape(-1, 1).float()
x_unlabeled = torch.from_numpy(x_unlabeled).float()
y_unlabeled = torch.from_numpy(y_unlabeled).reshape(-1, 1).float()

net = Simple()
loss = torch.nn.BCELoss(reduction = 'sum')
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
sum = 0
for i in range(5000):
    optimizer.zero_grad()
    l = loss(net(x_labeled[i%labeled_size]), y_labeled[i%labeled_size])
    l.backward()
    optimizer.step()
    sum += l.item()
    if i%100 == 0:
        writer.add_scalar('training loss', sum/100, i)
        sum = 0

# print (y)
# print (net(x))
# print (loss(net(x_labeled), y_labeled))
# print (net.linear.weight.detach().numpy()[0], net.linear.bias.detach().numpy()[0])
w0, w1, b = net.linear.weight.detach().numpy()[0][0], net.linear.weight.detach().numpy()[0][1], net.linear.bias.detach().numpy()[0]
abline(-w0/w1, -b/w1)

x_labeled.requires_grad = True
loss(net(x_labeled), y_labeled).backward()
alpha = 0.1
# print(x_labeled)
# print(x_labeled.grad)
# print(x_labeled + alpha*x_labeled.grad)
adv_x_labeled  = x_labeled + alpha*x_labeled.grad.sign()

plt.scatter(adv_x_labeled[:, 0].detach().numpy(), adv_x_labeled[:, 1].detach().numpy(), marker='o', c=y_labeled.reshape(-1).detach().numpy(), s=25, edgecolor='g')

eta_unlabeled = net(x_unlabeled)
# print(net(x_unlabeled))
# print(loss(net(x_unlabeled), y_unlabeled))

K = 1
class Simple1(torch.nn.Module):
    def __init__(self):
        super(Simple1, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x, y):
#         print(type(-K*y*self.linear(x)))
#         print(-K*y*self.linear(x))
#         print(torch.exp(-K*y*self.linear(x)))
        return 1 / (1 + torch.exp(-K*y*self.linear(x)))

net1 = Simple1()
# print(x_unlabeled[0].requires_grad)
# print(x_unlabeled[0])

# print(type(x_unlabeled.size()))
# x_unlabeled.grad = torch.zeros(x_unlabeled.size())
# print(x_unlabeled.grad)
# print(x_unlabeled[0].grad)

# x_unlabeled_i = x_unlabeled[0]
# x_unlabeled_i.requires_grad = True
# print(x_unlabeled[0].requires_grad)
# print(x_unlabeled_i.requires_grad)

# plt.show()

class CumstomLoss(torch.nn.Module):
    def __init__(self):
        super(CumstomLoss,self).__init__()

    def forward(self, eta, phi1, phi2):
        return eta*phi1 + (1-eta)*phi2

loss = CumstomLoss()
optim = torch.optim.SGD(net1.parameters(), lr = 0.1)
sum = 0
for i in range(5000):
    x_unlabeled_i = x_unlabeled[i%unlabeled_size]
    x_unlabeled_i.requires_grad = True

    net1.zero_grad()
    net1(x_unlabeled_i, 1).backward()
    adv_x_unlabeled1 = x_unlabeled_i + alpha*x_unlabeled_i.grad.sign()
#     print(x_unlabeled_i)
#     print(x_unlabeled_i.grad)
#     print(adv_x_unlabeled1)
#     print(net1(x_unlabeled_i, 1))
#     print(net1(adv_x_unlabeled1, 1))
    net1.zero_grad()
    x_unlabeled_i.grad.zero_()

    net1(x_unlabeled_i, -1).backward()
#     print(x_unlabeled_i.grad)


    adv_x_unlabeled2 = x_unlabeled_i + alpha*x_unlabeled_i.grad.sign()
#     print(x_unlabeled_i)
#     print(x_unlabeled_i.grad)
#     print(adv_x_unlabeled2)
#     print(net1(x_unlabeled_i, 1))
#     print(net1(adv_x_unlabeled2, 1))

    optim.zero_grad()
#     loss = eta_unlabeled[i]*net1(adv_x_unlabeled1, 1) + (1-eta_unlabeled[i])*net1(adv_x_unlabeled2, -1)
    l = loss(eta_unlabeled[i%unlabeled_size], net1(adv_x_unlabeled1, 1), net1(adv_x_unlabeled2, -1))
    if l > 0.1:
        print(i, eta_unlabeled[i%unlabeled_size], net1(adv_x_unlabeled1, 1), net1(adv_x_unlabeled2, -1), l)
    l.backward(retain_graph=True)
    optim.step()
    sum += l.item()
    if i%100 == 0:
        writer.add_scalar('training_adv loss', sum/100, i)
        sum = 0
#     writer.add_scalar('training_adv loss', l.item(), i)

w0, w1, b = net.linear.weight.detach().numpy()[0][0], net.linear.weight.detach().numpy()[0][1], net.linear.bias.detach().numpy()[0]
w0_, w1_, b_ = net1.linear.weight.detach().numpy()[0][0], net1.linear.weight.detach().numpy()[0][1], net1.linear.bias.detach().numpy()[0]
print(w0, w1, b)
print(w0_, w1_, b_)
abline(-w0_/w1_, -b_/w1_)

plt.show()
writer.close()
