import os
import shutil
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def abline(slope, intercept, style):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, style)

class Simple(torch.nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.fc1 = torch.nn.Linear(2, 120)
        self.fc2 = torch.nn.Linear(120, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)
if os.path.exists('./runs'):
    shutil.rmtree('./runs')
torch.manual_seed(1)
np.random.seed(0)
size = 4000
labeled_size = 500
unlabeled_size = 1500
test_size = 2000
n_round = 2000
x, y = make_classification(n_samples=size, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=5, flip_y=0)
x_labeled = x[:labeled_size]
y_labeled = y[:labeled_size]
x_unlabeled = x[labeled_size:labeled_size+unlabeled_size]
y_unlabeled = y[labeled_size:labeled_size+unlabeled_size]
x_test = x[labeled_size+unlabeled_size:]
y_test = y[labeled_size+unlabeled_size:]
plt.scatter(x_labeled[:, 0], x_labeled[:, 1], marker='o', c=y_labeled, s=25, edgecolor='k')

plt.scatter(x_unlabeled[:, 0], x_unlabeled[:, 1], marker='o', c=y_unlabeled, s=25, edgecolor='r')

x_labeled = torch.from_numpy(x_labeled).float()
y_labeled = torch.from_numpy(y_labeled).reshape(-1, 1).float()
x_unlabeled = torch.from_numpy(x_unlabeled).float()
y_unlabeled = torch.from_numpy(y_unlabeled).reshape(-1, 1).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).reshape(-1, 1).float()

net = Simple()
loss = torch.nn.BCELoss(reduction = 'sum')
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
sum = 0
for i in range(n_round):
    optimizer.zero_grad()
    l = loss(net(x_labeled[i%labeled_size]), y_labeled[i%labeled_size])
    l.backward()
    optimizer.step()
    sum += l.item()
    if i%100 == 0:
        writer.add_scalar('training loss', sum/100, i)
        sum = 0
train_err = 0
for i in range(labeled_size):
    output = net(x_labeled[i])
    y = 1 if y_labeled[i] > 0 else -1
    if (output-0.5)*y < 0:
        train_err += 1
# print (y)
# print (net(x))
# print (loss(net(x_labeled), y_labeled))
# print (net.linear.weight.detach().numpy()[0], net.linear.bias.detach().numpy()[0])
# w0, w1, b = net.linear.weight.detach().numpy()[0][0], net.linear.weight.detach().numpy()[0][1], net.linear.bias.detach().numpy()[0]
# abline(-w0/w1, -b/w1, "--")

x_labeled.requires_grad = True
loss(net(x_labeled), y_labeled).backward()
alpha = 0.1
# print(x_labeled)
# print(x_labeled.grad)
# print(x_labeled + alpha*x_labeled.grad)
adv_x_labeled  = x_labeled + alpha*x_labeled.grad.sign()

# plt.scatter(adv_x_labeled[:, 0].detach().numpy(), adv_x_labeled[:, 1].detach().numpy(), marker='o', c=y_labeled.reshape(-1).detach().numpy(), s=25, edgecolor='g')

eta_unlabeled = net(x_unlabeled)
# print(net(x_unlabeled))
# print(loss(net(x_unlabeled), y_unlabeled))

K = 1
class Simple1(torch.nn.Module):
    def __init__(self):
        super(Simple1, self).__init__()
#         self.fc1 = torch.nn.Linear(2, 120)
#         self.fc2 = torch.nn.Linear(120, 1)
        self.fc2 = torch.nn.Linear(2, 1)

    def forward(self, x, y):
#         print(type(-K*y*self.linear(x)))
#         print(-K*y*self.linear(x))
#         print(torch.exp(-K*y*self.linear(x)))
#         x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return 1 / (1 + torch.exp(K*y*x))


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
for i in range(n_round):
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
#     if l > 0.1:
#         print(i, eta_unlabeled[i%unlabeled_size], net1(adv_x_unlabeled1, 1), net1(adv_x_unlabeled2, -1), l)
    l.backward(retain_graph=True)
    optim.step()
    sum += l.item()

    if i%150 == 0:
        writer.add_scalar('training_adv loss', sum/150, i)
        sum = 0

# w0, w1, b = net.linear.weight.detach().numpy()[0][0], net.linear.weight.detach().numpy()[0][1], net.linear.bias.detach().numpy()[0]
# w0_, w1_, b_ = net1.Linear.weight.detach().numpy()[0][0], net1.Linear.weight.detach().numpy()[0][1], net1.Linear.bias.detach().numpy()[0]
# print(w0, w1, b)
# print(w0_, w1_, b_)
# abline(-w0_/w1_, -b_/w1_, "-.")
# plt.ylim((-4, 4))

net2 = Simple1()

optim = torch.optim.SGD(net2.parameters(), lr = 0.1)
sum = 0
for i in range(n_round):
    x_unlabeled_i = x_unlabeled[i%unlabeled_size]
    x_unlabeled_i.requires_grad = True
    y = 1 if eta_unlabeled[i%unlabeled_size] > 0.5 else 0
    net2.zero_grad()
    net2(x_unlabeled_i, y).backward()
    adv_x_unlabeled1 = x_unlabeled_i + alpha*x_unlabeled_i.grad.sign()
    optim.zero_grad()
    l = net2(adv_x_unlabeled1, y)
    l.backward()
    optim.step()
    sum += l.item()
    if i%150 == 0:
        writer.add_scalar('training_hard_adv loss', sum/150, i)
        sum = 0

net3 = Simple1()

optim = torch.optim.SGD(net3.parameters(), lr = 0.1)
sum = 0
for i in range(n_round):
    x_unlabeled_i = x_unlabeled[i%unlabeled_size]
    x_unlabeled_i.requires_grad = True

    net3.zero_grad()
    net3(x_unlabeled_i, 1).backward()
    adv_x_unlabeled1 = x_unlabeled_i + alpha*x_unlabeled_i.grad.sign()
#     print(x_unlabeled_i)
#     print(x_unlabeled_i.grad)
#     print(adv_x_unlabeled1)
#     print(net1(x_unlabeled_i, 1))
#     print(net1(adv_x_unlabeled1, 1))
    net3.zero_grad()
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
    l = loss(y_unlabeled[i%unlabeled_size], net3(adv_x_unlabeled1, 1), net3(adv_x_unlabeled2, -1))
#     if l > 0.1:
#         print(i, eta_unlabeled[i%unlabeled_size], net1(adv_x_unlabeled1, 1), net1(adv_x_unlabeled2, -1), l)
    l.backward(retain_graph=True)
    optim.step()
    sum += l.item()

    if i%150 == 0:
        writer.add_scalar('training_true_adv loss', sum/150, i)
        sum = 0

w0, w1, b = net3.fc2.weight.detach().numpy()[0][0], net3.fc2.weight.detach().numpy()[0][1], net3.fc2.bias.detach().numpy()[0]
abline(-w0/w1, -b/w1, "-.")

standard_logistic_err = 0
robust_logistic_err = 0
standard_soft_err = 0
robust_soft_err = 0
standard_hard_err = 0
robust_hard_err = 0
standard_true_err = 0
robust_true_err = 0
for i in range(test_size):
    x_test_i = x_test[i]
    y_test_i = y_test[i]
    x_test_i.requires_grad = True
    y = 1 if y_test_i > 0 else -1
    net.zero_grad()
    l = torch.nn.BCELoss(reduction = 'sum')(net(x_test_i), y_test_i)
    l.backward()
    adv_x_test = x_test_i + alpha*x_test_i.grad.sign()
    if (net(x_test_i)-0.5)*y < 0:
        standard_logistic_err += 1
    if (net(adv_x_test)-0.5)*y < 0:
        robust_logistic_err += 1
    x_test_i.grad.zero_()
    net1.zero_grad()
    net1(x_test_i, y).backward()
    adv_x_test = x_test_i + alpha*x_test_i.grad.sign()
    if net1(x_test_i, y) > 0.5:
        standard_soft_err += 1
    if net1(adv_x_test, y) > 0.5:
        robust_soft_err += 1
    x_test_i.grad.zero_()
    net2.zero_grad()
    net2(x_test_i, y).backward()
    adv_x_test = x_test_i + alpha*x_test_i.grad.sign()
    if net2(x_test_i, y) > 0.5:
        standard_hard_err += 1
    if net2(adv_x_test, y) > 0.5:
        robust_hard_err += 1

    x_test_i.grad.zero_()
    net3.zero_grad()
    net3(x_test_i, y).backward()
    adv_x_test = x_test_i + alpha*x_test_i.grad.sign()
    if net3(x_test_i, y) > 0.5:
        standard_true_err += 1
    if net3(adv_x_test, y) > 0.5:
        robust_true_err += 1

print (standard_logistic_err/test_size)
print (robust_logistic_err/test_size)
print (standard_soft_err/test_size)
print (robust_soft_err/test_size)
print (standard_hard_err/test_size)
print (robust_hard_err/test_size)
print (standard_true_err/test_size)
print (robust_true_err/test_size)

plt.show()
writer.close()