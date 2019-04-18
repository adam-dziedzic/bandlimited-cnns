import torch

x = torch.ones(2, 2, requires_grad=True)
print("x: ", x)
print("x.grad_fn: ", x.grad_fn)

y = torch.zeros(2, 2)
y.requires_grad_(requires_grad=True)
print("y: ", y)

d = x + y + 3

print("y.grad_fn: ", y.grad_fn)

w = d * d
print("w: ", w)
# w = w.detach()

z = w * 3
print("z: ", z)

out = z.mean()
print("z, out: ", z, out)
print("out: ", out)

out.backward(torch.tensor(1.0))
print("out grad: ", out.grad)
print("z grad: ", z.grad)
print("w grad: ", w.grad)
print("y grad: ", y.grad)
print("x grad: ", x.grad)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

saved_weights = [1.0, 2.0, 3.0, 4.0]
loaded_weights = torch.tensor(saved_weights)
weights = loaded_weights + 1
print("weights: ", weights)
weights.requires_grad_(True)
out = weights ** 2
# out = weights.pow(2).sum()
out.backward(torch.tensor([1.0,1,1,1]))
print("weights grad: ", weights.grad)





