import matplotlib.pyplot as plt
import os

# get current file name
cwd = os.getcwd()
current_file_name = cwd.split('/')[-1]
epochs = 5
energy1 = [1, 2, 3, 4, 5]
energy2 = [1, 2, 1, 3, 4]
energy3 = [0, 0, 2, 3, 10]
energies = [(energy1, 0.99), (energy2, 0.90), (energy3, 0.80)]

epochs = [epoch for epoch in range(epochs)]
fig, ax = plt.subplots()
for energy, rate in energies:
    ax.plot(epochs, energy, label=str(rate))

ax.legend()
plt.xticks(epochs)
plt.title('Compare loss for naive, numpy and fft based convolution')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.show()


print(__file__.split(".")[0])
