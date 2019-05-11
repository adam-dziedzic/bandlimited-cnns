import datetime
import random
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

font = {'size'   : 20}
matplotlib.rc('font', **font)

static_x = (0,8.3855401,11.87663269,14.5,17.8,21.0941708,24,26,29.26308507,39.203635,49,53.53477122,67.50730502,77.77388775)
static_y = (93.69,93.42,93.12,93,93.06,93.24,93.16,93.2,92.89,92.61,91.95,91.64,89.71,87.97)

energy_x = (0,0.048475708,1.209367931,4.962953113,12.35137482,18.63192462,31.01018447,39.92,50.4,75.84286825)
energy_y = (93.69,93.69,93.12,93.01,92.39,91.32,88.99,87.97,83.84,69.47)

mix_x = (0,20,48.24,71)
mix_y = (93.69,92.73,88.85,81.71)

fig = plt.figure(figsize=(8,6))

plt.plot(static_x, static_y, label='static compression', lw=2, marker='o')
plt.plot(energy_x, energy_y, label='energy based compression', lw=2, marker='s')
plt.plot(mix_x, mix_y, label='energy + static', lw=2, marker='v')

plt.xlabel('Compression ratio (%)')
plt.ylabel('Test accuracy (%)')
plt.grid()
plt.legend(loc='lower left')

#plt.gcf().autofmt_xdate()
#plt.xticks(rotation=0)
plt.show()
fig.savefig("compression-compare-cifar10-font.pdf", bbox_inches='tight')