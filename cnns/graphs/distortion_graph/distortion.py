import datetime
import random
import matplotlib
import matplotlib.pyplot as plt

font = {'size'   : 20}

matplotlib.rc('font', **font)



cd_x = (36.81497106,15.77402479,7.368555653,3.597036099,1.790226158,0.899269972,0.166465718)
cd_acc = (32.8,77.2,87.6,87.4,84.2,74.7,72.6)

fft_x = (1.344878112,1.766248394,2.429652055,3.646892605,4.797592475,6.025413139,7.290727954,11.4227244)
fft_acc = (72.9,80.4,84.5,86.3,85.5,84.5,83.5,79.4)

gauss_x = (0.332492154,0.539711869,7.224593623,16.86231665,24.09024927,48.17626628,72.26406144,96.35248647)
gauss_acc = (56.2,73.4,87.1,84,81.1,58.9,31.1,10.9)

fig = plt.figure(figsize=(8,5))
plt.plot(cd_x[1:], cd_acc[1:], label='CD', lw=2, marker='o')
#plt.plot(fft_x, fft_acc, label='FC', lw=2, marker='s')
#plt.plot(gauss_x[:-3], gauss_acc[:-3], label='Gauss', lw=2, marker='s')

plt.xlabel('L2 Distortion')
plt.ylabel('Recovery (%)')
plt.title("Color Depth Compression")
plt.xlim(0,24)
plt.ylim(60,90)
plt.grid()
#plt.legend(bbox_to_anchor=(0, 0, 1.02, 1.13), ncol=4, fontsize=14)

plt.gcf().autofmt_xdate()

#plt.show()
fig.savefig("imagenet-recovery-1.pdf", bbox_inches='tight')

fig = plt.figure(figsize=(8,5))
#plt.plot(cd_x[1:], cd_acc[1:], label='CD', lw=2, marker='o')
plt.plot(fft_x, fft_acc, label='FC', lw=2, marker='s')
#plt.plot(gauss_x[:-3], gauss_acc[:-3], label='Gauss', lw=2, marker='s')

plt.xlabel('L2 Distortion')
plt.ylabel('Recovery (%)')
plt.title("Frequency Compression")
plt.xlim(0,24)
plt.ylim(60,90)
plt.grid()
#plt.legend(bbox_to_anchor=(0, 0, 1.02, 1.13), ncol=4, fontsize=14)

plt.gcf().autofmt_xdate()

#plt.show()
fig.savefig("imagenet-recovery-2.pdf", bbox_inches='tight')

fig = plt.figure(figsize=(8,5))
#plt.plot(cd_x[1:], cd_acc[1:], label='CD', lw=2, marker='o')
#plt.plot(fft_x, fft_acc, label='FC', lw=2, marker='s')
plt.plot(gauss_x[:-3], gauss_acc[:-3], label='Gauss', lw=2, marker='s')

plt.xlabel('L2 Distortion')
plt.ylabel('Recovery (%)')
plt.title("Gaussian Noise")
plt.xlim(0,24)
plt.ylim(60,90)
plt.grid()
#plt.legend(bbox_to_anchor=(0, 0, 1.02, 1.13), ncol=4, fontsize=14)

plt.gcf().autofmt_xdate()

#plt.show()
fig.savefig("imagenet-recovery-3.pdf", bbox_inches='tight')