import datetime
import random
import matplotlib.pyplot as plt

data1 = (1163557.14, 1137578.47, 1139094.66)
times1_raw = ('23:04:17', '23:04:27', '23:04:37')
times1 = [datetime.datetime.strptime(s, '%H:%M:%S') for s in times1_raw]

data2 = (1011000.00, 1011000.00, 1011000.00)
times2_raw = ('23:04:21', '23:04:31', '23:04:41')
times2 = [datetime.datetime.strptime(s, '%H:%M:%S') for s in times2_raw]

fig = plt.figure(figsize=(8,6))

plt.plot(times1, data1, label='data1', lw=2, marker='o')
plt.plot(times2, data2, label='data2', lw=2, marker='s')
plt.xlabel('time in seconds')
plt.ylabel('speed in bps')
plt.grid()
plt.legend(loc='upper right')

plt.gcf().autofmt_xdate()

plt.show()