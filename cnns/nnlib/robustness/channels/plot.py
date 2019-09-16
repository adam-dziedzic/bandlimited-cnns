import matplotlib.pyplot as plt
import numpy as np

y = [2,4,6,8,10,12,14,16,18,20]
y2 = [10,11,12,13,14,15,16,17,18,19]
x = np.arange(10)
fig = plt.figure()
plt.subplot(111)
plt.plot(x, y, label='$y = numbers')
plt.plot(x+1, y2, label='$y2 = other numbers')
plt.title('Legend inside')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00), shadow=True, ncol=2)
plt.show()
