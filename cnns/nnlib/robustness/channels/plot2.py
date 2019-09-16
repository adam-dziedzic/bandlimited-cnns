import numpy as np
# import pylab as p

import matplotlib.pyplot as p
fig = p.figure()
p.subplot(111)

data=np.array(np.random.rand(1000))
y,binEdges=np.histogram(data,bins=100)
bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
p.plot(bincenters,y,'-')
p.show()