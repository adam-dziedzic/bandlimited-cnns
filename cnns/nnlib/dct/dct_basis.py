#!/usr/bin/env python
"""
Plot the DCT basis functions.
"""
__author__ = 'Emil Mikulic <emikulic@gmail.com>'
import Image # sudo apt-get install python-imaging
import numpy as np
import matplotlib.pyplot as plt # sudo apt-get install python-matplotlib
import matplotlib.gridspec as gridspec

def idct(x):
  N = len(x)
  X = np.zeros(N, dtype=float)
  for k in range(N):
    out = np.sqrt(.5) * x[0]
    for n in range(1, N):
      out += x[n] * np.cos(np.pi * n * (k + .5) / N)
    X[k] = out * np.sqrt(2. / N)
  return X

def xy(a):
  """Returns arrays of x,y coords for plotting as bars."""
  x = []
  y = []
  #x = [-.5]
  #y = [0.]
  for idx, val in enumerate(a):
    x.append(idx - .5)
    y.append(val)
    x.append(idx + .5)
    y.append(val)
  #x.append(len(a) - .5)
  #y.append(0)
  return x,y

def main():
  n = 8
  out_fn = 'basis.png'

  plt.rc('font', size=9)
  dpi = 72
  plot_width = 790. / 2
  plot_height = 640.
  fig = plt.figure(figsize=(plot_width/dpi, plot_height/dpi), dpi=dpi)
  fig.patch.set(facecolor = 'white')
  gs = gridspec.GridSpec(n, 1)

  for i in range(n):
    a = np.zeros(n, dtype=float)
    a[i] = 1.
    b = idct(a)

    energy = (b * b).sum()
    assert abs(energy - 1.) < 0.001, \
      "expecting normal vector, but %s^2 = %s" % (repr(b), energy)

    ax = fig.add_subplot(gs[i, 0])
    ax.plot(*xy(b))
    ax.set_xlim(-.6, n - .4)
    ax.set_ylim(-.6, .6)

  fig.tight_layout()
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  figimg = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  Image.fromarray(figimg.reshape((h, w, 3))).save(out_fn)
  print 'wrote', out_fn

if __name__ == '__main__':
  main()

# vim:set ts=2 sw=2 et:
