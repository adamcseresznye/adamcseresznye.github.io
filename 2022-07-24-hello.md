Quarto Basics
================

## Check out this Polar Axis

For a demonstration of a line plot on a polar axis, see
[Figure 1](#fig-polar).

``` python
import numpy as np
import matplotlib.pyplot as plt

r = np.arange(0, 2, 0.01)
theta = theta = 4 * np.pi * r
fig, ax = plt.subplots(
  subplot_kw = {'projection': 'polar'} 
)
ax.plot(theta, r)
ax.set_rticks([0.5, 1, 1.5, 2])
ax.grid(True)
plt.show()
```

<figure>
<img src="hello_files/figure-gfm/fig-polar-output-1.png" id="fig-polar"
alt="Figure 1: A cool line plot on a polar axis" />
<figcaption aria-hidden="true">Figure 1: A cool line plot on a polar
axis</figcaption>
</figure>

``` python
# this is just a test
```
