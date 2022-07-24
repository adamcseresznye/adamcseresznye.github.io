Quarto Computations
================

## NumPy

``` python
import numpy as np
a = np.arange(15).reshape(3, 5)
a
```

    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])

## Matplotlib

``` python
import matplotlib.pyplot as plt

fig = plt.figure()
x = np.arange(10)
y = 2.5 * np.sin(x / 20 * np.pi)
yerr = np.linspace(0.05, 0.2, 10)

plt.errorbar(x, y + 3, yerr=yerr, label='both limits (default)')
plt.errorbar(x, y + 2, yerr=yerr, uplims=True, label='uplims=True')
plt.errorbar(x, y + 1, yerr=yerr, uplims=True, lolims=True,
             label='uplims=True, lolims=True')

upperlimits = [True, False] * 5
lowerlimits = [False, True] * 5
plt.errorbar(x, y, yerr=yerr, uplims=upperlimits, lolims=lowerlimits,
             label='subsets of uplims and lolims')

plt.legend(loc='lower right')
plt.show(fig)
```

![](assets/images/cell-3-output-1.png)

## Plotly

``` python
import plotly.express as px
import plotly.io as pio
gapminder = px.data.gapminder()
gapminder2007 = gapminder.query("year == 2007")
fig = px.scatter(gapminder2007, 
                 x="gdpPercap", y="lifeExp", color="continent", 
                 size="pop", size_max=60,
                 hover_name="country")
fig.show()
```

![](computations_files/figure-gfm/cell-4-output-1.png)

``` python
import plotly.express as px
import plotly.io as pio
gapminder = px.data.gapminder()
def gapminder_plot(year):
    gapminderYear = gapminder.query("year == " + 
                                    str(year))
    fig = px.scatter(gapminderYear, 
                     x="gdpPercap", y="lifeExp",
                     size="pop", size_max=60,
                     hover_name="country")
    fig.show()
    
gapminder_plot(1957)
gapminder_plot(2007)
```

![](computations_files/figure-gfm/cell-5-output-1.png)

![](computations_files/figure-gfm/cell-5-output-2.png)
