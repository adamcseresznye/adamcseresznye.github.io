A Gentle Introduction To Quarto
================
Adam Cseresznye
8/6/23

``` python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
```

# [Handling Images](https://quarto.org/docs/authoring/figures)

## Displaying an image from the hard drive

Resized with the <code>{width=50%}</code> code. Figures can be
referenced later on by including the <code>\#fig-wolf</code> as you can
see in [Figure 1](#fig-wolf). Attributes used <code>{fig-alt=“A drawing
of an elephant.” fig-align=“center” width=50% \#fig-wolf}</code>.

<figure>
<img src="image_example_1.jpg" id="fig-wolf"
data-fig-alt="A drawing of an elephant." data-fig-align="center"
style="width:50.0%" alt="Figure 1: Wolf" />
<figcaption aria-hidden="true">Figure 1: Wolf</figcaption>
</figure>

## Displaying multiple figures as subfigures

Multiple figures can be displayed in subfigures as can be seen at
[Figure 2](#fig-subfigures).

<div>

<table>
<colgroup>
<col style="width: 50%" />
<col style="width: 50%" />
</colgroup>
<tbody>
<tr class="odd">
<td style="text-align: center;"><div width="50.0%"
data-layout-align="center">
<figure>
<img src="image_example_1.jpg" id="fig-surus"
data-ref-parent="fig-subfigures" data-fig.extended="false"
alt="(a) Wolf" />
<figcaption aria-hidden="true">(a) Wolf</figcaption>
</figure>
</div></td>
<td style="text-align: center;"><div width="50.0%"
data-layout-align="center">
<figure>
<img src="image_example_2.jpg" id="fig-hanno"
data-ref-parent="fig-subfigures" data-fig.extended="false"
alt="(b) Winter scenery" />
<figcaption aria-hidden="true">(b) Winter scenery</figcaption>
</figure>
</div></td>
</tr>
</tbody>
</table>

Figure 2: Two random images from [Lorem Picsum](https://picsum.photos/)

</div>

## Displaying figures created by scientific computations

For a practical example demonstrating how to arrange the output of
scientific computations, please refer to **?@fig-bill-marginal** and
**?@fig-line-example** below.

``` python
penguins=pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")
```

``` python
sns.displot(penguins, 
            x = "bill_depth_mm", 
            hue = "species", 
            kind = "kde", fill = True, aspect = 1, height = 4)
plt.show()
sns.displot(penguins, 
            x = "bill_length_mm", 
            hue = "species", 
            kind = "kde", fill = True, aspect = 1, height = 4)
plt.show()
```

<figure>
<img
src="Quarto_example_files/figure-gfm/fig-bill-marginal-output-1.png"
id="fig-bill-marginal-1"
data-fig-alt="Density plot of bill depth by species."
alt="Figure 3: Gentoo penguins tend to have thinner bills," />
<figcaption aria-hidden="true">Figure 3: Gentoo penguins tend to have
thinner bills,</figcaption>
</figure>

<figure>
<img
src="Quarto_example_files/figure-gfm/fig-bill-marginal-output-2.png"
id="fig-bill-marginal-2"
data-fig-alt="Density plot of bill length by species."
alt="Figure 4: Adelie penguins tend to have shorter bills." />
<figcaption aria-hidden="true">Figure 4: Adelie penguins tend to have
shorter bills.</figcaption>
</figure>

Marginal distributions of bill dimensions

``` python
plt.plot([1,23,2,4])
plt.show()

plt.plot([8,65,23,90])
plt.show()
```

<figure>
<img src="Quarto_example_files/figure-gfm/fig-line-example-output-1.png"
id="fig-line-example-1" alt="Figure 5: First" />
<figcaption aria-hidden="true">Figure 5: First</figcaption>
</figure>

<figure>
<img src="Quarto_example_files/figure-gfm/fig-line-example-output-2.png"
id="fig-line-example-2" alt="Figure 6: Second" />
<figcaption aria-hidden="true">Figure 6: Second</figcaption>
</figure>

Charts

# [Tables](https://quarto.org/docs/authoring/tables)

## Create your own table and reference them

For reference see [Table 1](#tbl-example).

<div id="tbl-example">

| Default | Left | Right | Center |
|---------|:-----|------:|:------:|
| 12      | 12   |    12 |   12   |
| 123     | 123  |   123 |  123   |
| 1       | 1    |     1 |   1    |

Table 1: Demonstration of pipe table syntax

</div>

## Display tables as a result of computations

For reference see **?@tbl-penguins-example**.

``` python
penguins.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
      <th>bill_depth_mm</th>
      <th>flipper_length_mm</th>
      <th>body_mass_g</th>
      <th>sex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
      <td>18.7</td>
      <td>181.0</td>
      <td>3750.0</td>
      <td>MALE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
      <td>17.4</td>
      <td>186.0</td>
      <td>3800.0</td>
      <td>FEMALE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
      <td>18.0</td>
      <td>195.0</td>
      <td>3250.0</td>
      <td>FEMALE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
      <td>19.3</td>
      <td>193.0</td>
      <td>3450.0</td>
      <td>FEMALE</td>
    </tr>
  </tbody>
</table>
</div>

# Callouts

<div>

> **Note**
>
> Note that there are five types of callouts, including: `note`,
> `warning`, `important`, `tip`, and `caution`.

</div>

<div>

> **Tip with Title**
>
> This is an example of a callout with a title.

</div>

<div>

> **Expand To Learn About Collapse**
>
> This is an example of a ‘folded’ caution callout that can be expanded
> by the user. You can use `collapse="true"` to collapse it by default
> or `collapse="false"` to make a collapsible callout that is expanded
> by default.

</div>

# Article Layout

## Not specified

``` python
sns.displot(penguins, 
            x = "bill_depth_mm", 
            hue = "species", 
            kind = "kde", fill = True, aspect = 1, height = 4)
plt.show()
sns.displot(penguins, 
            x = "bill_length_mm", 
            hue = "species", 
            kind = "kde", fill = True, aspect = 1, height = 4)
plt.show()
```

<figure>
<img src="Quarto_example_files/figure-gfm/cell-7-output-1.png"
data-fig-alt="Density plot of bill depth by species."
alt="Gentoo penguins tend to have thinner bills," />
<figcaption aria-hidden="true">Gentoo penguins tend to have thinner
bills,</figcaption>
</figure>

<figure>
<img src="Quarto_example_files/figure-gfm/cell-7-output-2.png"
data-fig-alt="Density plot of bill length by species."
alt="Adelie penguins tend to have shorter bills." />
<figcaption aria-hidden="true">Adelie penguins tend to have shorter
bills.</figcaption>
</figure>

Marginal distributions of bill dimensions

## Column set to margin for a figure

``` python
sns.displot(penguins, 
            x = "bill_depth_mm", 
            hue = "species", 
            kind = "kde", fill = True, aspect = 1, height = 4)
plt.show()
sns.displot(penguins, 
            x = "bill_length_mm", 
            hue = "species", 
            kind = "kde", fill = True, aspect = 1, height = 4)
plt.show()
```

<figure>
<img src="Quarto_example_files/figure-gfm/cell-8-output-1.png"
data-fig-alt="Density plot of bill depth by species."
alt="Gentoo penguins tend to have thinner bills," />
<figcaption aria-hidden="true">Gentoo penguins tend to have thinner
bills,</figcaption>
</figure>

<figure>
<img src="Quarto_example_files/figure-gfm/cell-8-output-2.png"
data-fig-alt="Density plot of bill length by species."
alt="Adelie penguins tend to have shorter bills." />
<figcaption aria-hidden="true">Adelie penguins tend to have shorter
bills.</figcaption>
</figure>

Marginal distributions of bill dimensions

## Column set to margin for a table

``` python
penguins.iloc[:5, :3]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
    </tr>
  </tbody>
</table>
</div>

## Multiple Outputs figure to be placed in the margin

``` python
sns.displot(penguins, 
            x = "bill_depth_mm", 
            hue = "species", 
            kind = "kde", fill = True, aspect = 1, height = 4)
plt.show()
penguins.iloc[:5, :3]
```

![](Quarto_example_files/figure-gfm/cell-10-output-1.png)

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>species</th>
      <th>island</th>
      <th>bill_length_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>39.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>40.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adelie</td>
      <td>Torgersen</td>
      <td>36.7</td>
    </tr>
  </tbody>
</table>
</div>
