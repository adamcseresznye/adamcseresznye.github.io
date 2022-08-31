pymzML for parsing high-resolution mass spectrometry data
================

## Aim:

-   Explore the pymzML module designed for fast parsing of
    high-resolution mass spectrometry data
-   Prepare a 3D chromatogram that could be used for further data
    analysis or machine learning

**About pymzML**

pymzML was first introduced in 2012 in the journal of
[Bioinformatics](https://academic.oup.com/bioinformatics/article/28/7/1052/209917):
“*pymzML—Python module for high-throughput bioinformatics on mass
spectrometry data*”. pymzML is a module that allows an easy access to
mass spectrometry data, especially in mzML format.

pymzML v2.0 [came
out](https://academic.oup.com/bioinformatics/article/34/14/2513/4831092?login=false)
in 2018. In the new version faster libraries were integrated for
numerical calculations, data retrieving algorithms were improved and the
source code was optimized. The result is a library that is on par with
established C programs in terms of processing times, while offering the
versatility of a scripting language.

**About the data**

The example chromatogram used in this notebook was taken from the Center
for Computational Mass Spectrometry’s community resource called
[MassIVE](https://massive.ucsd.edu/ProteoSAFe/static/massive.jsp).
MassIVE is a community resource developed by the NIH-funded Center for
Computational Mass Spectrometry to promote the global, free exchange of
mass spectrometry data.

Our example data can be downloaded from
[here](https://massive.ucsd.edu/ProteoSAFe/dataset_files.jsp?task=8e6bbf633a7e4bc99ddcde7510bffa78#%7B%22table_sort_history%22%3A%22main.collection_asc%22%7D)
with the file name: f.MSV000090214/ccms_peak/Raw/QC_mix_2.mzML.

The file itself is fairly large: approximately 43 MB in size.

With the formalities out of the way, we can proceed to install all
dependencies. To install pymzml please visit the official pymzml
[documentation](https://pymzml.readthedocs.io/en/latest/intro.html#installation).

``` python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = None
import pymzml
```

``` python
path = r"your path here"
file = 'QC_mix_2.mzML'
msrun = pymzml.run.Reader(os.path.join(path, file))
msrun
```

    <pymzml.run.Reader at 0x1b3e01a1e80>

pymzML is fast indeed. %%timeit results in 12.3 ms ± 317 µs per loop
(mean ± std. dev. of 7 runs, 100 loops each).

Using dir() allows us to return a list of valid attributes of the
object:

``` python
# dir(msrun[1])
```

``` python
# example of the kind of attributes we can call

# for idx, n in enumerate(msrun):
#     print(f"Spectrum: {n.ID}, MS level: {n.ms_level} at Rt: {n.scan_time_in_minutes():.2f} min")
#     print(f"Base peak: {n.highest_peaks(1)[0][0]:.3f} with intensity of {n.highest_peaks(1)[0][1]:.0f}\n")
```

In addition, we can also extract all the mass features and their
corresponding intensities in each scan:

``` python
# msrun[1].mz # masses in the first scan
```

``` python
# msrun[1].i # intensities in the first scan 
```

We can also get the masses and intensities combined in one numpy array.
Here, the first column represents the mass-to-charge ratio while the
second column represents the intensity values.

``` python
msrun[1].peaks('centroided') # currently supported types are: raw, centroided and reprofiled
```

    array([[  120.089066, 34296.184   ],
           [  120.90026 , 45616.4     ],
           [  121.02848 ,  8870.239   ],
           ...,
           [ 1241.229   ,  3503.3232  ],
           [ 1398.1351  ,  3456.6538  ],
           [ 1466.8644  ,  3253.7253  ]], dtype=float32)

## Combining Matplotlib and pymzML

In this article we will demonstrate how to use matplotlib to plot the
extracted information. You may also explore pymzML’s [built in plotting
functions](https://pymzml.readthedocs.io/en/latest/plot.html) or
alternatively you could also use [PyQtGraph](https://www.pyqtgraph.org/)
for fast scientific plotting.

## Plotting a mass spectrum

``` python
scan = 2563

fig, axes = plt.subplots(1, 1, figsize=(6,4), dpi= 300)
axes.plot(msrun[scan].peaks('raw')[:,0],
          msrun[scan].peaks('raw')[:,1],
          "-") 
axes.set_xlabel("m/z")
axes.set_ylabel("Intensity")
axes.set_title(f'Spectrum at {msrun[scan].scan_time_in_minutes():.3f} min, Scan number: {scan}')
```

    Text(0.5, 1.0, 'Spectrum at 5.229 min, Scan number: 2563')

<img src="{{site.baseurl | prepend: site.url}}assets/images/2022_08_31_pymzML_1.jpg" alt="example" />

## Plotting the Total Ion Chromatogram (TIC)

``` python
fig, axes = plt.subplots(1, 1, figsize=(6,4), dpi= 300)
axes.plot(msrun['TIC'].peaks()[:,0], 
         msrun['TIC'].peaks()[:,1])
axes.title.set_text(f"TIC") 
axes.set_xlabel("Rt [min]")
axes.set_ylabel("Intensity")
```

    Text(0, 0.5, 'Intensity')

<img src="{{site.baseurl | prepend: site.url}}assets/images/2022_08_31_pymzML_2.jpg" alt="example" />

Alternatively, we could also use pymzML’s built in plotting function
called Factory. More information on this can be found on the project’s
[Github
page](https://github.com/pymzml/pymzML/blob/dev/example_scripts/plot_chromatogram.py),
which also contains an example code. Nice!

## Plotting an Extracted Ion Chromatogram (XIC)

To reduce computational burden, first we will separate the MS scans
belonging to level 1 and 2:

``` python
# separate ms1 and ms2 scans

ms1 = []
ms2 = []

for spectrum in msrun:
    if spectrum.ms_level == 1:
        ms1.append(spectrum)
    else:
        ms2.append(spectrum)
```

``` python
print(f"Number of MS1 scans: {len(ms1)}")
print(f"Number of MS2 scans: {len(ms2)}")
```

    Number of MS1 scans: 2229
    Number of MS2 scans: 2729

We can double check ourselves by calling get_spectrum_count() on the
msrun:

``` python
print(f"Total number of scans collected: {msrun.get_spectrum_count()}")
```

    Total number of scans collected: 4958

To get a nice XIC, we may want to select a good candidate peak that has
high intensities. For this, we can check the top n highest peaks in a
given spectrum:

``` python
msrun[2500].highest_peaks(5) # top 5 candidate peaks in scan 2500
```

    array([[1.3602165e+02, 7.7925738e+05],
           [2.2306372e+02, 8.6466119e+05],
           [1.7114923e+02, 1.2560536e+06],
           [2.6809680e+02, 1.6619788e+06],
           [3.1413858e+02, 2.5484445e+06]], dtype=float32)

``` python
XIC = []

to_extract = 3.1413858e+02

for spectrum in ms1:
    has_peak_matches = spectrum.has_peak(to_extract)
    if has_peak_matches != []:
        case = {'mz': spectrum.has_peak(to_extract)[0][0],
                'intensity': spectrum.has_peak(to_extract)[0][1],
                'rt':spectrum.scan_time_in_minutes()}
        XIC.append(case)

fig, axes = plt.subplots(1, 1, figsize=(6,4), dpi= 300)
axes.plot(pd.DataFrame(XIC).values[:,2], 
         pd.DataFrame(XIC).values[:,1])
axes.title.set_text(f"XIC of {XIC[0]['mz']:.3f}") 
axes.set_xlabel("Rt [min]")
axes.set_ylabel("Intensity")
```

    Text(0, 0.5, 'Intensity')

<img src="{{site.baseurl | prepend: site.url}}assets/images/2022_08_31_pymzML_3.jpg" alt="example" />

## Displaying the chromatogram as a heatmap

Lastly, since we are visual creatures I thought it would be a good idea
to capture the acquired chromatogram as a heatmap. The output image
could function as a piece of art, or a visual overview of our run, or we
could also use the prepared array as an input for our neuronal network.
The possibilities are endless. Let’s get to it.

### Converting all spectra to low resolution

To simplify our code and reduce computational overhead we may want to
convert our high-resolution data to low-resolution by calculating either
the mean, max or sum of all peak intensities belonging to the same
nominal masses.

Note: to accomplish this we will use the [numpy-groupies
package](https://github.com/ml31415/numpy-groupies) that allows us to
perform group-indexing operations with numpy arrays, similar to pandas’
groupby method.

numpy-groupies can be easily installed by pip install numpy_groupies.

``` python
import numpy_groupies as npg

test_agg = []
rt = []
start_mass = 120
end_mass = 1000

for scan in ms1:
    '''
    npg.aggregate is the np equivalent of the pandas groupby method. With this we can calculate means, maxes, or sums
    of peak intensities belonging to same mass integers. We are binning these peaks and calculate some 
    kind of statistics. 
    '''
    
    temp = npg.aggregate(np.round(scan.peaks('centroided')[:,0]).astype(int), 
                                  scan.peaks('centroided')[:,1],  
                                  func='max')
    '''
    npg.aggregate generates integers starting from 0 but our first mass, in this case is 120, ie all the peak intensities
    (sums, means or maxes) before our starting masses receive zeros. To prevent this, we only select the part of array that
    is of interest. In this case, intensities between masses 120 and 1000.
    '''
    
    temp = temp[np.s_[start_mass:end_mass]] 
    test_agg.append(temp)
    
    rt.append(scan.scan_time_in_minutes()) # store the retention time for plotting 
    
test_agg = pd.DataFrame(test_agg).values # convert the list of arrays to np.arrays
test_agg[np.isnan(test_agg)] = 0 # replace np.nans with 0
test_agg
```

    array([[ 34296.18359375, 147204.703125  ,  36123.08203125, ...,
                 0.        ,      0.        ,      0.        ],
           [ 33095.125     , 131241.265625  ,  39056.484375  , ...,
                 0.        ,      0.        ,      0.        ],
           [ 36464.03125   , 114666.09375   ,  32311.22070312, ...,
                 0.        ,      0.        ,      0.        ],
           ...,
           [ 21119.80078125,  54820.83984375,  35034.10546875, ...,
                 0.        ,      0.        ,      0.        ],
           [ 15056.91699219,  50564.875     ,  27216.95898438, ...,
                 0.        ,      0.        ,      0.        ],
           [ 17994.43164062,  53672.6171875 ,  24011.0859375 , ...,
                 0.        ,      0.        ,      0.        ]])

As you can see from the image below due to the highly complex nature of
the sample and the high dynamic range, first we may need to rescale our
input data before plotting.

``` python
fig, axes = plt.subplots(1, 1, figsize=(10,8),dpi = 300)
c = axes.imshow(test_agg.T,
            extent = [np.min(rt),
                      np.max(rt),
                      start_mass,
                      end_mass],
#                 vmin = 400000,
#                 vmax = 1000000,
                aspect = 'auto',
                cmap='viridis',
                origin = 'lower',
                interpolation = 'nearest'
               )
fig.colorbar(c, ax = axes)
axes.set_title('3D chromatogram before scaling', size = 20)
axes.set_xlabel('Rt [min]', size = 16)
axes.set_ylabel('m/z', size = 16)
```

    Text(0, 0.5, 'm/z')

<img src="{{site.baseurl | prepend: site.url}}assets/images/2022_08_31_pymzML_4.jpg" alt="example" />

``` python
print(f"Mininum peak intensity observed: {np.min(test_agg):.0f}")
print(f"Maximum peak intensity observed: {np.max(test_agg):.0f}")
```

    Mininum peak intensity observed: 0
    Maximum peak intensity observed: 1213927936

To rescale our input array, we will use Scikit-learn’s
[MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
This function transforms features by scaling each feature to a given
range. In this case between 0 and 1.

``` python
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()

minmax = minmax.fit_transform(test_agg)
print(f"Mininum peak intensity observed: {np.min(minmax):.0f}")
print(f"Maximum peak intensity observed: {np.max(minmax):.0f}")
```

    Mininum peak intensity observed: 0
    Maximum peak intensity observed: 1

``` python
fig, axes = plt.subplots(1, 1, figsize=(10,8),dpi = 300)
c = axes.imshow(minmax.T,
            extent = [np.min(rt),
                      np.max(rt),
                      start_mass,
                      end_mass],
                aspect = 'auto',
                cmap='viridis',
                origin = 'lower',
                interpolation = 'nearest'
               )
fig.colorbar(c, ax = axes)
axes.set_title('3D chromatogram after scaling', size = 20)
axes.set_xlabel('Rt [min]', size = 16)
axes.set_ylabel('m/z', size = 16)
```

    Text(0, 0.5, 'm/z')

<img src="{{site.baseurl | prepend: site.url}}assets/images/2022_08_31_pymzML_5.jpg" alt="example" />

**To sum up:**

-   In this article we went through some basic use cases of the pymzML
    module
-   We read in an example chromatogram acquired on a high-resolution
    mass spectrometer
-   We plotted a mass spectrum, TIC and an XIC using Matplotlib
-   Demonstrated how to process high-resolution data to prepare a 3D
    chromatogram that could be explored further down our data analysis
    pipeline
