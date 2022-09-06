Analysing the belgian rental property market with GeoPy and Plotly
================

``` python
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
from geopy.geocoders import Nominatim
import geopandas
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
import re
import plotly.express as px
import plotly.io as pio
from sklearn.impute import SimpleImputer
```

**Aim:**

-   Scrape Belgian real estate property information such as title of
    ads, prices, ZIP codes, number of bedrooms, etc
-   Find the addresses of rental properties corresponding to a set of
    coordinates using GeoPy/Nominatim
-   Plot the aggregated summary of the acquired geographic data using
    Plotly
-   Plot some of the price trends using Seaborn and Matplotlib

## Getting the data

First of all, in order to get our data we would need to identify a
suitable website that hosts a fairly comprehensive list of all rental
properties available in our target country. Since at the moment I live
in Belgium, my country of choice, maybe not too much of a surprise, was
Belgium. The data was obtained on the first of September, 2022. We
restricted our search criteria to rental properties, albeit covering the
whole country.

When this search was conducted, the website hosted over 8000 ads. To
keep our analysis up-to-date, in the future, we may consider saving the
newly published ads in a custom database. This would allow us to conduct
time-series analysis and spot some trends in the market. But this should
be a topic for another day…

To obtain the data, first we will construct two functions. The first
one, extract, allows us to obtain an HTML object containing all
information such as content, status, etc. Once we have the page content,
we would call the transform function that would allow us to extract all
the relevant information. This is where
[BeautifulSoup](https://beautiful-soup-4.readthedocs.io/en/latest/)
comes in. BeautifulSoup is a Python package for parsing HTML and XML
documents. It creates a parse tree for parsed pages that can be used to
extract data from HTML, which is useful for web scraping.

``` python
def extract(page):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.134 Safari/537.36'}
    url = f'https://immo.vlan.be/en/real-estate?transactiontypes=for-rent,to-share&propertytypes=flat,house&propertysubtypes=flat---apartment,duplex,flat---studio,ground-floor,penthouse,loft,triplex,residence,villa,master-house,bungalow,mixed-building,cottage,mansion&page={page}'
    r = requests.get(url, headers = headers)
    soup = BeautifulSoup(r.content, "html.parser")
    return soup
```

``` python
def transform(soup):
    
    divs= soup.find_all('article', class_= "list-view-item")
    for item in divs:
        title = item.find('h2').text.strip()
        price = item.find('div', class_ = 'float-right text-right mt-1').text.replace(' ', '').replace(' €', '')
        ZIP = item.find('span', itemprop= "postalCode").text
        city = item.find('span', itemprop= "addressLocality").text
        
        try:
            bedroom = item.find('div', class_= "text-center ml-2 mr-2 mb-2 NrOfBedrooms").text
        except:
            bedroom = np.nan
        
        try:
            bathroom = item.find('div', class_= "text-center ml-2 mr-2 mb-2 NrOfBathrooms").text
        except:
            bathroom = np.nan
            
        try:
            surface = item.find('div', class_= "text-center ml-2 mr-2 mb-2 LivableSurface").text.replace(' m²', '')
        except:
            surface = np.nan
            
        try:
            terrace = item.find('div', class_= "text-center ml-2 mr-2 mb-2 TerraceSurface").text.replace(' m²', '')
        except:
            terrace = np.nan
        
        website = item.find('a', href=True)['href']
        
        hit = {
            
            'title' : title,
            'price' : price,
             'ZIP' : ZIP,
             'city' : city,
             'bedroom' : bedroom,
             'bathroom' : bathroom,
             'surface' : surface,
             'terrace' : terrace,
             'website' : website
        }
        
        hitlist.append(hit)
```

In order to obtain all the ads available on the website we could scrape
the pages using a simple for loop. The range was set to 442 since there
were 442 pages available on the website containing rental properties.

``` python
hitlist = []

for i in range(1,442):
    print(f'Getting page: {i}')
    results = extract(i)
    transform(results)
    time.sleep(1)
```

Finally, we could save all of our results as a parquet file.

``` python
# convert data to pd dataframe

apartments = pd.DataFrame(hitlist)


apartments.to_csv('220901_apartment.csv')
apartments.to_parquet('220901_apartment.parquet.gzip', compression='gzip')
```

## Tidy up the data

``` python
# read data back in

apartments = pd.read_parquet('220901_apartment.parquet.gzip')
apartments
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
      <th>title</th>
      <th>price</th>
      <th>ZIP</th>
      <th>city</th>
      <th>bedroom</th>
      <th>bathroom</th>
      <th>surface</th>
      <th>terrace</th>
      <th>website</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Residence for rent</td>
      <td>850</td>
      <td>8760</td>
      <td>Meulebeke</td>
      <td>3</td>
      <td>1</td>
      <td>None</td>
      <td>None</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Flat - Apartment for rent</td>
      <td>725</td>
      <td>8940</td>
      <td>Geluwe</td>
      <td>2</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Flat - Apartment for rent</td>
      <td>750</td>
      <td>9120</td>
      <td>Beveren</td>
      <td>2</td>
      <td>1</td>
      <td>70</td>
      <td>5</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Flat - Apartment for rent</td>
      <td>950</td>
      <td>2630</td>
      <td>Aartselaar</td>
      <td>2</td>
      <td>None</td>
      <td>None</td>
      <td>15</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Residence for rent</td>
      <td>700</td>
      <td>5340</td>
      <td>Mozet</td>
      <td>2</td>
      <td>1</td>
      <td>60</td>
      <td>None</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8815</th>
      <td>Flat - Apartment for rent</td>
      <td>695</td>
      <td>9255</td>
      <td>Buggenhout</td>
      <td>2</td>
      <td>1</td>
      <td>None</td>
      <td>None</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
    </tr>
    <tr>
      <th>8816</th>
      <td>Bungalow for rent</td>
      <td>1250</td>
      <td>9850</td>
      <td>Nevele</td>
      <td>3</td>
      <td>1</td>
      <td>156</td>
      <td>None</td>
      <td>https://immo.vlan.be/en/detail/bungalow/for-re...</td>
    </tr>
    <tr>
      <th>8817</th>
      <td>Flat - Apartment for rent</td>
      <td>1370</td>
      <td>1150</td>
      <td>Sint-Pieters-Woluwe</td>
      <td>2</td>
      <td>1</td>
      <td>95</td>
      <td>None</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
    </tr>
    <tr>
      <th>8818</th>
      <td>Residence for rent</td>
      <td>850</td>
      <td>7780</td>
      <td>Comines</td>
      <td>3</td>
      <td>1</td>
      <td>160</td>
      <td>None</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
    </tr>
    <tr>
      <th>8819</th>
      <td>Flat - Apartment for rent</td>
      <td>1150</td>
      <td>1140</td>
      <td>Evere</td>
      <td>1</td>
      <td>1</td>
      <td>None</td>
      <td>None</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
    </tr>
  </tbody>
</table>
<p>8820 rows × 9 columns</p>
</div>

``` python
# check dtypes

apartments.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8820 entries, 0 to 8819
    Data columns (total 9 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   title     8820 non-null   object
     1   price     8820 non-null   object
     2   ZIP       8820 non-null   object
     3   city      8820 non-null   object
     4   bedroom   8795 non-null   object
     5   bathroom  8757 non-null   object
     6   surface   7492 non-null   object
     7   terrace   2465 non-null   object
     8   website   8820 non-null   object
    dtypes: object(9)
    memory usage: 620.3+ KB

### Converting columns to numerical

Since columns such as price, ZIP, bedroom, etc are saved as objects, we
would need to convert them to floats. We use the errors = ‘coerce’
parameter to convert any missing values to np.nan and downcast = ‘float’
to reduce the size of the resulting dataframe by storing numerical
values as the smallest numerical dtype possible.

``` python
# convert numeric columns from object to numeric

apartments[['price', 'ZIP', 'bedroom', 'bathroom', 'surface', 'terrace']] = apartments[['price', 'ZIP', 'bedroom', 'bathroom', 'surface', 'terrace']].apply(pd.to_numeric, errors = 'coerce', downcast = 'float')
apartments.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8820 entries, 0 to 8819
    Data columns (total 9 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   title     8820 non-null   object 
     1   price     8817 non-null   float32
     2   ZIP       8820 non-null   float32
     3   city      8820 non-null   object 
     4   bedroom   8795 non-null   float32
     5   bathroom  8757 non-null   float32
     6   surface   7492 non-null   float32
     7   terrace   2465 non-null   float32
     8   website   8820 non-null   object 
    dtypes: float32(6), object(3)
    memory usage: 413.6+ KB

``` python
# trim unnecessary parts from the title

apartments.title = apartments.title.str.split(' for rent', regex = True, expand = True)[0]
apartments.head()
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
      <th>title</th>
      <th>price</th>
      <th>ZIP</th>
      <th>city</th>
      <th>bedroom</th>
      <th>bathroom</th>
      <th>surface</th>
      <th>terrace</th>
      <th>website</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Residence</td>
      <td>850.0</td>
      <td>8760.0</td>
      <td>Meulebeke</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Flat - Apartment</td>
      <td>725.0</td>
      <td>8940.0</td>
      <td>Geluwe</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Flat - Apartment</td>
      <td>750.0</td>
      <td>9120.0</td>
      <td>Beveren</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>70.0</td>
      <td>5.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Flat - Apartment</td>
      <td>950.0</td>
      <td>2630.0</td>
      <td>Aartselaar</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Residence</td>
      <td>700.0</td>
      <td>5340.0</td>
      <td>Mozet</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>60.0</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
    </tr>
  </tbody>
</table>
</div>

## Getting the coordinates using GeoPy

Since we managed to obtain the ZIP codes and locations of the rental
properties, we could try to geolocate these apartments and later on plot
them based on their addresses.

**NOTE: the addresses we will obtain in the following steps represent
approximate locations, based on the city and ZIP code information
displayed within the ads.** They are in no way accurate, however, they
do allow us to observe some spatial trends when in comes to apartment
prices.

The library we will use next is the [GeoPy
library](https://geopy.readthedocs.io/en/stable/). GeoPy makes it easy
for Python developers to locate the coordinates of addresses, cities,
countries, and landmarks across the globe using third-party geocoders
and other data sources.

``` python
locations = []

apartment_locations = apartments['ZIP'].astype(str) + ',' + apartments['city']

geolocator = Nominatim(user_agent="myApp")

for idx, element in enumerate(apartment_locations):
    print(f'Working on line {idx}...')
    location = geolocator.geocode(apartment_locations[idx])
    case = {
        'latitude' : location.latitude,
        'longitude' : location.longitude,
        'address' : location.address
    }
    
    locations.append(case)
    
```

``` python
pd.DataFrame(locations).to_parquet('locations.parquet.gzip', compression='gzip') #saving coordinates to disk
```

As you can see, GeoPy allowed us to obtain information such as latitude,
longitude and the full address associated with the queried items.

``` python
locations = pd.read_parquet('locations.parquet.gzip') # reading back the saved locations data
locations
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
      <th>latitude</th>
      <th>longitude</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.948582</td>
      <td>3.286390</td>
      <td>Meulebeke, Tielt, West-Vlaanderen, Vlaanderen,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50.811434</td>
      <td>3.077164</td>
      <td>Geluwe, Wervik, Ieper, West-Vlaanderen, Vlaand...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>51.212885</td>
      <td>4.249081</td>
      <td>Beveren, Sint-Niklaas, Oost-Vlaanderen, Vlaand...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>51.133297</td>
      <td>4.387024</td>
      <td>Aartselaar, Antwerpen, Vlaanderen, 2630, Belgi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50.441640</td>
      <td>4.984670</td>
      <td>Mozet, Gesves, Namur, Wallonie, 5340, België /...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8815</th>
      <td>51.011310</td>
      <td>4.192796</td>
      <td>Buggenhout, Dendermonde, Oost-Vlaanderen, Vlaa...</td>
    </tr>
    <tr>
      <th>8816</th>
      <td>51.033037</td>
      <td>3.549107</td>
      <td>Nevele, Deinze, Gent, Oost-Vlaanderen, Vlaande...</td>
    </tr>
    <tr>
      <th>8817</th>
      <td>50.837025</td>
      <td>4.427464</td>
      <td>Woluwe-Saint-Pierre - Sint-Pieters-Woluwe, Bru...</td>
    </tr>
    <tr>
      <th>8818</th>
      <td>50.768777</td>
      <td>2.998824</td>
      <td>Comines, Comines-Warneton, Tournai-Mouscron, H...</td>
    </tr>
    <tr>
      <th>8819</th>
      <td>50.872010</td>
      <td>4.403418</td>
      <td>Evere, Brussel-Hoofdstad - Bruxelles-Capitale,...</td>
    </tr>
  </tbody>
</table>
<p>8820 rows × 3 columns</p>
</div>

## Plotting a Choropleth map using Plotly

Next, in order to plot the data with Plotly we would also need a GeoJSON
file that represents the Belgian Provinces. GeoJSON is a format for
encoding a variety of geographic data structures. Usually, it is very
easy to find a suitable GeoJSON file. I obtained mine from
[Github](https://github.com/CharleyGui/Belgium_GEOJSON/blob/main/provinces.geojson)
using a simple Google Search:

``` python
#getting the geojson file contaning Belgium province 

BelgiumProvinces = r"C:\Users\s0212777\OneDrive - Universiteit Antwerpen\Jupyter_projects\Articles\220831_apartment_search\Belgium_shapefile\BelgiumProvinces.geojson"
BelgiumProvinces = geopandas.read_file(BelgiumProvinces)
BelgiumProvinces.head()
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
      <th>ID_0</th>
      <th>ISO</th>
      <th>NAME_0</th>
      <th>ID_1</th>
      <th>NAME_1</th>
      <th>ID_2</th>
      <th>NAME_2</th>
      <th>TYPE_2</th>
      <th>ENGTYPE_2</th>
      <th>NL_NAME_2</th>
      <th>VARNAME_2</th>
      <th>NAME_2_NEW</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>BEL</td>
      <td>Belgium</td>
      <td>1</td>
      <td>Bruxelles</td>
      <td>1</td>
      <td>Bruxelles</td>
      <td>Hoofdstedelijk Gewest|Région Capitale</td>
      <td>Capital Region</td>
      <td></td>
      <td>Brussel Hoofstadt|Brusselse Hoofdstedelijke Ge...</td>
      <td>PRO-BRUXELLES</td>
      <td>POLYGON ((4.40988 50.90991, 4.41248 50.90925, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23</td>
      <td>BEL</td>
      <td>Belgium</td>
      <td>2</td>
      <td>Vlaanderen</td>
      <td>2</td>
      <td>Antwerpen</td>
      <td>Provincie</td>
      <td>Province</td>
      <td></td>
      <td>Amberes|Antuérpia|Antwerp|Anvers|Anversa</td>
      <td>PRO-ANVERS</td>
      <td>MULTIPOLYGON (((4.96307 51.45438, 4.96765 51.4...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23</td>
      <td>BEL</td>
      <td>Belgium</td>
      <td>2</td>
      <td>Vlaanderen</td>
      <td>3</td>
      <td>Limburg</td>
      <td>Provincie</td>
      <td>Province</td>
      <td></td>
      <td>Limbourg|Limburgo</td>
      <td>PRO-LIMBOURG</td>
      <td>MULTIPOLYGON (((5.89612 50.75958, 5.89459 50.7...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>BEL</td>
      <td>Belgium</td>
      <td>2</td>
      <td>Vlaanderen</td>
      <td>4</td>
      <td>Oost-Vlaanderen</td>
      <td>Provincie</td>
      <td>Province</td>
      <td></td>
      <td>Flandres Oriental|Fiandra Orientale|Flandes Or...</td>
      <td>PRO-FLANDRE ORIENTALE</td>
      <td>POLYGON ((4.23260 51.35566, 4.24206 51.35140, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>BEL</td>
      <td>Belgium</td>
      <td>2</td>
      <td>Vlaanderen</td>
      <td>5</td>
      <td>Vlaams Brabant</td>
      <td>Provincie</td>
      <td>Province</td>
      <td></td>
      <td>Brabant Flamand|Brabante Flamenco|Brabante Fla...</td>
      <td>PRO-BRABANT FLAMAND</td>
      <td>POLYGON ((4.99464 51.04142, 4.99680 51.04022, ...</td>
    </tr>
  </tbody>
</table>
</div>

We will join the the apartments dataframe that contains our search
results and the locations dataframe that includes the geographical data.

``` python
apartments_joined = apartments.join(locations) # joining the two dataframes
apartments_joined.head()
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
      <th>title</th>
      <th>price</th>
      <th>ZIP</th>
      <th>city</th>
      <th>bedroom</th>
      <th>bathroom</th>
      <th>surface</th>
      <th>terrace</th>
      <th>website</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Residence</td>
      <td>850.0</td>
      <td>8760.0</td>
      <td>Meulebeke</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
      <td>50.948582</td>
      <td>3.286390</td>
      <td>Meulebeke, Tielt, West-Vlaanderen, Vlaanderen,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Flat - Apartment</td>
      <td>725.0</td>
      <td>8940.0</td>
      <td>Geluwe</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>50.811434</td>
      <td>3.077164</td>
      <td>Geluwe, Wervik, Ieper, West-Vlaanderen, Vlaand...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Flat - Apartment</td>
      <td>750.0</td>
      <td>9120.0</td>
      <td>Beveren</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>70.0</td>
      <td>5.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>51.212885</td>
      <td>4.249081</td>
      <td>Beveren, Sint-Niklaas, Oost-Vlaanderen, Vlaand...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Flat - Apartment</td>
      <td>950.0</td>
      <td>2630.0</td>
      <td>Aartselaar</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>51.133297</td>
      <td>4.387024</td>
      <td>Aartselaar, Antwerpen, Vlaanderen, 2630, Belgi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Residence</td>
      <td>700.0</td>
      <td>5340.0</td>
      <td>Mozet</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>60.0</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
      <td>50.441640</td>
      <td>4.984670</td>
      <td>Mozet, Gesves, Namur, Wallonie, 5340, België /...</td>
    </tr>
  </tbody>
</table>
</div>

While I was playing around with the data, I noticed we had some ads with
addresses located in the Philippines. Not too sure what this is about so
it’s best to just exclude these entries for now.

``` python
apartments_joined2 = apartments_joined[apartments_joined.address != 'Brussels, Pallas Athena Executive Village Phase 2, Sitio Pulo, Imus, Cavite, Calabarzon, 4103, Philippines'].reset_index(drop = True)
apartments_joined2.head()
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
      <th>title</th>
      <th>price</th>
      <th>ZIP</th>
      <th>city</th>
      <th>bedroom</th>
      <th>bathroom</th>
      <th>surface</th>
      <th>terrace</th>
      <th>website</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Residence</td>
      <td>850.0</td>
      <td>8760.0</td>
      <td>Meulebeke</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
      <td>50.948582</td>
      <td>3.286390</td>
      <td>Meulebeke, Tielt, West-Vlaanderen, Vlaanderen,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Flat - Apartment</td>
      <td>725.0</td>
      <td>8940.0</td>
      <td>Geluwe</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>50.811434</td>
      <td>3.077164</td>
      <td>Geluwe, Wervik, Ieper, West-Vlaanderen, Vlaand...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Flat - Apartment</td>
      <td>750.0</td>
      <td>9120.0</td>
      <td>Beveren</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>70.0</td>
      <td>5.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>51.212885</td>
      <td>4.249081</td>
      <td>Beveren, Sint-Niklaas, Oost-Vlaanderen, Vlaand...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Flat - Apartment</td>
      <td>950.0</td>
      <td>2630.0</td>
      <td>Aartselaar</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>51.133297</td>
      <td>4.387024</td>
      <td>Aartselaar, Antwerpen, Vlaanderen, 2630, Belgi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Residence</td>
      <td>700.0</td>
      <td>5340.0</td>
      <td>Mozet</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>60.0</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
      <td>50.441640</td>
      <td>4.984670</td>
      <td>Mozet, Gesves, Namur, Wallonie, 5340, België /...</td>
    </tr>
  </tbody>
</table>
</div>

To use the GeoJSON data with our apartments_joined2 dataframe, we would
need to extract the province information out of the address column. This
is needed, since at the moment the address column contains all the
geographical information obtained by GeoPy. We will create a new column,
called NAME_2 that is identical to the NAME_2 column located in our
Geojson dataframe.

``` python
# saving unique province names: provinces = map_df['NAME_2'].unique()

provinces = ['Bruxelles', 'Antwerpen', 'Limburg', 'Oost-Vlaanderen',
             'Vlaams-Brabant', 'West-Vlaanderen', 'Brabant wallon', 'Hainaut',
             'Liège', 'Luxembourg', 'Namur']

# temporarily replaced Vlaams Brabant and Brabant Wallon due to spelling differences between the two columns
```

``` python
# extracting province names from string using regular expression
apartments_joined2['NAME_2'] = apartments_joined2.address.apply(lambda x: re.findall(r"(?=("+'|'.join(provinces)+r"))", x)[0])
```

``` python
# converting the names back to their original spelling

apartments_joined2.loc[apartments_joined2.NAME_2 == 'Vlaams-Brabant', 'NAME_2'] = 'Vlaams Brabant'
apartments_joined2.loc[apartments_joined2.NAME_2 == 'Brabant wallon', 'NAME_2'] = 'Brabant Wallon'
apartments_joined2.head()
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
      <th>title</th>
      <th>price</th>
      <th>ZIP</th>
      <th>city</th>
      <th>bedroom</th>
      <th>bathroom</th>
      <th>surface</th>
      <th>terrace</th>
      <th>website</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>address</th>
      <th>NAME_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Residence</td>
      <td>850.0</td>
      <td>8760.0</td>
      <td>Meulebeke</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
      <td>50.948582</td>
      <td>3.286390</td>
      <td>Meulebeke, Tielt, West-Vlaanderen, Vlaanderen,...</td>
      <td>West-Vlaanderen</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Flat - Apartment</td>
      <td>725.0</td>
      <td>8940.0</td>
      <td>Geluwe</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>50.811434</td>
      <td>3.077164</td>
      <td>Geluwe, Wervik, Ieper, West-Vlaanderen, Vlaand...</td>
      <td>West-Vlaanderen</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Flat - Apartment</td>
      <td>750.0</td>
      <td>9120.0</td>
      <td>Beveren</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>70.0</td>
      <td>5.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>51.212885</td>
      <td>4.249081</td>
      <td>Beveren, Sint-Niklaas, Oost-Vlaanderen, Vlaand...</td>
      <td>Oost-Vlaanderen</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Flat - Apartment</td>
      <td>950.0</td>
      <td>2630.0</td>
      <td>Aartselaar</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>51.133297</td>
      <td>4.387024</td>
      <td>Aartselaar, Antwerpen, Vlaanderen, 2630, Belgi...</td>
      <td>Antwerpen</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Residence</td>
      <td>700.0</td>
      <td>5340.0</td>
      <td>Mozet</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>60.0</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
      <td>50.441640</td>
      <td>4.984670</td>
      <td>Mozet, Gesves, Namur, Wallonie, 5340, België /...</td>
      <td>Namur</td>
    </tr>
  </tbody>
</table>
</div>

Now we can calculate the median house price per province and find out
how many apartments are advertised in that specific province.

``` python
median_price = apartments_joined2.groupby('NAME_2')['price'].agg(['median', 'size']).reset_index()
median_price
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
      <th>NAME_2</th>
      <th>median</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Antwerpen</td>
      <td>650.0</td>
      <td>454</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brabant Wallon</td>
      <td>795.0</td>
      <td>430</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bruxelles</td>
      <td>1370.0</td>
      <td>1758</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hainaut</td>
      <td>730.0</td>
      <td>1992</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Limburg</td>
      <td>870.0</td>
      <td>438</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Liège</td>
      <td>742.5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Luxembourg</td>
      <td>762.5</td>
      <td>8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Namur</td>
      <td>850.0</td>
      <td>432</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Oost-Vlaanderen</td>
      <td>695.0</td>
      <td>1315</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Vlaams Brabant</td>
      <td>1350.0</td>
      <td>865</td>
    </tr>
    <tr>
      <th>10</th>
      <td>West-Vlaanderen</td>
      <td>630.0</td>
      <td>478</td>
    </tr>
  </tbody>
</table>
</div>

``` python
import plotly.graph_objects as go

pio.renderers.default = "notebook_connected"

fig = px.choropleth_mapbox(data_frame = median_price,
                           geojson = BelgiumProvinces,
                           color = 'median',
                           locations = 'NAME_2',
                           featureidkey = 'properties.NAME_2',
                           hover_name = 'NAME_2',
                           hover_data = ['size', 'median'],
                           labels = {'NAME_2' : 'Province', 
                                    'median' : 'Median rent price in euros',
                                    'size' : 'Number of apartments in the area'},
                           mapbox_style = 'carto-positron',
                           center = {'lat': 50.5, 'lon' : 4},
                           zoom = 6,
                           opacity = 0.5,
                           title = 'Median rental prices in Belgium [€], 2022 September'
                          )

fig.add_scattermapbox(lat = apartments_joined2.latitude,
                      lon = apartments_joined2.longitude,
                      hovertext = apartments_joined2.city,
                      name = '',
                      marker_size = 5,
                      marker_color = 'black',
                      opacity = 0.25)

#fig.write_html('median_price.html', auto_open=True)
fig.show()
```

{% include 2022_08_31_apartment_search_median_price.html %} 

## Using sklearn SimpleImputer to replace missing surface values

We would also be interested in what is considered to be a bargain price
in Belgium. Since, we have quite a few missing values in the surface
column (about 10%), it would be difficult to calculate the corresponding
median_imputed_surface values to determine in which province/city we can
get the most bang for our buck. Luckily we can use [sklearn’s
SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html),
that allows us to replace missing values using a descriptive statistic
(e.g. mean, median, or most frequent) along each column, or using a
constant value.

``` python
imp = SimpleImputer(missing_values=np.nan, strategy='median')

apartments_joined2['median_imputed_surface'] =  imp.fit_transform(apartments_joined2['surface'].values.reshape(-1, 1))
apartments_joined2.head()
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
      <th>title</th>
      <th>price</th>
      <th>ZIP</th>
      <th>city</th>
      <th>bedroom</th>
      <th>bathroom</th>
      <th>surface</th>
      <th>terrace</th>
      <th>website</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>address</th>
      <th>NAME_2</th>
      <th>median_imputed_surface</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Residence</td>
      <td>850.0</td>
      <td>8760.0</td>
      <td>Meulebeke</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
      <td>50.948582</td>
      <td>3.286390</td>
      <td>Meulebeke, Tielt, West-Vlaanderen, Vlaanderen,...</td>
      <td>West-Vlaanderen</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Flat - Apartment</td>
      <td>725.0</td>
      <td>8940.0</td>
      <td>Geluwe</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>50.811434</td>
      <td>3.077164</td>
      <td>Geluwe, Wervik, Ieper, West-Vlaanderen, Vlaand...</td>
      <td>West-Vlaanderen</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Flat - Apartment</td>
      <td>750.0</td>
      <td>9120.0</td>
      <td>Beveren</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>70.0</td>
      <td>5.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>51.212885</td>
      <td>4.249081</td>
      <td>Beveren, Sint-Niklaas, Oost-Vlaanderen, Vlaand...</td>
      <td>Oost-Vlaanderen</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Flat - Apartment</td>
      <td>950.0</td>
      <td>2630.0</td>
      <td>Aartselaar</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>51.133297</td>
      <td>4.387024</td>
      <td>Aartselaar, Antwerpen, Vlaanderen, 2630, Belgi...</td>
      <td>Antwerpen</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Residence</td>
      <td>700.0</td>
      <td>5340.0</td>
      <td>Mozet</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>60.0</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
      <td>50.441640</td>
      <td>4.984670</td>
      <td>Mozet, Gesves, Namur, Wallonie, 5340, België /...</td>
      <td>Namur</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
</div>

``` python
apartments_joined2['price_to_surface'] = apartments_joined2['price'] / apartments_joined2['median_imputed_surface']
apartments_joined2.head()
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
      <th>title</th>
      <th>price</th>
      <th>ZIP</th>
      <th>city</th>
      <th>bedroom</th>
      <th>bathroom</th>
      <th>surface</th>
      <th>terrace</th>
      <th>website</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>address</th>
      <th>NAME_2</th>
      <th>median_imputed_surface</th>
      <th>price_to_surface</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Residence</td>
      <td>850.0</td>
      <td>8760.0</td>
      <td>Meulebeke</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
      <td>50.948582</td>
      <td>3.286390</td>
      <td>Meulebeke, Tielt, West-Vlaanderen, Vlaanderen,...</td>
      <td>West-Vlaanderen</td>
      <td>96.0</td>
      <td>8.854167</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Flat - Apartment</td>
      <td>725.0</td>
      <td>8940.0</td>
      <td>Geluwe</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>50.811434</td>
      <td>3.077164</td>
      <td>Geluwe, Wervik, Ieper, West-Vlaanderen, Vlaand...</td>
      <td>West-Vlaanderen</td>
      <td>96.0</td>
      <td>7.552083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Flat - Apartment</td>
      <td>750.0</td>
      <td>9120.0</td>
      <td>Beveren</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>70.0</td>
      <td>5.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>51.212885</td>
      <td>4.249081</td>
      <td>Beveren, Sint-Niklaas, Oost-Vlaanderen, Vlaand...</td>
      <td>Oost-Vlaanderen</td>
      <td>70.0</td>
      <td>10.714286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Flat - Apartment</td>
      <td>950.0</td>
      <td>2630.0</td>
      <td>Aartselaar</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.0</td>
      <td>https://immo.vlan.be/en/detail/flat---apartmen...</td>
      <td>51.133297</td>
      <td>4.387024</td>
      <td>Aartselaar, Antwerpen, Vlaanderen, 2630, Belgi...</td>
      <td>Antwerpen</td>
      <td>96.0</td>
      <td>9.895833</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Residence</td>
      <td>700.0</td>
      <td>5340.0</td>
      <td>Mozet</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>60.0</td>
      <td>NaN</td>
      <td>https://immo.vlan.be/en/detail/residence/for-r...</td>
      <td>50.441640</td>
      <td>4.984670</td>
      <td>Mozet, Gesves, Namur, Wallonie, 5340, België /...</td>
      <td>Namur</td>
      <td>60.0</td>
      <td>11.666667</td>
    </tr>
  </tbody>
</table>
</div>

## Plot some interesting price trends using Seaborn and Matplotlib

``` python
plt.style.use('seaborn-darkgrid')
#sns.set_style("white")
plt.figure(dpi= 300)
sns.set_context(context= 'talk', font_scale=0.5)


fig, axes = plt.subplots(3, 2, figsize=(15,10))

sns.barplot(x = apartments_joined2.NAME_2.value_counts().values, 
            y = apartments_joined2.NAME_2.value_counts().index,
            ax = axes[0,0],
           dodge = False,
           palette= 'colorblind')
axes[0,0].set_ylabel('')
axes[0,0].set_xlabel('')
axes[0,0].set_title('Number of ads per province')

sns.barplot(
            x = apartments_joined2.groupby(['city', 'NAME_2']).size().reset_index().sort_values(by = 0, ascending = False).head(15)[0], 
            y = apartments_joined2.groupby(['city', 'NAME_2']).size().reset_index().sort_values(by = 0, ascending = False).head(15).city, 
            ax = axes[0,1],
            hue = apartments_joined2.groupby(['city', 'NAME_2']).size().reset_index().sort_values(by = 0, ascending = False).head(15).NAME_2,
           dodge = False,
           palette= 'colorblind')
axes[0,1].set_ylabel('')
axes[0,1].set_xlabel('')
axes[0,1].set_title('Number of ads per city')
axes[0,1].legend(title = 'Province')

sns.barplot(
            x = apartments_joined2.groupby('NAME_2')['price'].median().sort_values(ascending = False).values, 
            y = apartments_joined2.groupby('NAME_2')['price'].median().sort_values(ascending = False).index, 
            ax = axes[1,0],
           dodge = False,
           palette= 'colorblind')
axes[1,0].set_ylabel('')
axes[1,0].set_xlabel('')
axes[1,0].set_title('Median apartment prices per province')

sns.barplot(x = apartments_joined2.groupby(['city', 'NAME_2'])['price'].median().reset_index().sort_values(by = 'price', ascending = False).head(15).price, 
            y = apartments_joined2.groupby(['city', 'NAME_2'])['price'].median().reset_index().sort_values(by = 'price', ascending = False).head(15).city,
            hue = apartments_joined2.groupby(['city', 'NAME_2'])['price'].median().reset_index().sort_values(by = 'price', ascending = False).head(15).NAME_2,
            ax = axes[1,1],
            dodge = False,
            palette= 'colorblind')
axes[1,1].set_ylabel('')
axes[1,1].set_xlabel('')
axes[1,1].set_title('Median apartment prices per city')
axes[1,1].legend(title = 'Province')


sns.barplot(x = apartments_joined2.groupby(['NAME_2'])['price_to_surface'].median().sort_values().values, 
            y = apartments_joined2.groupby(['NAME_2'])['price_to_surface'].median().sort_values().index,
            ax = axes[2,0],
            dodge = False,
            palette= 'colorblind')
axes[2,0].set_ylabel('')
axes[2,0].set_xlabel('')
axes[2,0].set_title('Median Euro to m2 ratio per province')


sns.barplot(x = apartments_joined2.groupby(['NAME_2', 'city'])['price_to_surface'].median().reset_index().sort_values(by = 'price_to_surface').head(15).price_to_surface, 
            y = apartments_joined2.groupby(['NAME_2', 'city'])['price_to_surface'].median().reset_index().sort_values(by = 'price_to_surface').head(15).city,
            ax = axes[2,1],
            hue = apartments_joined2.groupby(['NAME_2', 'city'])['price_to_surface'].median().reset_index().sort_values(by = 'price_to_surface').head(15).NAME_2,
            dodge = False,
            palette= 'colorblind')
axes[2,1].set_ylabel('')
axes[2,1].set_xlabel('')
axes[2,1].set_title('Median Euro to m2 ratio per city')
axes[2,1].legend(title = 'Province')
fig.suptitle('What are the most popular Belgian provinces/cities? How much do apartments cost there \nand where can we find a good bargain?', fontsize = 18)


fig.tight_layout()
```

    <Figure size 1920x1440 with 0 Axes>

<img src="{{site.baseurl | prepend: site.url}}assets/images/2022_08_31_apartment_search.jpeg" alt="example" />

**To sum up:**

-   We showed how to use requests, a python built in module and
    BeautifulSoup to obtain up-to-date information about rental
    properties in the area
-   Demonstrated how one can utilize GeoPy/Nominatim when looking for
    geographical data
-   Highlighted how Plotly’s choropleth_mapbox can be a useful tool when
    one would like to inestigate how a variable, in our case rental
    price, can vary across a given geographic area
-   We also constructed a graph that shows us the most popular Belgian
    provinces/cities, median rental costs and most affordable places
    around the country
