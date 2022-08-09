Scraping and analyzing Pubmed articles related to Machine learning and
Deep learning
================

# Scraping Pubmed articles

## Aim:

-   Download information from Pubmed pertaining to ML/DL research
    articles, such as title, author name, abstract etc
-   Tidy up data so that it is amenable to further data analysis
-   Show some trends in the field through visualization

**Note:** For the sake of this demonstration we are limited to 10,000
search results out of the availabe 22,483 articles. This is due to the
fact that Pubmed can only display [10,000
search](https://pubmed.ncbi.nlm.nih.gov/help/) results at any given
time. The data was retrieved on 7/8/2022, using the search terms
(machine learning) AND (deep learning).

Hope you will enjoy it!

``` python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
```

## Defining functions

``` python
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def extract(page):
    retry_strategy = Retry(
        total=10,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"],
        backoff_factor= 1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    headers= {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.134 Safari/537.36'}
    url= f'https://pubmed.ncbi.nlm.nih.gov/?term=(machine%20learning)%20AND%20(deep%20learning)&format=abstract&size=200&page={page}'
    s = requests.Session()
    r= s.get(url, headers=headers)
    #r= requests.get(url, headers)
    soup= BeautifulSoup(r.content, 'html.parser')
    #return r.status_code
    return soup
```

``` python
def transform(soup):
    divs= soup.find_all('article', class_= "article-overview")
    global institute
    for item in divs:
        title=  item.find('h1', class_= 'heading-title').text.strip()#article name
        journal= item.find('button').text.strip() #journal name
        date= item.find('span', class_= 'cit').text.strip() #publication year, this needs to be worked on
        try:
            DOI= item.find('span', class_= 'citation-doi').text.strip() #DOI
        except:
            DOI= 'NA'
        for i in item.find_all('a', class_= 'full-name'):
            first_author= i.text.strip() #Authors, extracts names twice
        try:
            for p in item.find_all('ul', attrs={'class': 'item-list'}):
                institute= p.text.strip() #Authors, extracts names twice
        except:
            institute= 'NA'
        abstract= item.find('div', class_= 'abstract').text.strip() #abstract
        publication= {
            'title': title,
            'journal':journal,
            'date': date,
            'DOI': DOI,
            'first_author': first_author,
            'institute': institute,
            'abstract': abstract,
        }
        publications.append(publication)
    return 
```

## Pipeline

``` python
publications= []

for i in range(0,51):
    #print(f'Getting page {i}')
    global institute
    temp= extract(i)
    transform(temp)
```

## Convert list to dataframe and save it as a parquet file

``` python
publications= pd.DataFrame(publications)
publications.to_parquet('publications.parquet.gzip',compression='gzip')  #saving file to drive
```

``` python
publications= pd.read_parquet('publications.parquet.gzip') #reading in the saved file. This could be used for future read-ins
publications.head()
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
      <th>journal</th>
      <th>date</th>
      <th>DOI</th>
      <th>first_author</th>
      <th>institute</th>
      <th>abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Machine Learning and Deep Learning in Medical ...</td>
      <td>J Med Imaging Radiat Sci</td>
      <td>2019 Dec;50(4):477-487.</td>
      <td>doi: 10.1016/j.jmir.2019.09.005.</td>
      <td>Ran Klein</td>
      <td>1 Charles Sturt University, NSW, Australia. El...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Machine and deep learning methods for radiomics</td>
      <td>Med Phys</td>
      <td>2020 Jun;47(5):e185-e202.</td>
      <td>doi: 10.1002/mp.13678.</td>
      <td>Issam El Naqa</td>
      <td>1 Department of Medical Physics, Centro di Rif...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Introduction to Machine Learning, Neural Netwo...</td>
      <td>Transl Vis Sci Technol</td>
      <td>2020 Feb 27;9(2):14.</td>
      <td>doi: 10.1167/tvst.9.2.14.</td>
      <td>J Peter Campbell</td>
      <td>1 Department of Ophthalmology, Casey Eye Insti...</td>
      <td>Abstract\n        \n      \n\n\n\n          Pu...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Deep Learning in Medical Image Analysis</td>
      <td>Adv Exp Med Biol</td>
      <td>2020;1213:3-21.</td>
      <td>doi: 10.1007/978-3-030-33128-3_1.</td>
      <td>Chuan Zhou</td>
      <td>1 Department of Radiology, University of Michi...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Diagnosis of COVID-19 Using Machine Learning a...</td>
      <td>Curr Med Imaging</td>
      <td>2021;17(12):1403-1418.</td>
      <td>doi: 10.2174/1573405617666210713113439.</td>
      <td>Prajoy Podder</td>
      <td>1 Institute of ICT, Bangladesh University of E...</td>
      <td>Abstract\n        \n      \n\n\n\n          Ba...</td>
    </tr>
  </tbody>
</table>
</div>

## Data wrangling

### Splitting publication date column and converting it to datetime format

``` python
publications['date']= publications['date'].str.split(pat= ';', expand= True)[0]
publications
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
      <th>journal</th>
      <th>date</th>
      <th>DOI</th>
      <th>first_author</th>
      <th>institute</th>
      <th>abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Machine Learning and Deep Learning in Medical ...</td>
      <td>J Med Imaging Radiat Sci</td>
      <td>2019 Dec</td>
      <td>doi: 10.1016/j.jmir.2019.09.005.</td>
      <td>Ran Klein</td>
      <td>1 Charles Sturt University, NSW, Australia. El...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Machine and deep learning methods for radiomics</td>
      <td>Med Phys</td>
      <td>2020 Jun</td>
      <td>doi: 10.1002/mp.13678.</td>
      <td>Issam El Naqa</td>
      <td>1 Department of Medical Physics, Centro di Rif...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Introduction to Machine Learning, Neural Netwo...</td>
      <td>Transl Vis Sci Technol</td>
      <td>2020 Feb 27</td>
      <td>doi: 10.1167/tvst.9.2.14.</td>
      <td>J Peter Campbell</td>
      <td>1 Department of Ophthalmology, Casey Eye Insti...</td>
      <td>Abstract\n        \n      \n\n\n\n          Pu...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Deep Learning in Medical Image Analysis</td>
      <td>Adv Exp Med Biol</td>
      <td>2020</td>
      <td>doi: 10.1007/978-3-030-33128-3_1.</td>
      <td>Chuan Zhou</td>
      <td>1 Department of Radiology, University of Michi...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Diagnosis of COVID-19 Using Machine Learning a...</td>
      <td>Curr Med Imaging</td>
      <td>2021</td>
      <td>doi: 10.2174/1573405617666210713113439.</td>
      <td>Prajoy Podder</td>
      <td>1 Institute of ICT, Bangladesh University of E...</td>
      <td>Abstract\n        \n      \n\n\n\n          Ba...</td>
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
    </tr>
    <tr>
      <th>9995</th>
      <td>Developed and validated a prognostic nomogram ...</td>
      <td>EBioMedicine</td>
      <td>2019 Jan</td>
      <td>doi: 10.1016/j.ebiom.2018.12.028.</td>
      <td>Guoxin Li</td>
      <td>1 Department of General Surgery, Nanfang Hospi...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>Signal identification system for developing re...</td>
      <td>Artif Intell Med</td>
      <td>2020 Jan</td>
      <td>doi: 10.1016/j.artmed.2019.101755.</td>
      <td>Radeep Krishna Radhakrishnan Nair</td>
      <td>1 School of Management, Jilin University, Chan...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>Challenges and Opportunities in Machine-Augmen...</td>
      <td>Trends Plant Sci</td>
      <td>2021 Jan</td>
      <td>doi: 10.1016/j.tplants.2020.07.010.</td>
      <td>Koushik Nagasubramanian</td>
      <td>1 Department of Agronomy, Iowa State Universit...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>Uncertainty and interpretability in convolutio...</td>
      <td>Med Image Anal</td>
      <td>2020 Feb</td>
      <td>doi: 10.1016/j.media.2019.101619.</td>
      <td>Robert Jenssen</td>
      <td>1 Department of Physics and Technology, UiT Th...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>Table Cleaning Task by Human Support Robot Usi...</td>
      <td>Sensors (Basel)</td>
      <td>2020 Mar 18</td>
      <td>doi: 10.3390/s20061698.</td>
      <td>Anh Vu Le</td>
      <td>1 Engineering Product Development Pillar, Sing...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 7 columns</p>
</div>

``` python
publications.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 7 columns):
     #   Column        Non-Null Count  Dtype 
    ---  ------        --------------  ----- 
     0   title         10000 non-null  object
     1   journal       10000 non-null  object
     2   date          10000 non-null  object
     3   DOI           10000 non-null  object
     4   first_author  10000 non-null  object
     5   institute     10000 non-null  object
     6   abstract      10000 non-null  object
    dtypes: object(7)
    memory usage: 547.0+ KB

``` python
publications['date']= pd.to_datetime(publications.date, errors= 'coerce')
publications.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 7 columns):
     #   Column        Non-Null Count  Dtype         
    ---  ------        --------------  -----         
     0   title         10000 non-null  object        
     1   journal       10000 non-null  object        
     2   date          9853 non-null   datetime64[ns]
     3   DOI           10000 non-null  object        
     4   first_author  10000 non-null  object        
     5   institute     10000 non-null  object        
     6   abstract      10000 non-null  object        
    dtypes: datetime64[ns](1), object(6)
    memory usage: 547.0+ KB

## Extract country names from citation data

``` python
publications.institute= publications.institute.str.replace(pat= r'[0-9]+', repl= '', regex= True).str.replace(pat= '\n',repl= '',) #remove numbers and extra spaces from the institute column
publications.head()
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
      <th>journal</th>
      <th>date</th>
      <th>DOI</th>
      <th>first_author</th>
      <th>institute</th>
      <th>abstract</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Machine Learning and Deep Learning in Medical ...</td>
      <td>J Med Imaging Radiat Sci</td>
      <td>2019-12-01</td>
      <td>doi: 10.1016/j.jmir.2019.09.005.</td>
      <td>Ran Klein</td>
      <td>Charles Sturt University, NSW, Australia. Ele...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Machine and deep learning methods for radiomics</td>
      <td>Med Phys</td>
      <td>2020-06-01</td>
      <td>doi: 10.1002/mp.13678.</td>
      <td>Issam El Naqa</td>
      <td>Department of Medical Physics, Centro di Rife...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Introduction to Machine Learning, Neural Netwo...</td>
      <td>Transl Vis Sci Technol</td>
      <td>2020-02-27</td>
      <td>doi: 10.1167/tvst.9.2.14.</td>
      <td>J Peter Campbell</td>
      <td>Department of Ophthalmology, Casey Eye Instit...</td>
      <td>Abstract\n        \n      \n\n\n\n          Pu...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Deep Learning in Medical Image Analysis</td>
      <td>Adv Exp Med Biol</td>
      <td>2020-01-01</td>
      <td>doi: 10.1007/978-3-030-33128-3_1.</td>
      <td>Chuan Zhou</td>
      <td>Department of Radiology, University of Michig...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Diagnosis of COVID-19 Using Machine Learning a...</td>
      <td>Curr Med Imaging</td>
      <td>2021-01-01</td>
      <td>doi: 10.2174/1573405617666210713113439.</td>
      <td>Prajoy Podder</td>
      <td>Institute of ICT, Bangladesh University of En...</td>
      <td>Abstract\n        \n      \n\n\n\n          Ba...</td>
    </tr>
  </tbody>
</table>
</div>

To extract country names from the citation records we will use the
pycountry library. For installation instructions please visit [this
website.](https://pypi.org/project/pycountry/). First, we will save the
extracted country names in a list of dictionaries. Please keep in mind
that there could be multiple countries (or institutions from multiple
countries) countributing to the articles.

Python dictionary is an ideal way of storing such data. Country names
belonging to the same publication recieves the same key.

For example, if the first article in our institute column is the result
of the collaboration of two countries, let’s say China and Germany, our
dictionary would look something like this: {‘key0’: ‘China’, ‘key0’:
‘Germany’} while the second article could look something like: {‘key1’:
‘France’}.

``` python
import pycountry
#here we create a list of dictionaries where countries belonging to the same row recieve the same dict key (publications could come from multiple countries), this also contains nans
l= []

for idx,item in enumerate(publications.institute):
    for country in pycountry.countries:
        d= {}
        if country.name in item:
            #d= {}
            #test.append(country.name)
            d[idx] = [country.name]
            l.append(d)
        elif country.alpha_3 in item: #publications['institute'][0].split('.')[4]:
            d[idx] = [country.name]
            l.append(d)
        else: #(country.alpha_3 or country.name) not in item: #publications['institute'][0].split('.')[4]:
            d[idx] = np.nan
            l.append(d)
```

``` python
#here we 'flatten' the dictionaries, so that values with same keys go to the same level
def combine(dictionaries):
    combined_dict = {}
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            combined_dict.setdefault(key, []).append(value)
    return combined_dict

countries= combine(l)
```

``` python
countries= pd.DataFrame.from_dict(countries, orient='index').dropna(axis= 1, how= 'all') #drop all the columns that only contains nans
countries.shape
```

    (10000, 159)

``` python
#convert the lists to strings, for example Australia is a list instead of a str. Then we prepare a pandas series/dataframe

countries = [','.join(item) if isinstance(item, list) else item for colname in countries for item in countries[colname]] 
countries= pd.DataFrame(countries)
countries.shape
```

    (1590000, 1)

``` python
'''here we split up the list to multiple chuncks. We start out with a flat list with dimensions of (1590000, 1) and we would like to get back the original (10000, 159) dimansions. 
We would need to do this since eventually, we would like to place our list back into a DataFrame but at the moment we only have a flat series.
'''

chunked_list = list()
for i in range(0, len(countries), 10000):
    chunked_list.append(countries[i:i+10000])
#print(chunked_list)
```

``` python
#placed the chuncked list back to a dataframe
countries= pd.DataFrame(np.asarray(chunked_list).reshape(159, 10000).T)
countries.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>149</th>
      <th>150</th>
      <th>151</th>
      <th>152</th>
      <th>153</th>
      <th>154</th>
      <th>155</th>
      <th>156</th>
      <th>157</th>
      <th>158</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Australia</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 159 columns</p>
</div>

``` python
#Next we rename the columns with the unique country names. First we find all the unique country names without np.nans
unique= countries.describe().loc['top',:]
unique
```

    0                            Afghanistan
    1                               Anguilla
    2                                Albania
    3                                Andorra
    4                   United Arab Emirates
                         ...                
    154    Venezuela, Bolivarian Republic of
    155                 Virgin Islands, U.S.
    156                             Viet Nam
    157                                Yemen
    158                         South Africa
    Name: top, Length: 159, dtype: object

``` python
#renaming the columns
countries.columns= unique
countries.head()
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
      <th>top</th>
      <th>Afghanistan</th>
      <th>Anguilla</th>
      <th>Albania</th>
      <th>Andorra</th>
      <th>United Arab Emirates</th>
      <th>Argentina</th>
      <th>American Samoa</th>
      <th>Antarctica</th>
      <th>Australia</th>
      <th>Austria</th>
      <th>...</th>
      <th>Uganda</th>
      <th>Ukraine</th>
      <th>United States Minor Outlying Islands</th>
      <th>United States</th>
      <th>Uzbekistan</th>
      <th>Venezuela, Bolivarian Republic of</th>
      <th>Virgin Islands, U.S.</th>
      <th>Viet Nam</th>
      <th>Yemen</th>
      <th>South Africa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Australia</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>United States</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 159 columns</p>
</div>

``` python
countries= countries.replace(np.nan,0) #converting all np.nan values to zeros
countries.head()
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
      <th>top</th>
      <th>Afghanistan</th>
      <th>Anguilla</th>
      <th>Albania</th>
      <th>Andorra</th>
      <th>United Arab Emirates</th>
      <th>Argentina</th>
      <th>American Samoa</th>
      <th>Antarctica</th>
      <th>Australia</th>
      <th>Austria</th>
      <th>...</th>
      <th>Uganda</th>
      <th>Ukraine</th>
      <th>United States Minor Outlying Islands</th>
      <th>United States</th>
      <th>Uzbekistan</th>
      <th>Venezuela, Bolivarian Republic of</th>
      <th>Virgin Islands, U.S.</th>
      <th>Viet Nam</th>
      <th>Yemen</th>
      <th>South Africa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Australia</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>United States</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>United States</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>United States</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>United States</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 159 columns</p>
</div>

``` python
'''converting all str values to ones
for this we need the value names in the columns and the values we want them to be replaced with (1 in this case). Create a dict by zipping the two lists together:'''
to_rename= dict(zip(unique, np.ones(shape= unique.shape,  dtype= int).tolist()))
dict(list(to_rename.items())[0:5]) #display first 5 items
```

    {'Afghanistan': 1,
     'Anguilla': 1,
     'Albania': 1,
     'Andorra': 1,
     'United Arab Emirates': 1}

``` python
countries= countries.applymap(lambda s: to_rename.get(s) if s in to_rename else s) #replace country names with ones
countries.head()
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
      <th>top</th>
      <th>Afghanistan</th>
      <th>Anguilla</th>
      <th>Albania</th>
      <th>Andorra</th>
      <th>United Arab Emirates</th>
      <th>Argentina</th>
      <th>American Samoa</th>
      <th>Antarctica</th>
      <th>Australia</th>
      <th>Austria</th>
      <th>...</th>
      <th>Uganda</th>
      <th>Ukraine</th>
      <th>United States Minor Outlying Islands</th>
      <th>United States</th>
      <th>Uzbekistan</th>
      <th>Venezuela, Bolivarian Republic of</th>
      <th>Virgin Islands, U.S.</th>
      <th>Viet Nam</th>
      <th>Yemen</th>
      <th>South Africa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 159 columns</p>
</div>

``` python
# We can finally join together our original dataframe with the countries dataframe containing frequecy occurrences

publications= publications.join(countries)
publications.head()
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
      <th>journal</th>
      <th>date</th>
      <th>DOI</th>
      <th>first_author</th>
      <th>institute</th>
      <th>abstract</th>
      <th>Afghanistan</th>
      <th>Anguilla</th>
      <th>Albania</th>
      <th>...</th>
      <th>Uganda</th>
      <th>Ukraine</th>
      <th>United States Minor Outlying Islands</th>
      <th>United States</th>
      <th>Uzbekistan</th>
      <th>Venezuela, Bolivarian Republic of</th>
      <th>Virgin Islands, U.S.</th>
      <th>Viet Nam</th>
      <th>Yemen</th>
      <th>South Africa</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Machine Learning and Deep Learning in Medical ...</td>
      <td>J Med Imaging Radiat Sci</td>
      <td>2019-12-01</td>
      <td>doi: 10.1016/j.jmir.2019.09.005.</td>
      <td>Ran Klein</td>
      <td>Charles Sturt University, NSW, Australia. Ele...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Machine and deep learning methods for radiomics</td>
      <td>Med Phys</td>
      <td>2020-06-01</td>
      <td>doi: 10.1002/mp.13678.</td>
      <td>Issam El Naqa</td>
      <td>Department of Medical Physics, Centro di Rife...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Introduction to Machine Learning, Neural Netwo...</td>
      <td>Transl Vis Sci Technol</td>
      <td>2020-02-27</td>
      <td>doi: 10.1167/tvst.9.2.14.</td>
      <td>J Peter Campbell</td>
      <td>Department of Ophthalmology, Casey Eye Instit...</td>
      <td>Abstract\n        \n      \n\n\n\n          Pu...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Deep Learning in Medical Image Analysis</td>
      <td>Adv Exp Med Biol</td>
      <td>2020-01-01</td>
      <td>doi: 10.1007/978-3-030-33128-3_1.</td>
      <td>Chuan Zhou</td>
      <td>Department of Radiology, University of Michig...</td>
      <td>Abstract\n        \n      \n\n\n      \n      ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Diagnosis of COVID-19 Using Machine Learning a...</td>
      <td>Curr Med Imaging</td>
      <td>2021-01-01</td>
      <td>doi: 10.2174/1573405617666210713113439.</td>
      <td>Prajoy Podder</td>
      <td>Institute of ICT, Bangladesh University of En...</td>
      <td>Abstract\n        \n      \n\n\n\n          Ba...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 166 columns</p>
</div>

``` python
#Here we further butify the Abstract column by separating it from the Keywords. The keywords section were originally part of the article abstract.

publications['abstract']= publications['abstract'].str.replace('\n','')
publications['Abstract']= publications['abstract'].str.extract(r"Abstract(.+)")
publications['Keywords']= publications['abstract'].str.extract(r"Keywords:(.+)")
publications.head()
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
      <th>journal</th>
      <th>date</th>
      <th>DOI</th>
      <th>first_author</th>
      <th>institute</th>
      <th>abstract</th>
      <th>Afghanistan</th>
      <th>Anguilla</th>
      <th>Albania</th>
      <th>...</th>
      <th>United States Minor Outlying Islands</th>
      <th>United States</th>
      <th>Uzbekistan</th>
      <th>Venezuela, Bolivarian Republic of</th>
      <th>Virgin Islands, U.S.</th>
      <th>Viet Nam</th>
      <th>Yemen</th>
      <th>South Africa</th>
      <th>Abstract</th>
      <th>Keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Machine Learning and Deep Learning in Medical ...</td>
      <td>J Med Imaging Radiat Sci</td>
      <td>2019-12-01</td>
      <td>doi: 10.1016/j.jmir.2019.09.005.</td>
      <td>Ran Klein</td>
      <td>Charles Sturt University, NSW, Australia. Ele...</td>
      <td>Abstract                          Artificial i...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Artificial intellige...</td>
      <td>Medical imaging; artificia...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Machine and deep learning methods for radiomics</td>
      <td>Med Phys</td>
      <td>2020-06-01</td>
      <td>doi: 10.1002/mp.13678.</td>
      <td>Issam El Naqa</td>
      <td>Department of Medical Physics, Centro di Rife...</td>
      <td>Abstract                          Radiomics is...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Radiomics is an emer...</td>
      <td>deep learning; machine lea...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Introduction to Machine Learning, Neural Netwo...</td>
      <td>Transl Vis Sci Technol</td>
      <td>2020-02-27</td>
      <td>doi: 10.1167/tvst.9.2.14.</td>
      <td>J Peter Campbell</td>
      <td>Department of Ophthalmology, Casey Eye Instit...</td>
      <td>Abstract                        Purpose:      ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Purpose:              ...</td>
      <td>artificial intelligence; d...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Deep Learning in Medical Image Analysis</td>
      <td>Adv Exp Med Biol</td>
      <td>2020-01-01</td>
      <td>doi: 10.1007/978-3-030-33128-3_1.</td>
      <td>Chuan Zhou</td>
      <td>Department of Radiology, University of Michig...</td>
      <td>Abstract                          Deep learnin...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Deep learning is the...</td>
      <td>Artificial intelligence; B...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Diagnosis of COVID-19 Using Machine Learning a...</td>
      <td>Curr Med Imaging</td>
      <td>2021-01-01</td>
      <td>doi: 10.2174/1573405617666210713113439.</td>
      <td>Prajoy Podder</td>
      <td>Institute of ICT, Bangladesh University of En...</td>
      <td>Abstract                        Background:   ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Background:           ...</td>
      <td>Artificial intelligence; C...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 168 columns</p>
</div>

``` python
publications['Abstract']= publications['Abstract'].str.extract("(.+) Keywords:")
publications['Abstract']= publications['Abstract'].str.strip()
```

``` python
#Here we prepare the long form of our publications dataframe. We would need this for easier plotting and data analysis.

publications_melted= pd.melt(publications, id_vars=['title', 'journal', 'date', 'DOI', 'first_author', 'institute', 'Abstract', 'Keywords'], 
                             value_vars= publications.columns[7:-2],
                             var_name='publishing_country',
                             value_name='value',)
publications_melted.head()
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
      <th>journal</th>
      <th>date</th>
      <th>DOI</th>
      <th>first_author</th>
      <th>institute</th>
      <th>Abstract</th>
      <th>Keywords</th>
      <th>publishing_country</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Machine Learning and Deep Learning in Medical ...</td>
      <td>J Med Imaging Radiat Sci</td>
      <td>2019-12-01</td>
      <td>doi: 10.1016/j.jmir.2019.09.005.</td>
      <td>Ran Klein</td>
      <td>Charles Sturt University, NSW, Australia. Ele...</td>
      <td>Artificial intelligence (AI) in medical imagin...</td>
      <td>Medical imaging; artificia...</td>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Machine and deep learning methods for radiomics</td>
      <td>Med Phys</td>
      <td>2020-06-01</td>
      <td>doi: 10.1002/mp.13678.</td>
      <td>Issam El Naqa</td>
      <td>Department of Medical Physics, Centro di Rife...</td>
      <td>Radiomics is an emerging area in quantitative ...</td>
      <td>deep learning; machine lea...</td>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Introduction to Machine Learning, Neural Netwo...</td>
      <td>Transl Vis Sci Technol</td>
      <td>2020-02-27</td>
      <td>doi: 10.1167/tvst.9.2.14.</td>
      <td>J Peter Campbell</td>
      <td>Department of Ophthalmology, Casey Eye Instit...</td>
      <td>Purpose:                    To present an over...</td>
      <td>artificial intelligence; d...</td>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Deep Learning in Medical Image Analysis</td>
      <td>Adv Exp Med Biol</td>
      <td>2020-01-01</td>
      <td>doi: 10.1007/978-3-030-33128-3_1.</td>
      <td>Chuan Zhou</td>
      <td>Department of Radiology, University of Michig...</td>
      <td>Deep learning is the state-of-the-art machine ...</td>
      <td>Artificial intelligence; B...</td>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Diagnosis of COVID-19 Using Machine Learning a...</td>
      <td>Curr Med Imaging</td>
      <td>2021-01-01</td>
      <td>doi: 10.2174/1573405617666210713113439.</td>
      <td>Prajoy Podder</td>
      <td>Institute of ICT, Bangladesh University of En...</td>
      <td>Background:                    This paper prov...</td>
      <td>Artificial intelligence; C...</td>
      <td>Afghanistan</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

``` python
#Let's remove the entries where publishing_country equals zero (this entries has no information values anyways). Once we are done with that we would drop the value column and reset the index as well. 

publications_melted= publications_melted.drop(publications_melted[publications_melted['value'] == 0].index).reset_index().drop(['value', 'index'], axis= 1)
publications_melted.head()
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
      <th>journal</th>
      <th>date</th>
      <th>DOI</th>
      <th>first_author</th>
      <th>institute</th>
      <th>Abstract</th>
      <th>Keywords</th>
      <th>publishing_country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Large-Scale Textual Datasets and Deep Learning...</td>
      <td>Comput Intell Neurosci</td>
      <td>2022-04-12</td>
      <td>doi: 10.1155/2022/5731532.</td>
      <td>Fardin Ahmadi</td>
      <td>Computer Science &amp; Engineering, Lloyd Institu...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Applications of Deep Learning and Reinforcemen...</td>
      <td>IEEE Trans Neural Netw Learn Syst</td>
      <td>2018-06-01</td>
      <td>doi: 10.1109/TNNLS.2018.2790388.</td>
      <td>Stefano Vassanelli</td>
      <td>Computer Science &amp; Engineering, Lloyd Institu...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Detection of Types of Mental Illness through t...</td>
      <td>Comput Intell Neurosci</td>
      <td>2022-03-26</td>
      <td>doi: 10.1155/2022/9404242.</td>
      <td>Asadullah Jalali</td>
      <td>Department of Information Systems, College of...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A sensitivity analysis of probability maps in ...</td>
      <td>J Appl Clin Med Phys</td>
      <td>2021-08-01</td>
      <td>doi: 10.1002/acm2.13331.</td>
      <td>Mohamad Fakhreddine</td>
      <td>Department of Radiation Oncology, UT Health S...</td>
      <td>Purpose:                    Deep-learning-base...</td>
      <td>deep learning; machine lea...</td>
      <td>Anguilla</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Federated Deep Learning for the Diagnosis of C...</td>
      <td>IEEE Trans Neural Syst Rehabil Eng</td>
      <td>2022-01-01</td>
      <td>doi: 10.1109/TNSRE.2022.3161272.</td>
      <td>Melissa Roberts</td>
      <td>Department of Radiation Oncology, UT Health S...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Anguilla</td>
    </tr>
  </tbody>
</table>
</div>

## Save final dataframe to parquet

``` python
#Save the dataframe to parquet file
publications_melted.to_parquet('publications_melted.parquet.gzip',compression='gzip')
```

``` python
''' It would be interesting to take a look at the most frequently used keywords. Since one article can have multiple keywords we would need convert the strings containing multiple words into multiple strings. 
For this we will use the pd.str.split method based on ';', but first we should convert series of lists to one series.'''

keywords= publications_melted['Keywords'].apply(pd.Series).stack(dropna= True).reset_index(drop = True)
keywords= keywords.dropna().reset_index(drop= True)
```

``` python
''' The next bit of code is quite a bit involving: first, we split the strings based on ';' and expand the resulting values, then convert the list to a numpy array and flatten it. Moreover, since similar keywords can appear in
multiple forms, such as Machine learning and machine learning, it is important to convert the letters to lower case. In addition, we would also like to remove any remaining characters that are not letters or numbers. 
Finally, we display the frequency counts of the top 25 entries.
'''

keywords= pd.Series(keywords.str.split(';', expand= True).to_numpy().flatten()).str.lower().replace(r'[^a-zA-Z0-9-]', '', regex= True).value_counts(normalize=True).head(25)
keywords.head()
```

    deeplearning                   0.124954
    machinelearning                0.029410
    artificialintelligence         0.023748
    convolutionalneuralnetwork     0.014417
    convolutionalneuralnetworks    0.009725
    dtype: float64

``` python
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.ticker as mtick


fig1, axes1= plt.subplots(figsize= (10,10), nrows= 2, ncols= 2, dpi= 200, squeeze= True)

axes1[0,0].bar(x= publications_melted.publishing_country.value_counts(normalize= True).head(25).index, 
             height= (publications_melted.publishing_country.value_counts(normalize= True).head(25).values)*100)
axes1[0,0].set_title('Top 25 countries most involved in ML/DL research')
axes1[0,0].set_xlabel('')
axes1[0,0].set_ylabel('% articles published')
axes1[0,0].tick_params(axis='x', rotation=90)
axes1[0,0].yaxis.set_major_formatter(mtick.PercentFormatter())

axes1[0,1].bar(x= publications_melted.journal.value_counts(normalize= True).head(25).index, 
             height= (publications_melted.journal.value_counts(normalize= True).head(25).values)*100)
axes1[0,1].set_title('Top 25 journals, where most ML/DL research is published')
axes1[0,1].set_xlabel('')
axes1[0,1].set_ylabel('% articles published')
axes1[0,1].tick_params(axis='x', rotation=90)
axes1[0,1].yaxis.set_major_formatter(mtick.PercentFormatter())


axes1[1,0].bar(x= keywords.index, 
             height= (keywords.values)*100)
axes1[1,0].set_title('Top 25 keywords mentioned in the articles')
axes1[1,0].set_xlabel('')
axes1[1,0].set_ylabel('% articles published')
axes1[1,0].tick_params(axis='x', rotation=90)
axes1[1,0].yaxis.set_major_formatter(mtick.PercentFormatter())


axes1[1,1].plot(publications_melted['date'].dt.year.value_counts().sort_index().index,
              publications_melted['date'].dt.year.value_counts().sort_index().values
             )
axes1[1,1].set_title('Number of articles publised in the field of ML/DL per year')
axes1[1,1].set_xlabel('')
axes1[1,1].set_ylabel('Number of articles published')

fig1.suptitle('Summary of ML/DL research articles published', fontsize=16, y= 1.01)
plt.figtext(0.5, 0.975, "Due to limitations only 10,000 out of the available 22,483 articles were included", ha="center", fontsize=8)
plt.figtext(0.9, 0.01, "Contains articles up to and including 08/08/2022", ha="center", fontsize=8)

plt.tight_layout() #w_pad=0.5, h_pad=1.0
```

![](220808_Webscraping_files/figure-gfm/cell-31-output-1.png)

## To sum up:

-   Based on the figures we can arrive at the conclusion that
    universities and institutions located in China and the US are the
    main contributors to the ML/DL field
-   There are various journals one can follow to keep up-to-date about
    the newest research topics. One of the most popular among scientists
    seems to be the [Sensors open access
    journal](https://www.mdpi.com/journal/sensors)
-   Deep learning seems to be dominating the research area, amounting to
    approximately 12% of all the articles published up to date
-   The number of articles related to ML/DL are steadily rising from
    2016 indicating a growing interest in the field
