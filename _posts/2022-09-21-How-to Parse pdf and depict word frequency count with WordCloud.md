How-to: Parse pdf and depict word frequency count with WordCloud
================

``` python
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
```

**Aim:**

-   demonstrate how easy it is to parse pdf documents with python
-   plot word frequency count in a unique and interesting way using
    wordclouds

Today’s blog post will be fairly short and written in a format I would
like to call as ’How-to’s. The idea behind the ’How-to’s is to highlight
some awesome python libraries we could all benefit from. ’How-to’s are
not intended to be a deep-dive into the nitty gritties of the packages
but more like designed to whet our appetite so that we want to get to
know these libraries better.

Anyways, one of the things I would like to get into more is natural
language processing, especially research articles. To start working on
this, it is kind of necessary to be able to access such data. Since most
of the research articles uploaded on the internet are in pdf format we
would need to find a way to access them.

Here comes the [PdfReader
library](https://pdfreader.readthedocs.io/en/latest/index.html).
PdfReader is a trully fantastic tool that allows us to:

-   Extract texts (plain and formatted)
-   Extract forms data (plain and formatted)
-   Extract images and image masks as Pillow/PIL Images
-   and so much more

Let’s test PdfReader on an
[article](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8401202/) I was
personally involved in.

``` python
reader = PdfReader("s12967-021-03035-6.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text(orientations = 0) + "\n"

text[:1000] # the first 1000 character of the parsed document
```

    'Nkiliza\xa0et\xa0al. J Transl Med          (2021) 19:370  https://doi.org/10.1186/s12967-021-03035-6\nRESEARCH\nSex-specific plasma lipid profiles of\xa0ME/CFS patients and\xa0their association with\xa0pain, fatigue, and\xa0cognitive symptoms\nAurore Nkiliza1,2* , Megan Parks1,2, Adam Cseresznye1,2, Sarah Oberlin1,2, James E. Evans1,2, Teresa Darcey1,2, Kristina Aenlle\n3, Daniel Niedospial1,2, Michael Mullan1,2, Fiona Crawford1,2, Nancy Klimas3 and Laila Abdullah1,2 \nAbstract \nBackground: Myalgic encephalomyelitis/chronic fatigue syndrome (ME/CFS) is a complex illness which dispropor - tionally affects females. This illness is associated with immune and metabolic perturbations that may be influenced by lipid metabolism. We therefore hypothesized that plasma lipids from ME/CFS patients will provide a unique biomarker signature of disturbances in immune, inflammation and metabolic processes associated with ME/CFS.Methods: Lipidomic analyses were performed on plasma from a cohort of 50 ME/CFS patients and 50 '

What if we would like to know the top n most frequent words in the
document? We could write a function that counts the occurrence of each
words and stores them in a dictionary. Like so:

``` python
def frequency_counter(sentence):
    sentence = sentence.lower()
    frequency_dict = dict()
    for idx, item in enumerate(sentence.split()):
        if item in frequency_dict:
            frequency_dict[item] += 1
        else:
            frequency_dict[item] = 1
    sorted_list = sorted(frequency_dict.items(), key=lambda x: x[1], reverse = True)
    return sorted_list

def weighted_frequency_counter(sorted_list):
    new_dict = dict()
    max_value = sorted_list[0][1]
    for item in sorted_list:
        new_dict[item[0]] = item[1] / max_value
    return new_dict
```

``` python
freq = frequency_counter(text)
normalized_freq = weighted_frequency_counter(freq)
```

``` python
fig, ax = plt.subplots(1,1, figsize = (8,8), dpi = 100)

plot = 50

sns.barplot(y = pd.DataFrame(freq)[0][:plot],
            x = pd.DataFrame(freq)[1][:plot], ax = ax)

ax.set_ylabel('')
ax.set_xlabel('')
ax.set_title(f"Top {plot} words according to occurrence")
```

    Text(0.5, 1.0, 'Top 50 words according to occurrence')

![](2022-09-21-pdf-wordcloud_files/figure-gfm/cell-6-output-2.png)

This is not too bad. As expected words such as ‘and’, ‘the’, ‘in’, ‘to’,
etc are heavily used in the text. Apart from that, we have some symbols
slipping in the result, such as ‘-’ and some single letters, too. What
if there was a better way to, a more artistic way to represent such
data? Let’s bring in
[WordCloud](https://amueller.github.io/word_cloud/index.html):

``` python
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["sp", "car", "file", "figure", "j", "page", "B"]) # we can add some words we do not want to display

# Generate a word cloud image
plt.figure(figsize = (10,10), dpi = 300,)
wordcloud = WordCloud(stopwords=stopwords, 
                      background_color="white",
                      width=1000, height=1000).generate(text)
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.tight_layout(pad=0)
plt.axis("off")
```

![](2022-09-21-pdf-wordcloud_files/figure-gfm/cell-7-output-1.png)

As you can see, with a few lines of code, WordCloud makes it incredibly
easy to picture a visually appealing summary of our document. Not only
that, we can also create image colored wordclouds that look just as
incredible. Take a look at this:

``` python
stopwords = set(STOPWORDS)
stopwords.update(["sp", "car", "file", "figure", "j", "page", "B", 'et al', 'M'])

mask  = np.array(Image.open("brain.jpg").convert('RGB'))
wc = WordCloud(stopwords=stopwords, 
               background_color="white", 
               mode="RGB", 
               max_words=1000, 
               width=1600, 
               height=800,
               scale = 5,
               mask=mask)
wc.generate(text)
image_colors = ImageColorGenerator(mask)

fig, ax = plt.subplots(1,2, figsize = (8,3), dpi = 300)

ax[0].imshow(mask)
ax[0].axis("off")
ax[0].set_title("Original image")

ax[1].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
ax[1].axis("off")
ax[1].set_title("Wordcloud image")
plt.tight_layout(pad=0)
```

![](2022-09-21-pdf-wordcloud_files/figure-gfm/cell-8-output-1.png)

I think enough’s said. If you like these libraries, please go ahead and
explore them. They are great fun :)
