Achieving 99% recall for pneumonia detection using transfer learning
================

**Aim :**

-   Use transfer learning to detect pneumonia with 500 images in total
    for training and 100 for validation

**Background :**

Chest X-ray images are frequently used for diagnosis of various medical
conditions, including pneumonia. Pneumonia is an infection of the lungs
that can be caused by bacteria, viruses, or other organisms, and it is a
leading cause of morbidity and mortality worldwide. In this article, we
will discuss how transfer learning can be used to analyze chest X-ray
images for the detection of pneumonia. Transfer learning is a machine
learning technique that involves the repurposing of a model pre-trained
on a large dataset and then fine-tuned for a specific task. In this
case, detecting signs of pneumonia.

This approach has been found to be particularly effective for image
classification tasks, and it can significantly reduce the amount of
labeled data and computational resources required for training. By using
transfer learning, we can leverage the knowledge learned from previous
tasks to improve the performance of our pneumonia detection model.

# Import libraries

``` python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torchvision
import torch 
from torch import nn
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import importlib.util
from google.colab import drive, files
from pathlib import Path
import os
from PIL import Image
import random
import pathlib
import gc

import time
import copy
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
```

# Which pretrained model to use?

In order to choose a suitable pre-trained model for our task, in terms
of size and learning speed, we compare all the currently available
models that have been trained for classification tasks on
[TorchVision](https://pytorch.org/vision/stable/models.html#classification).
Based on the Acc@1 and number of Params, the EfficientNet_V2_S seems to
be an appropriate starting point for this project.

Please note, we do not conduct an exhaustive comparison and evaluation
of all these models. This may be performed as part of a future
experiment.

``` python
link = 'https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights'


pretrained = (pd.read_html(link)[1].loc[:,:'Params']
              .assign(
                  Params = lambda df_: df_['Params'].str.replace('M', '').
                  astype(float),
                  Acc1_Params = lambda df_: df_['Acc@1'] / df_['Params'],
                  Acc5_Params = lambda df_: df_['Acc@5'] / df_['Params'],
                  category = lambda df_: df_.Weight.str.split(pat = '_',expand = True)[0].
                  str.replace('\d+', '', regex = True),
                  subcategory = lambda df_: df_.Weight.str.split(pat = '_',expand = True)[1],
                  )
              .sort_values(by = 'Acc1_Params', ascending=False)
              )
pretrained.head()
```

  <div id="df-e1339b15-dc7e-4afd-8874-5adbc5f2d618">
    <div class="colab-df-container">
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
      <th>Weight</th>
      <th>Acc@1</th>
      <th>Acc@5</th>
      <th>Params</th>
      <th>Acc1_Params</th>
      <th>Acc5_Params</th>
      <th>category</th>
      <th>subcategory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85</th>
      <td>SqueezeNet1_1_Weights.IMAGENET1K_V1</td>
      <td>58.178</td>
      <td>80.624</td>
      <td>1.2</td>
      <td>48.481667</td>
      <td>67.186667</td>
      <td>SqueezeNet</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84</th>
      <td>SqueezeNet1_0_Weights.IMAGENET1K_V1</td>
      <td>58.092</td>
      <td>80.420</td>
      <td>1.2</td>
      <td>48.410000</td>
      <td>67.016667</td>
      <td>SqueezeNet</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1</td>
      <td>60.552</td>
      <td>81.746</td>
      <td>1.4</td>
      <td>43.251429</td>
      <td>58.390000</td>
      <td>ShuffleNet</td>
      <td>V2</td>
    </tr>
    <tr>
      <th>23</th>
      <td>MNASNet0_5_Weights.IMAGENET1K_V1</td>
      <td>67.734</td>
      <td>87.490</td>
      <td>2.2</td>
      <td>30.788182</td>
      <td>39.768182</td>
      <td>MNASNet</td>
      <td>5</td>
    </tr>
    <tr>
      <th>81</th>
      <td>ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1</td>
      <td>69.362</td>
      <td>88.316</td>
      <td>2.3</td>
      <td>30.157391</td>
      <td>38.398261</td>
      <td>ShuffleNet</td>
      <td>V2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e1339b15-dc7e-4afd-8874-5adbc5f2d618')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e1339b15-dc7e-4afd-8874-5adbc5f2d618 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e1339b15-dc7e-4afd-8874-5adbc5f2d618');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  

``` python
meanpointprops = dict(linestyle='--', linewidth=2.5, color='purple')

plt.style.use('default')
sns.set_context(context= 'talk', font_scale=0.6)
fig, axs = plt.subplots(1,2, figsize = (10,5
                                       ))

labels = pretrained.groupby('category').median().sort_values(by = 'Acc1_Params').index

g = sns.boxplot(data = pretrained, 
            x = 'category', 
            y = 'Acc1_Params',
            order = pretrained.groupby('category').median().sort_values(by = 'Acc1_Params').index,
            flierprops={"marker": "o"},
            medianprops={"color": "red"},
            meanprops=meanpointprops, 
            meanline=True,
            showmeans=True,
            linewidth = 2,
            ax = axs[0]).set_xticklabels(rotation=90, labels = labels)

sns.pointplot(x = pretrained.groupby('category').median().sort_values(by = 'Acc1_Params')['Acc@1'].index,
              y = pretrained.groupby('category').median().sort_values(by = 'Acc1_Params')['Acc@1'].values,
              order = pretrained.groupby('category').median().sort_values(by = 'Acc1_Params').index,
              scale = 0.5,
              linestyles = '--',
              color = 'k',
             ax = axs[0]
             ).set(xlabel = '', title = 'Efficiencies and max reported Acc@1s of pretrained models')

axs[0].hlines(y = 70, xmin = 0,
           xmax = pretrained.groupby('category').median().sort_values(by = 'Acc1_Params')['Acc@1'].index.shape[0] -1,
           colors= 'red',
           linestyles='--',
          )

sns.scatterplot(x = pretrained[pretrained.category == 'EfficientNet']['Params'],
                y = pretrained[pretrained.category == 'EfficientNet']['Acc@1'],
                ax = axs[1]
               ).set(title = 'Acc@1s of pretrained EfficientNet models')

plt.tight_layout()

plt.savefig('comparison.png', dpi = 300)
```

![](Copy_of_XRay_transfer_learning_files/figure-gfm/cell-4-output-1.png)

``` python
EfficientNets = pretrained[pretrained.category == 'EfficientNet'].sort_values('Acc1_Params', ascending = False)
EfficientNets.query('`Acc@1` > 84 & Params < 25')
```

  <div id="df-e5306838-3fb4-4892-aec0-ea5eb5b252bf">
    <div class="colab-df-container">
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
      <th>Weight</th>
      <th>Acc@1</th>
      <th>Acc@5</th>
      <th>Params</th>
      <th>Acc1_Params</th>
      <th>Acc5_Params</th>
      <th>category</th>
      <th>subcategory</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>EfficientNet_V2_S_Weights.IMAGENET1K_V1</td>
      <td>84.228</td>
      <td>96.878</td>
      <td>21.5</td>
      <td>3.917581</td>
      <td>4.505953</td>
      <td>EfficientNet</td>
      <td>V2</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-e5306838-3fb4-4892-aec0-ea5eb5b252bf')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">
        
  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>
      
  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-e5306838-3fb4-4892-aec0-ea5eb5b252bf button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-e5306838-3fb4-4892-aec0-ea5eb5b252bf');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>
  

### Checking version of torch and torchvision and un/reinstall them if needed

``` python
if int(torchvision.__version__.split(sep = '.')[1]) < 13:
    !conda uninstall pytorch
    !pip uninstall torch --yes
    !pip uninstall torch --yes# run this command twice
    
    !conda uninstall torchvision
    !pip uninstall torchvision --yes
    !pip uninstall torchvision --yes # run this command twice
    
    !conda install --yes pytorch torchvision
    import torch 
    import torchvision
    print(f'Current version of torch: {torch.__version__}')
    print(f'Current version of torchvision: {torchvision.__version__}')
    
else:
    import torch 
    import torchvision
    print(f'Current version of torch: {torch.__version__}')
    print(f'Current version of torchvision: {torchvision.__version__}')
```

    Current version of torch: 1.13.0+cu116
    Current version of torchvision: 0.14.0+cu116

``` python
if importlib.util.find_spec('torchinfo') is None:
  print('torchinfo' +" is not installed")
  !pip install torchinfo
  from torchinfo import summary
  from tqdm.auto import tqdm
else:
  from torchinfo import summary
  from tqdm.auto import tqdm
```

    torchinfo is not installed
    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting torchinfo
      Downloading torchinfo-1.7.1-py3-none-any.whl (22 kB)
    Installing collected packages: torchinfo
    Successfully installed torchinfo-1.7.1

``` python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
  print(torch.cuda.get_device_name(0))
device
```

    Tesla T4

    'cuda'

### Getting the data from Kaggle

We’ll use the dataset by Paul Mooney, called Chest X-Ray Images
(Pneumonia) from
[Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
Since this notebook was created using Google Colab, we first need to
connect Colab with our Kaggle account. The steps are down below:

``` python
# steps provided by VARSHA: https://www.kaggle.com/general/156610

drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive

``` python
files.upload() #this will prompt you to upload the kaggle.json
```

     <input type="file" id="files-54ecb9cd-8d0a-42fe-bcb6-169a7b6b99af" name="files[]" multiple disabled
        style="border:none" />
     <output id="result-54ecb9cd-8d0a-42fe-bcb6-169a7b6b99af">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script>// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @fileoverview Helpers for google.colab Python module.
 */
(function(scope) {
function span(text, styleAttributes = {}) {
  const element = document.createElement('span');
  element.textContent = text;
  for (const key of Object.keys(styleAttributes)) {
    element.style[key] = styleAttributes[key];
  }
  return element;
}

// Max number of bytes which will be uploaded at a time.
const MAX_PAYLOAD_SIZE = 100 * 1024;

function _uploadFiles(inputId, outputId) {
  const steps = uploadFilesStep(inputId, outputId);
  const outputElement = document.getElementById(outputId);
  // Cache steps on the outputElement to make it available for the next call
  // to uploadFilesContinue from Python.
  outputElement.steps = steps;

  return _uploadFilesContinue(outputId);
}

// This is roughly an async generator (not supported in the browser yet),
// where there are multiple asynchronous steps and the Python side is going
// to poll for completion of each step.
// This uses a Promise to block the python side on completion of each step,
// then passes the result of the previous step as the input to the next step.
function _uploadFilesContinue(outputId) {
  const outputElement = document.getElementById(outputId);
  const steps = outputElement.steps;

  const next = steps.next(outputElement.lastPromiseValue);
  return Promise.resolve(next.value.promise).then((value) => {
    // Cache the last promise value to make it available to the next
    // step of the generator.
    outputElement.lastPromiseValue = value;
    return next.value.response;
  });
}

/**
 * Generator function which is called between each async step of the upload
 * process.
 * @param {string} inputId Element ID of the input file picker element.
 * @param {string} outputId Element ID of the output display.
 * @return {!Iterable<!Object>} Iterable of next steps.
 */
function* uploadFilesStep(inputId, outputId) {
  const inputElement = document.getElementById(inputId);
  inputElement.disabled = false;

  const outputElement = document.getElementById(outputId);
  outputElement.innerHTML = '';

  const pickedPromise = new Promise((resolve) => {
    inputElement.addEventListener('change', (e) => {
      resolve(e.target.files);
    });
  });

  const cancel = document.createElement('button');
  inputElement.parentElement.appendChild(cancel);
  cancel.textContent = 'Cancel upload';
  const cancelPromise = new Promise((resolve) => {
    cancel.onclick = () => {
      resolve(null);
    };
  });

  // Wait for the user to pick the files.
  const files = yield {
    promise: Promise.race([pickedPromise, cancelPromise]),
    response: {
      action: 'starting',
    }
  };

  cancel.remove();

  // Disable the input element since further picks are not allowed.
  inputElement.disabled = true;

  if (!files) {
    return {
      response: {
        action: 'complete',
      }
    };
  }

  for (const file of files) {
    const li = document.createElement('li');
    li.append(span(file.name, {fontWeight: 'bold'}));
    li.append(span(
        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +
        `last modified: ${
            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :
                                    'n/a'} - `));
    const percent = span('0% done');
    li.appendChild(percent);

    outputElement.appendChild(li);

    const fileDataPromise = new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        resolve(e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
    // Wait for the data to be ready.
    let fileData = yield {
      promise: fileDataPromise,
      response: {
        action: 'continue',
      }
    };

    // Use a chunked sending to avoid message size limits. See b/62115660.
    let position = 0;
    do {
      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);
      const chunk = new Uint8Array(fileData, position, length);
      position += length;

      const base64 = btoa(String.fromCharCode.apply(null, chunk));
      yield {
        response: {
          action: 'append',
          file: file.name,
          data: base64,
        },
      };

      let percentDone = fileData.byteLength === 0 ?
          100 :
          Math.round((position / fileData.byteLength) * 100);
      percent.textContent = `${percentDone}% done`;

    } while (position < fileData.byteLength);
  }

  // All done.
  yield {
    response: {
      action: 'complete',
    }
  };
}

scope.google = scope.google || {};
scope.google.colab = scope.google.colab || {};
scope.google.colab._files = {
  _uploadFiles,
  _uploadFilesContinue,
};
})(self);
</script> 

    Saving kaggle.json to kaggle.json

    {'kaggle.json': b'{"username":"unworried1686","key":"cb7a7d15651846534b582961b1d3d54d"}'}

``` python
!pip install -q kaggle
```

``` python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
```

``` python
!chmod 600 /root/.kaggle/kaggle.json
```

``` python
!pwd
```

    /content

``` python
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

    Downloading chest-xray-pneumonia.zip to /content
    100% 2.28G/2.29G [00:12<00:00, 231MB/s]
    100% 2.29G/2.29G [00:12<00:00, 193MB/s]

``` python
!unzip -q chest-xray-pneumonia.zip
```

``` python
# taking a look at a random image

img_path = Path('/content/chest_xray/chest_xray')

train_dir = img_path / 'train'
test_dir = img_path / 'test'
val_dir = img_path / 'val'

image_path_list = list(img_path.glob("*/*/*.jpeg"))
random_image_path = random.choice(image_path_list)
random_image_path_class = random_image_path.parent.stem


img = Image.open(random_image_path)

print(f'Image class: {random_image_path_class}')
print(f'Image height: {img.height}')
print(f'Image width: {img.width}')

img.resize((300, 300)).save('example.png')
```

    Image class: PNEUMONIA
    Image height: 1768
    Image width: 1912

``` python
print('Number of items in the train directory:')
for pth in train_dir.iterdir():
  print(len(list(pth.glob("*.jpeg"))))

print('Number of items in the test directory:')
for pth in test_dir.iterdir():
    print(len(list(pth.glob("*.jpeg"))))

print('Number of items in the validation directory:')
for pth in val_dir.iterdir():
    print(len(list(pth.glob("*.jpeg"))))
```

    Number of items in the train directory:
    1341
    3875
    0
    Number of items in the test directory:
    234
    390
    0
    Number of items in the validation directory:
    8
    8
    0

### Assigning indices for the training and validation samples

As you can see, we have over 5000 images at our disposal. However, to
see what we can do with only a fraction of it, we will limit ourselves
to using only 250 images per class (this is less than 10% of the
original dataset). In addition, to obtain a more reliable validation
accuracy, we will increase the number validation samples from 8 to 50
images per class.

``` python
random.seed(0)

train_data = datasets.ImageFolder(root = train_dir,
                                  transform = None,
                                  target_transform = None,
                                  )

# select indices for training images:

NUMBER_OF_TRANINING_SAMPLES = 250
NUMBER_OF_VALIDATION_SAMPLES = 50

train_zeros = random.sample(range(0,train_data.targets.index(1)), NUMBER_OF_TRANINING_SAMPLES)
train_ones = random.sample(range(train_data.targets.index(1), len(train_data.targets)), NUMBER_OF_TRANINING_SAMPLES)

train_indices = [*train_zeros, *train_ones]

# Locate the indices that are not selected for the training samples
remaining_zeros = set(list(range(0,train_data.targets.index(1)))) - set(train_zeros)
remaining_ones = set(list(range(train_data.targets.index(1), len(train_data.targets)))) - set(train_ones)

# Select indices for validation samples

val_zeros = random.sample(remaining_zeros, NUMBER_OF_VALIDATION_SAMPLES)
val_ones = random.sample(remaining_ones, NUMBER_OF_VALIDATION_SAMPLES)

val_indices = [*val_zeros, *val_ones]

print('SANITY CHECK:')
print(f'  Length of training samples : {len(train_indices)}')
print(f'      Length of zeros : {len(train_zeros)}')
print(f'      Length of ones : {len(train_ones)}')

print(f'  Length of validation samples : {len(val_indices)}')
print(f'      Length of zeros : {len(val_zeros)}')
print(f'      Length of ones : {len(val_ones)}')
```

    SANITY CHECK:
      Length of training samples : 500
          Length of zeros : 250
          Length of ones : 250
      Length of validation samples : 100
          Length of zeros : 50
          Length of ones : 50

``` python
# check if the indices in the validation subset are unique

any(item in train_indices for item in val_indices)
```

    False

We resampled our training and validation images from over 5000 to 250
per class for the training dataset and 50 per class for the validation
dataset. Additionally, we checked that all the samples are unique,
meaning no data leakage can happen during validation.

### Checking out some of the training images

``` python
loaded_samples = datasets.ImageFolder(root=train_dir,
                                      transform= ToTensor(),
                                      ) 

loaded_samples_subset = torch.utils.data.Subset(loaded_samples, train_indices)
```

``` python
fig = plt.figure(figsize=(9, 9))
rows, cols = 3, 3

for i in range(1, rows * cols + 1):
    fig.add_subplot(rows, cols, i)
    i = random.randint(0, len(loaded_samples_subset))
    plt.imshow(loaded_samples_subset[i][0].permute(1,2,0))
    plt.title(''.join([item for item, key in train_data.class_to_idx.items() if key == loaded_samples_subset[i][1]]))
    plt.axis(False)

plt.savefig('example_grid.png', dpi = 300)
```

![](Copy_of_XRay_transfer_learning_files/figure-gfm/cell-22-output-1.png)

As you can see, for someone inexperienced in this field, it can be
difficult to tell the difference between healthy individuals and
patients with pneumonia. With the help of machine learning, maybe we can
improve our abilities in this area.

### Define functions

Let’s create some functions that we will use throughout this notebook.

#### Train step

``` python
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device,
               use_cuda=True):
  
  ''' This function performs a training step on a given model.

  The function first sets the model to training mode, and then iterates through
   the data in data_loader. For each batch of data, it sends the data to the 
   specified device. The data is then passed through the model to get the logits. 
   The sigmoid function is applied to the logits to get the predicted probabilities.
   
   The gradients are then zeroed out, the loss is backpropagated, and the
    optimizer takes a step to update the model's parameters. The loss and 
    accuracy are then accumulated over all the batches. Finally, the average 
    loss and accuracy are calculated and returned.

  Args:
  model: a PyTorch module. This is the model that we want to train
  data_loader: a PyTorch DataLoader. This is used to load the data for the 
  training step
  loss_fn: a PyTorch module. This is the loss function that we want to use to 
  calculate the loss of the model
  optimizer: a PyTorch optimizer. This is the optimizer that we want to use to 
  update the model's parameters
  device: a PyTorch device. This specifies the device that we want to use to 
  run the model (e.g. 'cpu' or 'cuda'). If none is provided, the default device 
  specified in the global variable device will be used
  use_cuda: a boolean. If True, this specifies that we want to use a GPU to 
  calculate the mixed data and labels. The default value is True.

  Returns:
  Average loss and accuracy of the training set are returned.

  '''
  train_loss, train_acc = 0, 0
  model.train()
  for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        y_pred_logits = model(X).squeeze()
        y_pred = torch.sigmoid(y_pred_logits)

        loss = loss_fn(y_pred, y.float())
    
        train_loss += loss.item() 
    
          # 3. Optimizer zero grad
        optimizer.zero_grad()
    
          # 4. Loss backward
        loss.backward()

      # 5. Optimizer step
        optimizer.step()
    
        y_pred_class = torch.round(y_pred)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    
  train_loss = train_loss / len(train_dataloader)
  train_acc = train_acc / len(train_dataloader)
        
#     print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")    
  return train_loss, train_acc
```

#### Validation step

``` python
def val_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device = device):
  
  ''' This function performs validation on a given model

  The function sets the model to evaluation mode, and then iterates through the
   data in data_loader. For each batch of data, it sends the data to the specified 
   device, and then passes it through the model to get the logits. It then 
   applies the sigmoid function to the logits to get the predicted probabilities. 
   The loss is then calculated using the loss_fn and the predicted probabilities 
   and the true labels. The loss and accuracy are then accumulated over all the 
   batches. Finally, the average loss and accuracy are calculated and returned.

   Args:

   model: a PyTorch module. This is the model that we want to validate.
   data_loader: a PyTorch DataLoader. This is used to load data for the validation 
   step.
   loss_fn: a PyTorch module. This is the loss function that we want to use to
   calculate the loss of the model.
   device: a PyTorch device. This specifies the device that we want to use to 
   run the model (e.g. 'cpu' or 'cuda'). If none is provided, the default device 
   specified in the global variable device will be used.

   Returns:

   Average loss and accuracy of the validation set are returned.

  '''
    
  val_loss, val_acc = 0, 0
  model.eval()
  with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            val_pred_logits = model(X).squeeze()
            val_pred = torch.sigmoid(val_pred_logits)
            
            loss = loss_fn(val_pred, y.float())
            val_loss += loss.item()
            
            val_pred_labels = torch.round(val_pred)
            val_acc += (val_pred_labels == y).sum().item()/len(y)
            
  val_loss = val_loss / len(data_loader)
  val_acc = val_acc / len(data_loader)
       
  return val_loss, val_acc
```

#### Plot training

``` python
def plot_training(results:dict):

  ''' This function plots the training and validation losses and accuracies over
   the epochs. 

   The function creates a figure with two subplots: one for the losses and one 
   for the accuracies. It plots the training and validation values for each metric 
   in separate lines. The epochs are used as the x-axis. The resulting plot is 
   then displayed.
   
   Args:
   results: a dictionary containing the following keys:
   'train_loss': a list of floats representing the training losses at each epoch.
   'val_loss': a list of floats representing the validation losses at each epoch.
   'train_acc': a list of floats representing the training accuracies at each epoch.
   'val_acc': a list of floats representing the validation accuracies at each epoch.

   Returns:

   Image depicting training and validation losses and accuracies. 
  '''
    
  fig, axs = plt.subplots(1,2, figsize = (8,4))

  epochs = range(len(results['train_loss']))

  axs[0].plot(epochs, results['train_loss'], label = 'train', ls = '--', marker = 'o')
  axs[0].plot(epochs, results['val_loss'], label = 'validation', marker = 'o')
  axs[0].set_title('Losses')
  axs[0].legend()

  axs[1].plot(epochs, results['train_acc'], label = 'train',ls = '--', marker = 'o')
  axs[1].plot(epochs, results['val_acc'], label = 'validation', marker = 'o')
  axs[1].set_title('Accuracies')
  axs[1].legend()

  return fig
```

#### Fit function

``` python
def fit(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          seed : torch.manual_seed = 42):
  
  ''' This function trains a model for a given number of epochs and performs 
  validation at each epoch. 

  The function initializes some variables to keep track of the best model and 
  its performance, and also a dictionary to store the training and validation 
  losses and accuracies at each epoch. It then loops through the specified number 
  of epochs and calls the train_step and val_step functions to perform training 
  and validation, respectively. The training and validation losses and accuracies 
  are appended to the appropriate lists in the results dictionary. If the
   validation accuracy at the current epoch is better than the best accuracy 
   seen so far, the model's state is saved as the best model. The learning rate 
   is then updated using the scheduler. Finally, the function returns the best 
   model and the results dictionary.

   Args:
   model: a PyTorch module. This is the model that we want to train
   train_dataloader: a PyTorch DataLoader. This is used to load the training data.
   val_dataloader: a PyTorch DataLoader. This is used to load the validation data.
   optimizer: a PyTorch Optimizer. This is the optimizer that will be used to 
   update the model's parameters during training.
   scheduler: a PyTorch learning rate scheduler. This is used to adjust the 
   learning rate of the optimizer over the course of training.
   loss_fn: a PyTorch module. This is the loss function that will be used to 
   calculate the loss of the model.
   epochs: an integer. This is the number of epochs that the model will be trained for.
   device: a PyTorch device. This specifies the device that we want to use to run 
   the model (e.g. 'cpu' or 'cuda').
   seed: an integer (optional). This sets the random seed for PyTorch.

   Returns:
   Best model and the results dictionary.
  '''
  
  best_model = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
               }
    
  EPOCHS = epochs
    
  for epoch in tqdm(range(EPOCHS)):
        train_loss, train_acc = train_step(data_loader=train_dataloader, 
                                       model=model, 
                                       loss_fn=loss_fn,
                                       optimizer=optimizer
                                      )
    
        val_loss, val_acc = val_step(data_loader=val_dataloader,
                                      model=model,
                                      loss_fn=loss_fn
                                     )
    
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)

        if val_acc > best_acc:
           best_acc = val_acc
           best_model = copy.deepcopy(model.state_dict())

        scheduler.step()

        # print(f"Epoch: {epoch} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | LR :  {optimizer.param_groups[0]['lr']}")
        
  model.load_state_dict(best_model)
  return model, results
```

#### Predict and plot image

``` python
def predict_image(model: torch.nn.Module,
                  image_path: str,
                  class_names: Dict,
                  transform: torchvision.transforms,
                  device: torch.device=device):
  
  ''' This function takes in an image and predicts the class of the image using 
  a trained model. 

  The function first opens the image and converts it to RGB. It then applies the 
  transform to the image and sends it to the specified device. It then passes the 
  image through the model to get the prediction logits and converts them to a 
  probability using the sigmoid function. It then rounds the probability to get 
  the predicted label and looks up the corresponding class name using the class 
  names dictionary. 

  Args:

  model: a PyTorch module. This is the trained model that will be used to make 
  the prediction.
  image_path: a string. This is the file path of the image that we want to predict 
  the class of.
  class_names: a dictionary. This maps the integer labels used in the model to the
   corresponding class names.
  transform: a PyTorch transform. This is used to preprocess the image in the same 
  way that the training data was preprocessed.
  device: a PyTorch device. This specifies the device that we want to use to run
   the model (e.g. 'cpu' or 'cuda').
   
   Returns:
   An image, displaying the ground truth, predicted label, and probability of the 
   image being the positive class.
  '''
  
  image = Image.open(image_path)
  color_image = image.convert('RGB')

  transformed_image = transform(color_image)

  with torch.inference_mode():
     transformed_image = transformed_image.to(device)

     pred_logits = model(transformed_image.unsqueeze(dim = 0))
     pred = torch.sigmoid(pred_logits)
     
     pred_labels = int(torch.round(pred).item())

     truth = 'NORMAL' if "NORMAL" in image_path else 'PNEUMONIA'
     predicted_label = list(class_names.keys())[list(class_names.values()).index(pred_labels)]

  plt.figure()
  plt.imshow(transformed_image.cpu().permute(1,2,0))
  plt.title(f"Ground truth: {truth} | Pred: {predicted_label} | Probability of being PNEUMONIA: {pred.item() * 100:.1f}%")
  plt.axis(False);

  return plt
```

#### Predict step

``` python
def predict(model: torch.nn.Module,
                  image_path: str,
                  class_names: Dict,
                  transform: torchvision.transforms,
                  device: torch.device=device):
  
  ''' This function takes in an image and predicts the class of the image using 
  a trained model.

  The function first opens the image and converts it to RGB. It then applies the 
  transform to the image and sends it to the specified device. It then passes the
  image through the model to get the prediction logits and converts them to a 
  probability using the sigmoid function. It then rounds the probability to get
  the predicted label. The function then returns the ground truth label and 
  predicted label. The ground truth label is obtained by checking if "NORMAL" 
  appears in the image file path.

  Args:
  model: a PyTorch module. This is the trained model that will be used to make 
  the prediction.
  image_path: a string. This is the file path of the image that we want to predict 
  the class of.
  class_names: a dictionary. This maps the integer labels used in the model to the 
  corresponding class names.
  transform: a PyTorch transform. This is used to preprocess the image in the same
   way that the training data was preprocessed.
  device: a PyTorch device. This specifies the device that we want to use to run 
  the model (e.g. 'cpu' or 'cuda').

  Returns:
  It returns the ground truth label and predicted label
  '''
  
  image = Image.open(image_path)
  color_image = image.convert('RGB')

  transformed_image = transform(color_image)

  with torch.inference_mode():
     transformed_image = transformed_image.to(device)
     
     pred_logits = model(transformed_image.unsqueeze(dim = 0))
     pred = torch.sigmoid(pred_logits)
     
     pred_label = int(torch.round(pred).item())

     truth = 0 if "NORMAL" in image_path else 1
     return truth, pred_label
```

# Training the EfficientNet

``` python
efficientnet_v2_weights = models.EfficientNet_V2_S_Weights.DEFAULT 
efficientnet_v2_autotransforms = efficientnet_v2_weights.transforms()

efficientnet_v2_model = torchvision.models.efficientnet_v2_s(weights=efficientnet_v2_weights).to(device)

for param in efficientnet_v2_model.parameters():
    param.requires_grad = False
    
efficientnet_v2_model.classifier[1] = torch.nn.Sequential(
    torch.nn.Linear(in_features=1280,
                    out_features=512,
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p = 0.3),
    torch.nn.Linear(in_features=512,
                    out_features=256,
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=256,
                    out_features=1,
                    bias=True),
    ).to(device)

summary(model=efficientnet_v2_model, 
        input_size=(32, 3, 384, 384), 
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
        )
```

    Downloading: "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth" to /root/.cache/torch/hub/checkpoints/efficientnet_v2_s-dd5fe13b.pth

      0%|          | 0.00/82.7M [00:00<?, ?B/s]

    ============================================================================================================================================
    Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
    ============================================================================================================================================
    EfficientNet (EfficientNet)                                  [32, 3, 384, 384]    [32, 1]              --                   Partial
    ├─Sequential (features)                                      [32, 3, 384, 384]    [32, 1280, 12, 12]   --                   False
    │    └─Conv2dNormActivation (0)                              [32, 3, 384, 384]    [32, 24, 192, 192]   --                   False
    │    │    └─Conv2d (0)                                       [32, 3, 384, 384]    [32, 24, 192, 192]   (648)                False
    │    │    └─BatchNorm2d (1)                                  [32, 24, 192, 192]   [32, 24, 192, 192]   (48)                 False
    │    │    └─SiLU (2)                                         [32, 24, 192, 192]   [32, 24, 192, 192]   --                   --
    │    └─Sequential (1)                                        [32, 24, 192, 192]   [32, 24, 192, 192]   --                   False
    │    │    └─FusedMBConv (0)                                  [32, 24, 192, 192]   [32, 24, 192, 192]   (5,232)              False
    │    │    └─FusedMBConv (1)                                  [32, 24, 192, 192]   [32, 24, 192, 192]   (5,232)              False
    │    └─Sequential (2)                                        [32, 24, 192, 192]   [32, 48, 96, 96]     --                   False
    │    │    └─FusedMBConv (0)                                  [32, 24, 192, 192]   [32, 48, 96, 96]     (25,632)             False
    │    │    └─FusedMBConv (1)                                  [32, 48, 96, 96]     [32, 48, 96, 96]     (92,640)             False
    │    │    └─FusedMBConv (2)                                  [32, 48, 96, 96]     [32, 48, 96, 96]     (92,640)             False
    │    │    └─FusedMBConv (3)                                  [32, 48, 96, 96]     [32, 48, 96, 96]     (92,640)             False
    │    └─Sequential (3)                                        [32, 48, 96, 96]     [32, 64, 48, 48]     --                   False
    │    │    └─FusedMBConv (0)                                  [32, 48, 96, 96]     [32, 64, 48, 48]     (95,744)             False
    │    │    └─FusedMBConv (1)                                  [32, 64, 48, 48]     [32, 64, 48, 48]     (164,480)            False
    │    │    └─FusedMBConv (2)                                  [32, 64, 48, 48]     [32, 64, 48, 48]     (164,480)            False
    │    │    └─FusedMBConv (3)                                  [32, 64, 48, 48]     [32, 64, 48, 48]     (164,480)            False
    │    └─Sequential (4)                                        [32, 64, 48, 48]     [32, 128, 24, 24]    --                   False
    │    │    └─MBConv (0)                                       [32, 64, 48, 48]     [32, 128, 24, 24]    (61,200)             False
    │    │    └─MBConv (1)                                       [32, 128, 24, 24]    [32, 128, 24, 24]    (171,296)            False
    │    │    └─MBConv (2)                                       [32, 128, 24, 24]    [32, 128, 24, 24]    (171,296)            False
    │    │    └─MBConv (3)                                       [32, 128, 24, 24]    [32, 128, 24, 24]    (171,296)            False
    │    │    └─MBConv (4)                                       [32, 128, 24, 24]    [32, 128, 24, 24]    (171,296)            False
    │    │    └─MBConv (5)                                       [32, 128, 24, 24]    [32, 128, 24, 24]    (171,296)            False
    │    └─Sequential (5)                                        [32, 128, 24, 24]    [32, 160, 24, 24]    --                   False
    │    │    └─MBConv (0)                                       [32, 128, 24, 24]    [32, 160, 24, 24]    (281,440)            False
    │    │    └─MBConv (1)                                       [32, 160, 24, 24]    [32, 160, 24, 24]    (397,800)            False
    │    │    └─MBConv (2)                                       [32, 160, 24, 24]    [32, 160, 24, 24]    (397,800)            False
    │    │    └─MBConv (3)                                       [32, 160, 24, 24]    [32, 160, 24, 24]    (397,800)            False
    │    │    └─MBConv (4)                                       [32, 160, 24, 24]    [32, 160, 24, 24]    (397,800)            False
    │    │    └─MBConv (5)                                       [32, 160, 24, 24]    [32, 160, 24, 24]    (397,800)            False
    │    │    └─MBConv (6)                                       [32, 160, 24, 24]    [32, 160, 24, 24]    (397,800)            False
    │    │    └─MBConv (7)                                       [32, 160, 24, 24]    [32, 160, 24, 24]    (397,800)            False
    │    │    └─MBConv (8)                                       [32, 160, 24, 24]    [32, 160, 24, 24]    (397,800)            False
    │    └─Sequential (6)                                        [32, 160, 24, 24]    [32, 256, 12, 12]    --                   False
    │    │    └─MBConv (0)                                       [32, 160, 24, 24]    [32, 256, 12, 12]    (490,152)            False
    │    │    └─MBConv (1)                                       [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (2)                                       [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (3)                                       [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (4)                                       [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (5)                                       [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (6)                                       [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (7)                                       [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (8)                                       [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (9)                                       [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (10)                                      [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (11)                                      [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (12)                                      [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (13)                                      [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    │    └─MBConv (14)                                      [32, 256, 12, 12]    [32, 256, 12, 12]    (1,005,120)          False
    │    └─Conv2dNormActivation (7)                              [32, 256, 12, 12]    [32, 1280, 12, 12]   --                   False
    │    │    └─Conv2d (0)                                       [32, 256, 12, 12]    [32, 1280, 12, 12]   (327,680)            False
    │    │    └─BatchNorm2d (1)                                  [32, 1280, 12, 12]   [32, 1280, 12, 12]   (2,560)              False
    │    │    └─SiLU (2)                                         [32, 1280, 12, 12]   [32, 1280, 12, 12]   --                   --
    ├─AdaptiveAvgPool2d (avgpool)                                [32, 1280, 12, 12]   [32, 1280, 1, 1]     --                   --
    ├─Sequential (classifier)                                    [32, 1280]           [32, 1]              --                   True
    │    └─Dropout (0)                                           [32, 1280]           [32, 1280]           --                   --
    │    └─Sequential (1)                                        [32, 1280]           [32, 1]              --                   True
    │    │    └─Linear (0)                                       [32, 1280]           [32, 512]            655,872              True
    │    │    └─ReLU (1)                                         [32, 512]            [32, 512]            --                   --
    │    │    └─Dropout (2)                                      [32, 512]            [32, 512]            --                   --
    │    │    └─Linear (3)                                       [32, 512]            [32, 256]            131,328              True
    │    │    └─ReLU (4)                                         [32, 256]            [32, 256]            --                   --
    │    │    └─Linear (5)                                       [32, 256]            [32, 1]              257                  True
    ============================================================================================================================================
    Total params: 20,964,945
    Trainable params: 787,457
    Non-trainable params: 20,177,488
    Total mult-adds (G): 267.69
    ============================================================================================================================================
    Input size (MB): 56.62
    Forward/backward pass size (MB): 18303.19
    Params size (MB): 83.86
    Estimated Total Size (MB): 18443.67
    ============================================================================================================================================

``` python
# setting up the data transforms including TrivialAugmentWide and RandomErasing

preprocess_train = transforms.Compose([
    transforms.Resize(384),  
    transforms.CenterCrop(384), 
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p = 0.1),
    ])

preprocess_val = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     ])
```

``` python
# taking a look at the transformed images
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
train_data = datasets.ImageFolder(root = train_dir,
                                  transform = preprocess_train,
                                  target_transform = None,
                                  )
  
train_subset = torch.utils.data.Subset(train_data, train_indices)
  
train_dataloader = DataLoader(train_subset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True,
                              )

images, labels = next(iter(train_dataloader))
grid = make_grid(images)

plt.figure(figsize = (15,25))

img = plt.imshow(grid.permute(1, 2, 0)).figure
plt.axis('off')
plt.tight_layout()

img.savefig('transformed_grid.png', dpi = 300)
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

![](Copy_of_XRay_transfer_learning_files/figure-gfm/cell-31-output-2.png)

``` python
start = time.time()

# set seed for reproducibility
os.environ['PYTHONHASHSEED'] = str(1) 
random.seed(0)

torch.cuda.manual_seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
EPOCHS = 20            

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(efficientnet_v2_model.parameters(), 
                            lr= 1e-4, 
                            weight_decay = 2e-5,
                            )
scheduler = CosineAnnealingLR(optimizer,
                              T_max = EPOCHS,
                              eta_min = 0)

# define train and val data
train_data = datasets.ImageFolder(root = train_dir,
                                  transform = preprocess_train,
                                  target_transform = None,
                                  )

val_data = datasets.ImageFolder(root = train_dir,
                                transform = preprocess_val,
                                target_transform = None,
                                )
  
train_subset = torch.utils.data.Subset(train_data, train_indices)
val_subset = torch.utils.data.Subset(val_data, val_indices)
  
train_dataloader = DataLoader(train_subset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True,
                              )
  
val_dataloader = DataLoader(val_subset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=True,
                            )

model, results = fit(model = efficientnet_v2_model,
                   train_dataloader = train_dataloader,
                   val_dataloader = val_dataloader,
                   optimizer = optimizer,
                   scheduler = scheduler,
                   loss_fn = loss_fn,
                   epochs = EPOCHS,
                   device = device)

finish = time.time()
print(f'Time taken to compute: {(finish - start)/60:.3f} min')
```

      0%|          | 0/20 [00:00<?, ?it/s]

    Time taken to compute: 6.027 min

``` python
img = plot_training(results)
img.savefig('train1.png', dpi = 300)

files.download('train1.png')
```

    <IPython.core.display.Javascript object>

    <IPython.core.display.Javascript object>

![](Copy_of_XRay_transfer_learning_files/figure-gfm/cell-33-output-3.png)

``` python
torch.save(model.state_dict(), 'tuned_classifier.pt')
```

### Saving the modified model architecture

``` python
# reload the trained model in a new architecture

model2 = torchvision.models.efficientnet_v2_s(pretrained = False).to(device)
    
model2.classifier[1] = torch.nn.Sequential(
    torch.nn.Linear(in_features=1280,
                    out_features=512,
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(p = 0.3),
    torch.nn.Linear(in_features=512,
                    out_features=256,
                    bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=256,
                    out_features=1,
                    bias=True),
    ).to(device)
PATH = 'tuned_classifier.pt'
model2.load_state_dict(torch.load(PATH))
```

    <All keys matched successfully>

``` python
# check if the saved model is identical to trained one:

for p1, p2 in zip(model.parameters(), model2.parameters()):
  if (p1.data.ne(p2.data).sum() != 0).item():
    print('Models are not identical')
```

## Training the whole model

As a next step, we can fine-tune the whole model while still keeping the
BatchNorm2d layers frozen to prevent losing what the model has already
learned.

``` python
for m in model2.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad_(False)
        m.bias.requires_grad_(False)

summary(model=model2, 
        input_size=(32, 3, 224, 224), 
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
        )
```

    ============================================================================================================================================
    Layer (type (var_name))                                      Input Shape          Output Shape         Param #              Trainable
    ============================================================================================================================================
    EfficientNet (EfficientNet)                                  [32, 3, 224, 224]    [32, 1]              --                   Partial
    ├─Sequential (features)                                      [32, 3, 224, 224]    [32, 1280, 7, 7]     --                   Partial
    │    └─Conv2dNormActivation (0)                              [32, 3, 224, 224]    [32, 24, 112, 112]   --                   Partial
    │    │    └─Conv2d (0)                                       [32, 3, 224, 224]    [32, 24, 112, 112]   648                  True
    │    │    └─BatchNorm2d (1)                                  [32, 24, 112, 112]   [32, 24, 112, 112]   (48)                 False
    │    │    └─SiLU (2)                                         [32, 24, 112, 112]   [32, 24, 112, 112]   --                   --
    │    └─Sequential (1)                                        [32, 24, 112, 112]   [32, 24, 112, 112]   --                   Partial
    │    │    └─FusedMBConv (0)                                  [32, 24, 112, 112]   [32, 24, 112, 112]   5,232                Partial
    │    │    └─FusedMBConv (1)                                  [32, 24, 112, 112]   [32, 24, 112, 112]   5,232                Partial
    │    └─Sequential (2)                                        [32, 24, 112, 112]   [32, 48, 56, 56]     --                   Partial
    │    │    └─FusedMBConv (0)                                  [32, 24, 112, 112]   [32, 48, 56, 56]     25,632               Partial
    │    │    └─FusedMBConv (1)                                  [32, 48, 56, 56]     [32, 48, 56, 56]     92,640               Partial
    │    │    └─FusedMBConv (2)                                  [32, 48, 56, 56]     [32, 48, 56, 56]     92,640               Partial
    │    │    └─FusedMBConv (3)                                  [32, 48, 56, 56]     [32, 48, 56, 56]     92,640               Partial
    │    └─Sequential (3)                                        [32, 48, 56, 56]     [32, 64, 28, 28]     --                   Partial
    │    │    └─FusedMBConv (0)                                  [32, 48, 56, 56]     [32, 64, 28, 28]     95,744               Partial
    │    │    └─FusedMBConv (1)                                  [32, 64, 28, 28]     [32, 64, 28, 28]     164,480              Partial
    │    │    └─FusedMBConv (2)                                  [32, 64, 28, 28]     [32, 64, 28, 28]     164,480              Partial
    │    │    └─FusedMBConv (3)                                  [32, 64, 28, 28]     [32, 64, 28, 28]     164,480              Partial
    │    └─Sequential (4)                                        [32, 64, 28, 28]     [32, 128, 14, 14]    --                   Partial
    │    │    └─MBConv (0)                                       [32, 64, 28, 28]     [32, 128, 14, 14]    61,200               Partial
    │    │    └─MBConv (1)                                       [32, 128, 14, 14]    [32, 128, 14, 14]    171,296              Partial
    │    │    └─MBConv (2)                                       [32, 128, 14, 14]    [32, 128, 14, 14]    171,296              Partial
    │    │    └─MBConv (3)                                       [32, 128, 14, 14]    [32, 128, 14, 14]    171,296              Partial
    │    │    └─MBConv (4)                                       [32, 128, 14, 14]    [32, 128, 14, 14]    171,296              Partial
    │    │    └─MBConv (5)                                       [32, 128, 14, 14]    [32, 128, 14, 14]    171,296              Partial
    │    └─Sequential (5)                                        [32, 128, 14, 14]    [32, 160, 14, 14]    --                   Partial
    │    │    └─MBConv (0)                                       [32, 128, 14, 14]    [32, 160, 14, 14]    281,440              Partial
    │    │    └─MBConv (1)                                       [32, 160, 14, 14]    [32, 160, 14, 14]    397,800              Partial
    │    │    └─MBConv (2)                                       [32, 160, 14, 14]    [32, 160, 14, 14]    397,800              Partial
    │    │    └─MBConv (3)                                       [32, 160, 14, 14]    [32, 160, 14, 14]    397,800              Partial
    │    │    └─MBConv (4)                                       [32, 160, 14, 14]    [32, 160, 14, 14]    397,800              Partial
    │    │    └─MBConv (5)                                       [32, 160, 14, 14]    [32, 160, 14, 14]    397,800              Partial
    │    │    └─MBConv (6)                                       [32, 160, 14, 14]    [32, 160, 14, 14]    397,800              Partial
    │    │    └─MBConv (7)                                       [32, 160, 14, 14]    [32, 160, 14, 14]    397,800              Partial
    │    │    └─MBConv (8)                                       [32, 160, 14, 14]    [32, 160, 14, 14]    397,800              Partial
    │    └─Sequential (6)                                        [32, 160, 14, 14]    [32, 256, 7, 7]      --                   Partial
    │    │    └─MBConv (0)                                       [32, 160, 14, 14]    [32, 256, 7, 7]      490,152              Partial
    │    │    └─MBConv (1)                                       [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (2)                                       [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (3)                                       [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (4)                                       [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (5)                                       [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (6)                                       [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (7)                                       [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (8)                                       [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (9)                                       [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (10)                                      [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (11)                                      [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (12)                                      [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (13)                                      [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    │    └─MBConv (14)                                      [32, 256, 7, 7]      [32, 256, 7, 7]      1,005,120            Partial
    │    └─Conv2dNormActivation (7)                              [32, 256, 7, 7]      [32, 1280, 7, 7]     --                   Partial
    │    │    └─Conv2d (0)                                       [32, 256, 7, 7]      [32, 1280, 7, 7]     327,680              True
    │    │    └─BatchNorm2d (1)                                  [32, 1280, 7, 7]     [32, 1280, 7, 7]     (2,560)              False
    │    │    └─SiLU (2)                                         [32, 1280, 7, 7]     [32, 1280, 7, 7]     --                   --
    ├─AdaptiveAvgPool2d (avgpool)                                [32, 1280, 7, 7]     [32, 1280, 1, 1]     --                   --
    ├─Sequential (classifier)                                    [32, 1280]           [32, 1]              --                   True
    │    └─Dropout (0)                                           [32, 1280]           [32, 1280]           --                   --
    │    └─Sequential (1)                                        [32, 1280]           [32, 1]              --                   True
    │    │    └─Linear (0)                                       [32, 1280]           [32, 512]            655,872              True
    │    │    └─ReLU (1)                                         [32, 512]            [32, 512]            --                   --
    │    │    └─Dropout (2)                                      [32, 512]            [32, 512]            --                   --
    │    │    └─Linear (3)                                       [32, 512]            [32, 256]            131,328              True
    │    │    └─ReLU (4)                                         [32, 256]            [32, 256]            --                   --
    │    │    └─Linear (5)                                       [32, 256]            [32, 1]              257                  True
    ============================================================================================================================================
    Total params: 20,964,945
    Trainable params: 20,811,073
    Non-trainable params: 153,872
    Total mult-adds (G): 91.19
    ============================================================================================================================================
    Input size (MB): 19.27
    Forward/backward pass size (MB): 6234.24
    Params size (MB): 83.86
    Estimated Total Size (MB): 6337.37
    ============================================================================================================================================

``` python
gc.collect()
torch.cuda.empty_cache()
```

``` python
start = time.time()

os.environ['PYTHONHASHSEED'] = str(1)
random.seed(0)

torch.cuda.manual_seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

BATCH_SIZE = 16
NUM_WORKERS = os.cpu_count() 
EPOCHS = 20       

loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(model2.parameters(), 
                            lr= 5e-5, 
                            weight_decay = 2e-5,
                            )

scheduler = CosineAnnealingLR(optimizer,
                              T_max = EPOCHS,
                              eta_min = 5e-8)

# define train and val data
train_data = datasets.ImageFolder(root = train_dir,
                                  transform = preprocess_train,
                                  target_transform = None,
                                  )

val_data = datasets.ImageFolder(root = train_dir,
                                transform = preprocess_val,
                                target_transform = None,
                                )
  
train_subset = torch.utils.data.Subset(train_data, train_indices)
val_subset = torch.utils.data.Subset(val_data, val_indices)
  
train_dataloader = DataLoader(train_subset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True,
                              )
  
val_dataloader = DataLoader(val_subset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=True,
                            )

model3, results = fit(model = model2,
                   train_dataloader = train_dataloader,
                   val_dataloader = val_dataloader,
                   optimizer = optimizer,
                   scheduler = scheduler,
                   loss_fn = loss_fn,
                   epochs = EPOCHS,
                   device = device)

finish = time.time()
print(f'Time taken to compute: {(finish - start)/60:.3f} min')
```

      0%|          | 0/20 [00:00<?, ?it/s]

    Time taken to compute: 7.377 min

``` python
img = plot_training(results)
img.savefig('train2.png', dpi = 300)

files.download('train2.png')
```

    <IPython.core.display.Javascript object>

    <IPython.core.display.Javascript object>

![](Copy_of_XRay_transfer_learning_files/figure-gfm/cell-40-output-3.png)

# Final evaluation

Since we have fine-tuned our custom EfficientNet model, it is time to
test how it performs on unseen data. Remember, we have 234 and 390
images, 624 in total, but this dataset is unbalanced, which will allow
us to obtain precision, recall and accuracy scores for our trained
model.

``` python
# Get a random list of image paths from test set

num_images_to_plot = 1
test_image_path_list = list(Path(test_dir).glob("*/*.jpeg")) # get list all image paths from test data 
test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                       k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

class_names = train_data.classes
# Make predictions on and plot the images
for image_path in test_image_path_sample:
  predict_image(model=model3,
                image_path=str(image_path),
                class_names=train_data.class_to_idx,
                transform=preprocess_val, 
                ).savefig('prediction.png', dpi = 300)
```

    WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

![](Copy_of_XRay_transfer_learning_files/figure-gfm/cell-41-output-2.png)

``` python
truths = []
pred_labels = []

for image_path in test_image_path_list:
  truth, pred_label = predict(model=model3,
                              image_path=str(image_path),
                              class_names=train_data.class_to_idx,
                              transform=preprocess_val,
                              )
  truths.append(truth)
  pred_labels.append(pred_label)
```

``` python
# get overall accuracy and display confusion matrix

ConfusionMatrixDisplay.from_predictions(truths, pred_labels).figure_.savefig('conf_mat.png',dpi=300)
print(f'Test accuracy: {accuracy_score(truths, pred_labels):.2f}')
```

    Test accuracy: 0.88

![](Copy_of_XRay_transfer_learning_files/figure-gfm/cell-43-output-2.png)

``` python
print(classification_report(truths, pred_labels))
```

                  precision    recall  f1-score   support

               0       0.98      0.69      0.81       234
               1       0.84      0.99      0.91       390

        accuracy                           0.88       624
       macro avg       0.91      0.84      0.86       624
    weighted avg       0.89      0.88      0.87       624

**In summary :**

-   Transfer learning is a powerful tool for automating the analysis of
    chest X-ray images for pneumonia detection.
-   First, based on empirical data, we identified the most efficient
    pre-trained model for classification currently avaliable on
    TorchVision based on the Acc@1 and number of Params. The
    EfficientNet_V2_S seemed like a good starting point for this
    project. Note: One may choose to do a more detailed model selection
    in future experiments.
-   We implemented several strategies to reduce the risk of overfitting
    the dataset, such as TrivialAugment, Cutmix, Random erasing, etc.
-   Secondly, we froze all layers and trained only the top layers. Then,
    proceeded to perform fine-tuning of the model with all layers set to
    param.requires_grad = True
-   Final accuracy of the model, based on the test dataset was 88% with
    a precision of 84% and recall of 99% with respect to the pneumonia
    cases.
