K-Fold cross validation of Deep Learning models using SciKeras
================

## Aim:

-   Prepare a mock dataset with Sklearn’s make_regression function
-   Build a simple sequential neuronal network using Keras
-   Show how k-fold cross validation can be achieved with SciKeras to
    test if our model can also perform well on unseen data

For further reading please consult the following resources:

-   Cross-validation on
    [Wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
-   Official cross-validation page on
    [Scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)
-   KDnuggets
    [article](https://www.kdnuggets.com/2022/07/kfold-cross-validation.html)

## Prepare dataset

First, we will create a synthetic dataset using Sklearn’s
[make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html?highlight=make_regression#sklearn.datasets.make_regression)
function. To make the classification task a little more challenging, we
will also apply a [logarimic
function](https://numpy.org/doc/stable/reference/generated/numpy.expm1.html#numpy.expm1)
to our y values to obtain non-linear targets which cannot be fitted
using a simple linear model.

Once we are done with this, we can also split our dataset into random
train and test subsets using Sklearn’s
[train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
function.

``` python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

X,y= make_regression(n_samples= 1000,
                     n_features= 10,
                     n_informative= 5,
                     noise= 100,
                     bias= 500,
                     tail_strength= 0.1,
                     random_state= 42)
y_trans = np.log1p(y)

X_train, X_test, y_train, y_test= train_test_split(X,y_trans, test_size=0.2, random_state=42)
```

``` python
for i in X_train, X_test, y_train, y_test:
  print(f'Shape: {i.shape}')
```

    Shape: (800, 10)
    Shape: (200, 10)
    Shape: (800,)
    Shape: (200,)

Let’s plot the data to see which attributes could be meaningful for our
classifier.

``` python
import seaborn as sns
import matplotlib.pyplot as plt

a = 2  # number of rows
b = 5  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(20,10), dpi= 600)

for i in range(X_train.shape[1]):
    plt.subplot(a, b, c)
    plt.title(f'Attribute {i}')
    plt.xlabel('x')
    plt.ylabel('y')
    sns.scatterplot(x= X_train[:,i],y= y_train)
    c += 1

plt.tight_layout()
plt.show()
```

<img src="{{site.baseurl | prepend: site.url}}assets/images/220817_SciKeras_1.jpg" alt="220817_SciKeras_1" />

## Standard scaling

While our generated synthetic dataset already appears to be normally
distributed, it is always good practice to perform standard scaling of
features. Sklearn’s
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler)
standardizes features by removing the mean and scaling them to unit
variance.

``` python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#Standard scaling data
X_train_transformed = StandardScaler().fit_transform(X_train)
X_test_transformed = StandardScaler().fit_transform(X_test)

#Plotting distribution
fig, axes= plt.subplots(1,2, figsize= (12,6))
sns.histplot(data= X_train, ax= axes[0], kde= True).set_title('Histogram of training data before standard scaling')
sns.histplot(data= X_train_transformed, ax= axes[1], kde= True).set_title('Histogram of training data after standard scaling')
```

    Text(0.5, 1.0, 'Histogram of training data after standard scaling')

<img src="{{site.baseurl | prepend: site.url}}assets/images/220817_SciKeras_2.jpg" alt="220817_SciKeras_2" />

## Building a base sequential NN model

Let’s start out with constructing a simple sequential deep learning
model. As you can see from the figure below, while our model reaches
very low loss values and a similarly low mean absolute error, this does
not give us an indication as to how our model would perform on unseen
data. After all, this is only a demonstration that our model can
memorize the patterns present in the training set.

``` python
from tensorflow import keras
from tensorflow.keras import layers

#Construct the model
model = keras.Sequential([layers.Dense(64, activation="relu"),
                          layers.Dense(1)])

#compile
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

#train
history= model.fit(X_train_transformed, 
                   y_train, 
                   epochs=200, 
                   batch_size=16,
                   verbose= 0
                   )
```

``` python
fig,axes= plt.subplots(1,2, figsize= (12,6), dpi= 300)
axes[0].plot(range(20, np.array(history.history['loss']).size),
             np.array(history.history['loss'])[20:])
axes[0].set_title('Training loss measured')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')

axes[1].plot(range(20, np.array(history.history['mae']).size),
             np.array(history.history['mae'])[20:])
axes[1].set_title('Training mean absolute error measured')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('MAE')

print(f"Epoch where training mae reaches minimum: {np.argmin(np.array(history.history['mae']))}")
print(f"Minimum training mae: {np.min(np.array(history.history['mae']))}")
print('')
print(f"Epoch where training loss reaches minimum: {np.argmin(np.array(history.history['loss']))}")
print(f"Minimum training loss: {np.min(np.array(history.history['loss']))}")
```

    Epoch where training mae reaches minimum: 196
    Minimum training mae: 0.12020333111286163

    Epoch where training loss reaches minimum: 199
    Minimum training loss: 0.02440175600349903

<img src="{{site.baseurl | prepend: site.url}}assets/images/220817_SciKeras_3.jpg" alt="220817_SciKeras_3" />

## Performing k-fold cross validation with SciKeras

To gain insight as to how well our model could generalize, we can
perform a k-fold cross validation.

While there are several ways one could perform this, for the sake of
this article, we decided to use the [SciKeras
library](https://www.adriangb.com/scikeras/stable/). The goal of
SciKeras is to “*make it possible to use Keras/TensorFlow with
sklearn*”. This is achieved by providing wrapped instances around
Keras.  
Why is this great, you may ask? This allows us to call methods like
fit(), predict() and score() on our models built by Keras. In short,
SciKeras allows us to use and operate keras models as if they were
scikit-learn models.

[Here](https://www.adriangb.com/scikeras/stable/install.html) you will
find instructions on how to install SciKeras on your local machine/or
Google Colab.

``` python
#pip install scikeras[tensorflow]
```

Implementing a 4-fold validation of our base NN model using SciKeras:

``` python
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import KFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import clear_session

'''
Step 1: Construct a model building function that compiles and 
returns the model 
'''

def build_model():
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=10, activation='relu'))
    model.add(Dense(1,))
    # Compile model
    model.compile(loss= keras.losses.MeanSquaredError(), 
                  optimizer= keras.optimizers.RMSprop(), 
                  metrics= [keras.metrics.MeanAbsoluteError()])
    return model

'''    
Step 2: Instantiate the empty lists. We will fill these with all the training
and validation losses and metrics recorded throughout the model training
'''

losses= []
mean_absolute_errors= []
val_losses= []
val_mean_absolute_errors= []

'''
Step 3: Construct the for loop:
To split our dataset into training and validation sets, we will use Sklearn's
KFold. Here, we will perform a 4-fold cross validation
'''

for train, test in KFold(n_splits=4, shuffle=False).split(X_train_transformed,y_train):
  #we use tf.keras.backend.clear_session to release the global state
  clear_session()
  #wrap our Keras model with SciKeras
  history = KerasRegressor(#build_fn=build_model,
                         model= build_model(),
                         epochs=200,
                         batch_size=16,
                         verbose=0,
                         optimizer= keras.optimizers.RMSprop(),
                         loss= keras.losses.MeanSquaredError(),
                         metrics= [keras.metrics.MeanAbsoluteError()],
                         )
  '''
  Let's fit our model. Please note the train and test indices used. These 
  indices keep track of which portion of the data is designated as training
  and validation sets. We train the model using the train indices while the
  validation_data argument contains the portion of the data that is designated 
  as validation set. This is different from iteration to iteration.
  '''
  temp= history.fit(X_train_transformed[train],
                    y_train[train],
                    validation_data = (X_train_transformed[test],
                                       y_train[test]))
  losses.append(temp.history_['loss'])
  mean_absolute_errors.append(temp.history_['mean_absolute_error'])
  val_losses.append(temp.history_['val_loss'])
  val_mean_absolute_errors.append(temp.history_['val_mean_absolute_error'])
```

``` python
import matplotlib.pyplot as plt

fig, axes= plt.subplots(1,2, figsize= (12,6), dpi= 600)
axes[0].plot(range(10,len(np.array(losses).mean(axis= 0))),
          np.array(losses).mean(axis= 0)[10:], label= 'training losses')
axes[0].plot(range(10, len(np.array(val_losses).mean(axis= 0))),
          np.array(val_losses).mean(axis= 0)[10:], label= 'validation losses')

axes[0].set_title('Training and validation losses calculated with Scikeras\ndata points 10:200 are displayed')
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epochs')
axes[0].legend()

axes[1].plot(range(10, len(np.array(mean_absolute_errors).mean(axis= 0))),
          np.array(mean_absolute_errors).mean(axis= 0)[10:], label= 'training mae')
axes[1].plot(range(10, len(np.array(val_mean_absolute_errors).mean(axis= 0))),
          np.array(val_mean_absolute_errors).mean(axis= 0)[10:], label= 'validation mae')
axes[1].legend()
axes[1].set_title('Training and validation mae-s calculated with Scikeras\ndata points 10:200 are displayed')
axes[1].set_ylabel('MAE')
axes[1].set_xlabel('Epochs')
axes[1].legend()

axes[0].axvline(x=np.argmin(np.array(val_losses).mean(axis= 0)), 
                ymin=0.05, ymax=0.95, color='r', ls='--')

axes[1].axvline(x=np.argmin(np.array(val_mean_absolute_errors).mean(axis= 0)), 
                ymin=0.05, ymax=0.95, color='r', ls='--')
```

    <matplotlib.lines.Line2D at 0x25bb06f0820>

<img src="{{site.baseurl | prepend: site.url}}assets/images/220817_SciKeras_4.jpg" alt="220817_SciKeras_4" />

As you can see from the figure above, while the training losses and mean
absolute errors are dicreasing throught our training, the validation set
clearly shows that our validation losses and mean absolute errors stop
significantly improving after 50-75 epochs. Thus indicating overfitting.

**To sum up:**

-   In this article we went through how SciKeras can be used to easily
    implement K-fold validation of our Keras deep learning regression
    model using the already familiar scikit-learn library.
-   We also demonstrated that K-fold validation can be an incredibly
    useful tool to reliably evaluate a model when there is little data
    available.
