---
title: "Peek My Home Price"
published: 2024-11-20
draft: false
author: "Adam Cseresznye"
toc: true
---

![Photo by Christian Allard UnSplash](https://images.unsplash.com/photo-1549407408-4b016f497e4d?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

Welcome to `Peek My Home Price`, a project, that dives into the key factors that influence real estate property prices in Belgium. Our ultimate goal with this project is to leverage up-to-date data from leading real estate platforms in the country to accurately predict house prices. We aim to create a platform that allows users to gain insights into the dynamic Belgian real estate market, province by province.

:::note
You can explore the project's app on its [website](https://peek-my-home-price.fly.dev/). For more details, visit the [GitHub repository](https://github.com/adamcseresznye/peek_my_home_price).

Check out the series for a deeper dive:
- [Part 1: Characterizing the Data](https://adamcseresznye.github.io/blog/projects/peek_my_home_price/part-1_characterizing_the_data.html)
- [Part 2: Building a Baseline Model](https://adamcseresznye.github.io/blog/projects/peek_my_home_price/part-2_building_a_baseline_model.html)
- [Part 3: Feature Selection](https://adamcseresznye.github.io/blog/projects/peek_my_home_price/part-3_feature_selection.html)
- [Part 4: Feature Engineering](https://adamcseresznye.github.io/blog/projects/peek_my_home_price/part-4_feature_engineering.html)
- [Part 5: Fine-Tuning](https://adamcseresznye.github.io/blog/projects/peek_my_home_price/part-5_fine_tuning.html)
:::

The app is divided into three main sections:

1) **Dashboard**: Get a quick snapshot of the latest real estate trends, including average prices, the most active regions, and interesting facts about the dataset.
2) **Trends**: Dive deeper into historical price trends. Explore median price changes over time for each Belgian province.
3) **Prediction**: Input specific variables and generate your own price predictions based on our latest trained model.

In this blog series, we'll take you behind the scenes of `Peek My Home Price` and guide you through the thought process that led to the creation of the application. Feel free to explore the topics that pique your interest or that you'd like to learn more about. We hope you'll find this information valuable for your own projects. Let's get started!

# Web scraping

The data used to train our model is sourced from prominent Belgian real estate platforms. Employing the `Scrapy` framework, we systematically extract relevant features. The collected data then undergoes a preprocessing pipeline, including duplicate removal and data type conversion. The cleaned dataset is subsequently stored in a `PostgreSQL` database, ready for model training. If you would like to learn more about this step, please visit the [`src/scraper`](https://github.com/adamcseresznye/peek_my_home_price/tree/main/src/scraper) module.

# Describing the data

Before diving into analysis, it's crucial to take a closer look at our dataset's preliminary version. This step is essential because, often, we collect more data than we actually need. By examining the initial dataset, we can gain a deeper understanding of the relationships between variables and our target variable of interest – in this case, price. For this, we will use a sample dataset that contains more features than we ended up using. It's often not so much about deciding what data to collect but rather what data to retain. First, we dive into the initial data collected to examine the features that are commonly shared among most ads. After identifying these common attributes, we can optimize our data collection process by keeping these key characteristics and removing the less common ones.

As depicted in @fig-fig1, the features `energy_class`, `lat`, `lng` demonstrate the highest completeness, with more than 90% of instances being present. In contrast, `subdivision_permit`, `yearly_theoretical_energy_consumption` and `width_of_the_total_lot` are among the least populated features, with roughly 10-15% of non-missing instances.

This information allows us to devise a strategy where we, for example, could retain features with a completeness of over 50%.

## Assessing Feature Cardinality

Now, let's assess the feature cardinality of our dataset to differentiate between categorical and numerical variables. To do this, we will analyze the percentage of unique values per feature.

## Distribution of the target variable

Upon examining the distribution of our target variable, which is the `price`, it becomes evident that there is a notable skew. Our median price stands at 379,000 EUR, with the lowest at 350,000 EUR and the highest reaching 10 million EUR. To increase the accuracy of our predictions, it is worth considering a transformation of our target variable before proceeding with modeling. This transformation serves several beneficial purposes:

1. **Normalization**: It has the potential to render the distribution of the target variable more symmetrical, resembling a normal distribution. Such a transformation can significantly enhance the performance of various regression models.

2. **Equalizing Variance**: By stabilizing the variance of the target variable across different price ranges, this transformation becomes particularly valuable for ensuring the effectiveness of certain regression algorithms.

3. **Mitigating Outliers**: It is effective at diminishing the impact of extreme outliers, bolstering the model's robustness against data anomalies.

4. **Interpretability**: Notably, when interpreting model predictions, this transformation allows for straightforward back-transformation to the original scale. This can be achieved using a base 10 exponentiation, ensuring that predictions are easily interpretable in their origination task.

## Relationship between independent and dependent variables

Next, we will investigate how house prices vary when grouped according to our independent variables. Please take into account that the price values have undergone log transformation to address skewness.

## Correlations

Finally, we will look into the correlations among variables with high cardinality through Spearman correlation analysis. As evident from the heatmap, the price exhibits a strong correlation with cadastral income (correlation coefficient = 0.77), living area (correlation coefficient = 0.74), and bathrooms (correlation coefficient = 0.59). For your reference, _cadastral income_ is an annual Flemish property tax based on the assessed rental income of immovable properties in the Flemish Region. This income is a notional rental income assigned to each property, whether it is rented out or not.

Having laid the groundwork with initial data exploration in Part 1, we're now ready to take the next step: building a foundational machine learning model. In Part 2, we'll put various algorithms to the test, establishing a benchmark that will serve as a reference point for our future model-building and feature engineering efforts. This baseline model will provide a crucial starting point, guiding us as we work to refine and enhance our predictive capabilities. See you there!
