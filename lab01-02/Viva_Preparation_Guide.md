# Viva Preparation Guide

## Project Title

**India Air Quality and Crop Production EDA**

## One-Line Explanation

This project uses exploratory data analysis to study Indian air-quality data and crop-production data, clean both datasets, visualize pollution patterns, and check whether states with higher pollution appear to have lower crop productivity.

## What Model Did I Use?

I did **not use a machine learning prediction model** such as Linear Regression, Decision Tree, Random Forest, or Neural Network.

This lab is mainly an **EDA project**, so I used statistical and visualization methods:

- Data profiling
- Missing-value treatment
- State-name standardization
- Duplicate removal
- IQR-based outlier treatment
- Time-series trend analysis
- Seasonal analysis
- Aggregation and merging
- Correlation analysis
- Scatter plot with regression line

If my mentor asks, "Which model did you use?", the best answer is:

> I did not build a predictive ML model because the lab objective was data preprocessing, visualization, and exploration. I used exploratory statistical methods such as median imputation, IQR outlier treatment, aggregation, Pearson correlation, and a regression line for visualization. The regression line is not used for prediction; it only helps show the direction of relationship between AQI and crop production per area.

## Why I Did Not Use a Machine Learning Model

I avoided building a prediction model because:

1. The two datasets do not match perfectly in time.
   - Air data covers mainly 2015 to 2020.
   - Crop data covers 1997 to 2015.

2. The datasets are at different levels.
   - Air data is city-day level.
   - Crop data is state/district/crop/year level.

3. Crop production depends on many missing factors.
   - Rainfall
   - Irrigation
   - Soil quality
   - Fertilizer use
   - Crop type
   - Pests
   - Market conditions

Because of these limitations, a machine learning model would be misleading. EDA is the correct first step.

## Logic Behind Techniques Based On UNIT1.pptx

Your Unit 1 slides focus on **data representation, descriptive statistics, data types, visualization, outliers, scatter diagrams, trendlines, statistical inference, and data-quality challenges**. I used those ideas directly in this lab.

### 1. Dataset Representation

From Unit 1:

- A dataset is a structured collection of data.
- Rows are samples/data points.
- Columns are features/attributes.

In this notebook:

- Air dataset rows are **city-day samples**.
- Crop dataset rows are **district-crop-year samples**.
- Pollutants like `PM2.5`, `PM10`, `NO2`, and `AQI` are features.
- Crop variables like `Area` and `Production` are features.

**Viva answer:**

> I first identified the samples and features in both datasets. This helped me understand that the air dataset and crop dataset are at different granularities, so they cannot be merged directly.

### 2. Descriptive Statistics

From Unit 1:

- Descriptive statistics summarize and present data in an understandable form.

In this notebook, I used:

- Shape
- Data types
- Missing counts
- Unique values
- Mean
- Median
- Standard descriptive summaries

**Viva answer:**

> I used descriptive statistics because the lab objective was to understand and summarize the data before any modeling. This helped identify missing values, skewness, outliers, and data-quality issues.

### 3. Qualitative And Quantitative Data

From Unit 1:

- Qualitative data describes categories or labels.
- Quantitative data contains numerical values that can be analyzed statistically.

In this notebook:

- `City`, `State`, `Season`, and `Crop` are qualitative/nominal features.
- `AQI`, `PM2.5`, `PM10`, `Area`, and `Production` are quantitative features.

**Why this matters:**

- I did not average city or crop names.
- I standardized categorical labels.
- I used statistical imputation only for numeric variables.

**Viva answer:**

> I selected cleaning methods based on data type. Numerical columns were imputed using medians, while categorical columns were standardized because labels cannot be averaged.

### 4. Median Imputation

From Unit 1:

- Mean and median are descriptive measures.
- Median is useful when data is skewed or affected by extreme values.

In this notebook:

- AQI and pollutant readings are skewed.
- Crop production also has large variation.
- So I used median instead of mean.

**Viva answer:**

> I used median imputation because environmental and production data can have extreme values. Median is more robust than mean, so it gives a better typical value for skewed data.

### 5. Histogram For AQI

From Unit 1:

- Histograms summarize quantitative data using intervals and frequency.

In this notebook:

- `AQI` is quantitative.
- A histogram shows where AQI values are concentrated.

**Viva answer:**

> I used a histogram because AQI is a quantitative variable, and a histogram shows the frequency distribution across AQI ranges.

### 6. Bar Charts For Categories

From Unit 1:

- Bar charts are used for categorical data and frequency/percentage comparisons.

In this notebook, I used bar charts for:

- Missing-value percentage by column
- Monthly AQI
- Seasonal AQI
- Most vs least polluted states

**Viva answer:**

> I used bar charts where I needed to compare categories, such as months, seasons, columns, or pollution groups.

### 7. Boxplot And Outlier Logic

From Unit 1:

- Outliers are important data-quality issues.
- They can distort analysis.

In this notebook:

- I used boxplots to detect extreme AQI values.
- I used IQR because it is based on quartiles.
- I capped outliers instead of deleting them.

**Viva answer:**

> I used the IQR method because it is robust for skewed data. I capped outliers to reduce their influence while preserving the records.

### 8. Line Plot For Time Trend

From Unit 1:

- Graphical displays help identify patterns and trends.

In this notebook:

- I extracted year from date.
- I plotted average AQI over time.

**Viva answer:**

> I used a line plot because it is the clearest way to show time-based trends, such as whether AQI is improving or worsening year by year.

### 9. Scatter Plot And Trendline

From Unit 1:

- Scatter diagrams show relationships between two quantitative variables.
- Trendlines approximate the relationship.

In this notebook:

- `Avg_AQI` is quantitative.
- `Production_per_Area` is quantitative.
- Scatter plot shows their relationship.
- Regression line shows direction of association.

**Viva answer:**

> I used a scatter plot with a trendline because Unit 1 explains that scatter diagrams are useful for studying relationships between two quantitative variables. The trendline helps show whether the relationship is positive or negative.

### 10. Statistical Inference And Correlation

From Unit 1:

- Inferential statistics draws conclusions from data but must consider uncertainty.

In this notebook:

- I used Pearson correlation to measure association.
- The value was about `-0.21`.
- This is weak negative association.
- It does not prove causation.

**Viva answer:**

> Correlation helped me quantify the relationship between AQI and crop productivity. Since the value is weakly negative, it suggests a slight inverse relationship, but it does not prove that pollution causes lower crop yield.

### 11. Data Quality Challenges

From Unit 1:

- Machine learning faces challenges such as poor-quality data, insufficient data, non-representative data, irrelevant features, and bias.

In this notebook:

- There are missing values.
- The two datasets have weak year overlap.
- The datasets have different levels of detail.
- Some important crop-yield factors are missing.

**Viva answer:**

> I treated this as an EDA problem because the data has quality limitations. Building a predictive model without rainfall, irrigation, soil, and exact time overlap could create biased or misleading results.

## Methods Used And Why

### 1. Data Profiling

**Method used:** `shape`, `head()`, `dtypes`, missing counts, unique values, duplicates, and `describe()`.

**Why used:**  
Before analysis, I needed to understand the size, structure, data types, missing values, and possible problems in both datasets.

**Viva answer:**  
> I used data profiling to get a first impression of the datasets. It helped me identify missing values, data types, duplicate rows, date ranges, city names, state names, and columns that may need cleaning before analysis.

### 2. Missing Value Treatment

**Method used:** Median imputation.

For air-quality numeric columns, I used:

```python
city median first, then overall median
```

For crop production, I used:

```python
state-crop median first, then overall median
```

**Why median instead of mean:**  
Pollution and production values can have extreme values. The mean is affected by outliers, but the median is more stable.

**Viva answer:**  
> I used median imputation because AQI and pollutant values are skewed and may contain extreme readings. I filled air-quality values using city-wise medians because each city has a different pollution pattern. For crop production, I used state-crop medians because production depends on both location and crop type.

### 3. AQI Bucket Recalculation

**Method used:** Function to convert numeric AQI into categories.

Categories:

- Good
- Satisfactory
- Moderate
- Poor
- Very Poor
- Severe

**Why used:**  
After filling missing AQI values, the old AQI bucket may no longer be correct. So I recalculated it.

**Viva answer:**  
> I recalculated AQI bucket from the cleaned AQI value to keep the category consistent with the numeric AQI.

### 4. State Name Standardization

**Method used:** Manual city-to-state mapping and string cleaning.

**Why used:**  
The air dataset had cities, while the crop dataset had states. To merge them, both needed a common `State` column.

**Viva answer:**  
> I created a state column in the air dataset by mapping each city to its state. I also removed extra spaces and fixed name variations like Orissa to Odisha. This was necessary because even a small spelling difference can break a merge.

### 5. Duplicate Removal

**Method used:** `drop_duplicates()`.

**Why used:**  
Duplicate rows can overrepresent some observations and distort averages, counts, and correlations.

**Viva answer:**  
> I removed exact duplicate rows so repeated records would not bias the analysis.

### 6. AQI Distribution Analysis

**Methods used:** Histogram, KDE curve, and boxplot.

**Why histogram:**  
It shows where most AQI values are concentrated.

**Why boxplot:**  
It shows spread and extreme values.

**Viva answer:**  
> I used a histogram to understand the distribution of AQI values and a boxplot to identify whether extreme values were pulling the average upward.

### 7. Outlier Detection

**Method used:** IQR method.

Formula:

```text
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
```

**Treatment used:** Capping / clipping.

**Why capping instead of deleting:**  
Deleting rows can remove useful data. Capping keeps the record but reduces the impact of unrealistic extreme values.

**Viva answer:**  
> I used the IQR method because it is simple, transparent, and robust for skewed data. I capped extreme AQI values instead of deleting them so the dataset size remained intact while reducing the effect of outliers.

### 8. Yearly AQI Trend

**Method used:** Extract year from date and calculate average cleaned AQI per year.

**Visualization used:** Line plot.

**Why line plot:**  
Line plots are best for showing trends over time.

**Viva answer:**  
> I extracted the year from the date column and calculated yearly average AQI. I used a line plot because it clearly shows whether pollution is increasing or decreasing over time.

### 9. Seasonal AQI Analysis

**Method used:** Extract month and group months into seasons.

Main season tested:

```text
October to December = Harvest/Post-monsoon
```

**Why used:**  
The lab question asked whether air quality is worse during harvest season.

**Viva answer:**  
> I grouped AQI by month and season to check whether October to December has higher pollution. The result partly supports the claim because November and December are high, but winter months like January are also very polluted.

### 10. Merging Both Datasets

**Problem:**  
The datasets cannot be directly merged.

Air dataset:

```text
City + Date level
```

Crop dataset:

```text
State + District + Crop + Year level
```

**Method used:** Aggregation before merge.

Air data was aggregated to:

```text
State-Year
```

Crop data was aggregated to:

```text
State-Year
```

Because the exact state-year overlap was weak, the notebook used a state-level fallback.

**Viva answer:**  
> I could not merge the datasets directly because they were collected at different levels. I aggregated air data to state-year level and crop data to state-year level. Since exact year overlap was limited, I used a state-level fallback for exploratory comparison.

### 11. Correlation Analysis

**Method used:** Pearson correlation.

**Why used:**  
It measures the direction and strength of a linear relationship between numeric variables.

Important result:

```text
Correlation between Avg_AQI and Production_per_Area ≈ -0.21
```

**Interpretation:**  
This is a weak negative relationship.

**Viva answer:**  
> The correlation between average AQI and production per area is about -0.21. This means higher pollution is weakly associated with lower crop production per area, but the relationship is not strong and does not prove causation.

### 12. Regression Line In Scatter Plot

**Method used:** `sns.regplot()`.

**Important point:**  
This is not a full prediction model. It is a visualization aid.

**Why used:**  
It shows whether the overall relationship between AQI and crop productivity slopes upward or downward.

**Viva answer:**  
> I used a regression line only to visualize the direction of association. It helps show whether the relationship is positive or negative, but I did not use it as a predictive model.

## Main Findings

1. Many AQI readings fall in the moderate-to-poor range.
2. Extreme AQI values exist and can distort the average.
3. AQI changes by season, with winter and post-monsoon months showing higher pollution.
4. The relationship between AQI and crop production per area is weakly negative.
5. The analysis does not prove that pollution directly reduces crop yield.

## Limitations

The analysis has important limitations:

- The two datasets do not fully overlap in years.
- Air data is city-level, while crop data is state/district-level.
- Crop yield depends on rainfall, irrigation, soil, crop type, pests, fertilizer, and economics.
- Correlation does not prove causation.
- Some state-level results may hide district-level variation.

## Strong Viva Answers

### Q1. What is the aim of your project?

> The aim is to clean and explore Indian air-quality and crop-production datasets, identify pollution patterns, and check whether states with worse air quality tend to have lower crop production per area.

### Q2. Which machine learning model did you use?

> I did not use a machine learning prediction model. This lab focuses on preprocessing, visualization, and exploratory analysis. I used statistical methods like median imputation, IQR outlier treatment, correlation analysis, and a regression line for visualizing association.

### Q3. Why did you use median imputation?

> I used median imputation because AQI and crop-production data can contain extreme values. Median is less affected by outliers than mean, so it gives a more reliable replacement value.

### Q4. Why did you use IQR for outliers?

> IQR is suitable because it uses quartiles and is not heavily affected by skewed data. AQI data can have extreme pollution readings, so IQR is a simple and robust method.

### Q5. Why did you cap outliers instead of removing them?

> I capped them because deleting rows could remove useful records. Capping reduces the effect of extreme values while keeping the dataset complete.

### Q6. Why did you use a histogram and boxplot?

> The histogram shows where most AQI values are concentrated, while the boxplot shows spread and outliers. Together, they answer whether pollution is common or caused by only a few extreme observations.

### Q7. Why did you use a line plot for yearly AQI?

> A line plot is best for time trends because it shows how AQI changes from year to year.

### Q8. Why did you aggregate before merging?

> The datasets were at different levels. Air data was city-day level and crop data was district-crop-year level. Aggregation was necessary to bring them to a common state-level format before merging.

### Q9. What does the correlation value mean?

> The correlation of around -0.21 means there is a weak negative relationship between AQI and crop production per area. Higher AQI is slightly associated with lower productivity, but the relationship is weak.

### Q10. Does your analysis prove pollution reduces crop yield?

> No. It only shows association, not causation. Crop production depends on many other factors such as rainfall, irrigation, soil quality, crop type, pests, and economic conditions.

### Q11. What is the most important chart in your notebook?

> The AQI versus crop production per area scatter plot is the most important because it directly addresses the main question: whether more polluted states have lower crop productivity.

### Q12. What is your recommendation?

> The government should strengthen pollution monitoring and crop-residue management support during high-risk months, especially in states with consistently high AQI.

## Short Presentation Script

Good morning. My project is based on exploratory data analysis of Indian air-quality and crop-production datasets. The main question is whether worsening air quality is linked to lower agricultural output.

First, I loaded and profiled both datasets to understand their size, columns, missing values, duplicates, and date ranges. Then I treated missing values using median imputation because the data contains extreme values and median is more robust than mean. For air-quality data, I used city-wise median values, and for crop data, I used state-crop median values.

Next, I standardized state names and mapped cities to states so both datasets could be compared. I also removed duplicate rows. For AQI analysis, I used histograms and boxplots to understand the distribution and detect extreme values. I handled outliers using the IQR method and capped extreme values instead of deleting records.

Then I studied AQI trends over time using a line plot and seasonal patterns using monthly and seasonal bar charts. To compare air quality with crop production, I aggregated both datasets to a common state-level format and calculated production per area.

Finally, I used correlation analysis and a scatter plot with a regression line. The correlation between AQI and crop production per area is about -0.21, which shows a weak negative association. This suggests that higher pollution may be linked with lower productivity, but it does not prove causation. Other factors like rainfall, irrigation, soil, and crop type also affect production.

## Final Thing To Remember

The safest sentence to repeat in viva:

> This is an exploratory analysis, not a causal or predictive machine learning model. The results show patterns and weak associations, but they do not prove that air pollution alone causes lower crop production.
