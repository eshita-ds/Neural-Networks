# Child Mind Institute: Problematic Internet Usage Detection
<img width=650 src="https://github.com/user-attachments/assets/f4c789c2-a714-4a57-92f7-6f07d18f719f">

## Introduction

Excessive internet use among children and adolescents poses significant public health challenges, affecting mental health and development. This project explores a novel approach to predict problematic internet use (PIU) by analyzing physical activity data.

Using the Healthy Brain Network (HBN) dataset of ~5000 participants aged 5-22, we aim to model the Severity Impairment Index (SII), which classifies PIU severity into four levels (none, mild, moderate, severe). By linking accelerometer-based physical activity metrics with SII scores, this project seeks to develop scalable predictive models for early intervention and healthier digital habits.

<img width="608" alt="image" src="https://github.com/user-attachments/assets/7a793443-d1b0-4756-97f8-9ec6d59877e4" />

## Dataset Overview
**Data Source** - [link](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview)

The dataset for this competition, sourced from the Healthy Brain Network, includes data from ~5000 participants aged 5-22. It combines tabular data and time-series data to explore the relationship between physical activity and problematic internet use.

**Key Features**

- **Tabular Data:** Demographics (age, sex), physical health metrics (BMI, blood pressure), fitness results (FitnessGram protocols), subjective activity levels (PAQ_A/PAQ_C), sleep disturbances (SDS), and internet use habits (e.g., daily screen time).
  
  - Includes the Severity Impairment Index (SII), derived from the Parent-Child Internet Addiction Test (PCIAT), which categorizes internet use severity into None (0), Mild (1), Moderate (2), and Severe (3).
    
- **Time-Series Data:** Continuous 5-second accelerometer recordings (X, Y, Z axes) over multiple days, stored in parquet files. Metrics like ENMO (Euclidean Norm Minus One) and arm angles provide detailed activity insights, alongside contextual data like light intensity, time of day, and wear flags.

## Problem Formulation

**Challenges**

- Merging diverse data types: clinical, behavioral, and time-series accelerometer data.
- Handling ~25% missing targets and inconsistencies between train and test datasets.
- Understanding complex interactions between digital engagement, physical activity, and behavior.
  
**Approach**
- Integrated data across formats and engineered features to address missing values and inconsistencies.
- Applied advanced machine learning and deep learning models to handle multi-dimensional inputs and perform multi-class classification of SII
- Focused on scalable methods for early identification of problematic internet use.

## **Data Preprocessing**

- **Data Integration**: Combined parquet files (physical activity data) and CSV files (internet usage) using a common id column.
- **Missing Values:**
  - Numerical: Imputed with KNN.
  - Categorical: Replaced with "unknown" and encoded as categories.
- **Feature Engineering:** Aggregated statistics from parquet files, encoded categorical data with custom mappings to prevent data leakage.
- **Feature Scaling:** Applied StandardScaler to normalize numerical features for training stability.
- **Class Imbalance:** Addressed with Borderline-SMOTE to improve classification of minority classes.

## Model Selection
  - **Implementation of Multi-Layer Perceptron (MLP)**- The Multi-Layer Perceptron (MLP) predicts ordinal class labels using a feedforward architecture with ReLU-activated hidden layers and a Softmax output for probabilistic multi-class classification, effectively capturing complex data relationships.
    
  - **TabNet Ensemble Classifier** - The TabNet Classifier, optimized for tabular data, dynamically selects relevant features using sequential attention for efficient and interpretable learning. Five TabNet models were trained with different seeds, using AdamW optimizer, early stopping, and ensemble averaging for predictions.
    
  - **Custom Implementation of a TabNet Regressor** - TabNet, with a custom wrapper, combines neural network expressiveness and decision tree interpretability using attention mechanisms, enabling efficient, interpretable modeling of tabular data with added preprocessing, cross-validation, and optimized training.
    
  - **Implementation of Long Short-Term Memory (LSTM)** - LSTM, a type of RNN, processes sequential data by capturing long- and short-term dependencies using "Forget," "Input," and "Output" gates, making it ideal for NLP and time series tasks.
    
  - **Implementation of Siamese Long Short-Term Memory (LSTM)** - The model combines MLP for static tabular data and LSTM for time-series data, aggregating time steps into daily features. LSTM processes time-series data with Global Max-Pooling before merging with static features, ensuring unified input for all participants.
    
  - **Ensemble of Voting Regressor with TabNet** - The ensemble model uses Voting Regressor to combine LightGBM, XGBoost, CatBoost, and TabNet regressors, leveraging their strengths with fine-tuned hyperparameters. Time-series data is reduced via autoencoder, and optimized weights improve predictions and classification accuracy.
 
## Model Evaluation

Predictions were averaged across cross-validation folds, with threshold optimization used to map continuous outputs to discrete classes.**Quadratic Weighted Kappa (QWK)** served as the primary metric, with optimized thresholds improving alignment with ground truth. Cross-validation results included mean QWK scores for training and validation, ensuring robust performance and generalization.

## Result Analysis

TabNet Voting Regressor Ensemble achieved the highest Quadratic Weighted Kappa (QWK) score of 0.450 and Kaggle score of 0.494, demonstrating superior predictive alignment. While the TabNet Wrapper performed well in QWK (0.477), its Kaggle score was lower at 0.303. Sequential models like LSTM and Siamese LSTM showed moderate Kaggle scores, with Siamese LSTM scoring 0.327. Overall, the TabNet Ensemble outperformed other models.

## Conclusion

This project demonstrated the effectiveness of advanced models like TabNet, LSTM, and MLP for multi-class classification with tabular and time-series data. The TabNet Voting Regressor Ensemble outperformed others, achieving the highest QWK and Kaggle score of 0.494, highlighting the power of ensemble methods. Feature engineering, dimensionality reduction, and threshold optimization significantly improved model accuracy. Overall, the project emphasizes the importance of tailored model selection and optimization for reliable predictions in complex classification tasks.

|Model Evaluation comparision|Kaggle Competition Ranking|
|---|---|
|<img width="417" alt="image" src="https://github.com/user-attachments/assets/a4ab5b69-1317-4be8-a44a-4d544d7a328c" />|<img width="417" alt="image" src="https://github.com/user-attachments/assets/f8f2cabd-e4dd-4fbf-bd22-1bf5aec3f5bc" />| 


## Future Work:
Future efforts could focus on advanced feature engineering, time-series preprocessing like frequency domain analysis, and hyperparameter tuning for better model performance. Exploring transfer learning, integrating explainable AI, and addressing hardware-specific issues will further enhance scalability and robustness.

**Kaggle Competition Ranking:**
The project was developed as part of our Deep Learning Coursework with a team of total 6 members. We achieved the competition ranking of 371 out of 15,404 participants, with around 300 participants sharing the same score.

## License
MIT License

Copyright (c) 2024 Eshita Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

