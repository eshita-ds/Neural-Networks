# Anime recommender system using Wide and Deep learning models

<p align=center><img src="https://github.com/user-attachments/assets/a1b6ca74-65af-4c09-a30a-99befbe3a3a6" alt="Description of Image" width="800" height="250">


## Introduction

Deep learning enhances recommendation systems by learning complex relationships in data, handling high-dimensional and sparse user-item interactions, and incorporating diverse data like images or text.

**Wide and Deep Learning**, introduced by Google, combines:

 1. **Wide Component**: Captures memorized patterns, such as co-occurrences between user preferences and anime genres, using a linear model.
 2. **Deep Component**: Learns abstract, generalized patterns through neural networks using dense embeddings of features like user preferences or anime attributes.

This hybrid approach improves recommendation accuracy by blending memorization and generalization, making it ideal for personalized anime recommendations.

## Dataset
The project utilizes three datasets:

**Link**: `https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset`

**Anime Dataset**: Details on anime titles, genres, studios, and airing dates.
**User Details**: Demographic and activity data for users. Contains about 731K records
**User Scores**: Ratings provided by users for different anime. Contains about 24 million records

These datasets are merged into a single dataset, `merged_data`, for building a Wide and Deep Learning recommender system.

### Data Preparation steps

1. **Anime Dataset Cleaning**
    - Irrelevant columns were removed to retain only essential features.
    - Missing values (unknowns) in columns such as `Score`, `Episodes`, and `Rank` were replaced with mean values.
    - The Aired column was split into `Air Start` and `Air End`, with missing dates filled using the median.
    - `Duration` was converted into minutes using a custom function, and zero durations were replaced with the mean.
   
2. **User Dataset Cleaning**
    - The `Username` column was dropped for anonymity.
    - Columns such as `Days Watched`, Watching, and Completed were removed.
    - Missing values in `Gender` and `Location` were replaced with "unknown," while other numeric missing values were imputed with the mean or median.
  
3. **Merging Datasets**
    - `user_details` and `user_scores` were merged on `user_id`, and the resulting dataset was combined with the anime dataset on `anime_id`. The resultant merged dataset size was about ~23 million records.
   
4. **Transformation and Encoding**
    - Categorical columns (e.g., `Gender`, `Genres`, `Type`) were label-encoded.
    - A combined `Type_Gender` column was created for stratified sampling.
    - Stratified sampling (50%) was performed based on the `Type_Gender` column to ensure balanced representation. The sampling set contained about ~11 million records. 

**Final Dataset**
The cleaned and transformed `merged_data` is saved as pickle files to be able to load further for training processes.

### Model Training

- Loaded sampled data (`data_sampled.pkl`).
- Separated features into:
  - **Wide Features** (categorical): `user_id`, `anime_id`, `Gender`, `Type`, `Status`,`Producers`, `Studios`, `Source`, `Rating`
  - **Deep Features** (continuous): `Days Watched`,`Episodes`, `Mean Score`, `Rank`, `Popularity`, `Favorites`, `Duration (min)`, `Completed`, `Total Entries`, `Genres`, `Air Start`, `Air End`
- Encoded categorical features using `LabelEncoder` and scaled continuous features using `StandardScaler`.
- Created `AnimeDataset` class to manage wide features, deep features, and labels (`Score`).
- Supported PyTorch `DataLoader` operations for efficient batch processing.

### Model Architecture
- **Wide Component**: Embeddings for wide features with an embedding dimension of 16.
- **Deep Component**: Fully connected layers (128, 64, 32) with ReLU activation and dropout (0.2).
- Combined outputs passed through a linear layer for prediction.

### Training Pipeline
- Split data into training (70%), validation (15%), and testing (15%).
- Defined loss function (`MSELoss`) and optimizer (`Adam`, LR: 0.001).
- Trained for 10 epochs with batch size 256, tracking loss and RMSE.

### Evaluation and Results
The model effectively predicts user-specific scores for unrated anime, providing personalized and accurate recommendations.
### Results
- The model achieved the following metrics:
  - **Validation Loss**: 0.0053  
  - **Test RMSE**: 0.0729  
- Top 5 anime recommendations for user ID `366396` based on predicted ratings:
  - **Predicted Scores**: All above 9.0, showcasing high personalization accuracy.
    
    <img width="684" alt="image" src="https://github.com/user-attachments/assets/77606eb5-114a-4fba-90ae-271fcbca4138" />

## Conclusion
The Wide and Deep Learning-based recommender system effectively combines categorical and continuous features to provide personalized anime recommendations. The system accurately predicts user-specific scores for unrated anime. It also demonstrates scalability and generalization with minimal validation loss and low RMSE.

Contributions and suggestions are welcome to improve the system further.

