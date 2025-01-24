<h1 align=center>Text Toxicity classification using LSTM and BILSTM</h1>

<p align=center><img width="600" alt="image" src="https://github.com/user-attachments/assets/235b62b6-7d08-4aee-a33a-da085728eb44" /></p>


## Introduction
This project aims to develop machine learning and deep learning models for multi-class classification to analyze and predict various categories of toxicity in text data. The dataset used contains approximately 1.8 million rows and nine columns, including a unique identifier (id), the actual text data (text), and toxicity-related labels such as toxicity, severe_toxicity, obscene, threat, insult, identity_attack, and sexual_explicit. Each label provides a numerical value indicating the level or probability of the respective toxicity type present in the text.

The primary goal is to classify text entries into predefined toxicity categories while ensuring high accuracy and robust performance. This project involves data preprocessing, feature engineering, and training advanced machine learning and deep learning models. The repository also includes visualization of results, performance metrics, and insights gained from the model predictions, making it a comprehensive resource for tackling text-based toxicity classification challenges.

## Dataset
The dataset is related to toxic comments that can be classified into several predefined categories, including toxicity, severe toxicity, obscene, threat, insult, identity attack, and sexual explicit. This is a multi-class classification dataset. The train dataset has about 1.8M rows having 9 columns. The columns comprise of columns:

- id: A unique identifier for each text entry.
- text: The actual text data that needs to be analyzed for toxicity.
- toxicity-related labels: These include `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, and `sexual_explicit`. Each of these columns contains a numerical value indicating the level or probability of the respective toxicity type present in the text

The dataset was part of a private Kaggle Competetion: 

**Link**: `https://www.kaggle.com/competitions/data-255-toxic-comment-in-class-competition`

## Data Preparation

The pre-processing steps prepared the textual data for model training and improved classification accuracy. Below are the techniques used:
- **Data Loading and Exploration**: The dataset was loaded using Pandas, and its structure was explored to understand the columns (id, text, and toxicity-related labels).
- **Text Cleaning:**
  - **Lowercasing**: Converted all text to lowercase for uniformity.
  - **Removing Special Characters and Punctuation:** Removed non-alphabetical characters, special symbols, and numbers using regular expressions to reduce noise.
  - **Whitespace Stripping:** Eliminated extra spaces from the text.
  - **Non-Printable Character Removal:** Cleared non-printable characters for clean data.
  - **Tokenization:** Split text into sentences and words using NLTK’s punkt tokenizer, enabling word-level analysis for better context understanding.
  - **Stopword Removal:** Omitted this step after experiments showed it removed critical contextual information. Research corroborated this decision, highlighting its negative impact on deep learning models for toxic classification.
  - **Stemming:** Applied stemming to reduce words to their base forms by removing prefixes and suffixes. This technique proved effective for models like fastText-BiLSTM and XGBoost.
  - **Lemmatization:** Skipped this step based on research indicating that stemming performed better for the chosen models in this task.
  - **Sequence of Transformations:** Applied sequential transformations: lowercasing → removing whitespace → trimming word length → removing non-printable characters → removing non-alphabets, as recommended in relevant research.

Finally, pre-processed datasets (train and test) were saved as pickle files for consistent reusability in subsequent modeling experiments.

## Modeling
We chose a Bidirectional Long Short-Term Memory (BiLSTM) model for this dataset due to its ability to capture contextual information from both past and future sequences. Unlike unidirectional LSTMs, BiLSTMs provide a more comprehensive understanding of the text, which is crucial for detecting subtle patterns in toxic language. They also handle long-range dependencies and address the vanishing gradient problem, making them well-suited for complex text data.

The text data was preprocessed by tokenizing and padding sequences to ensure uniform input sizes. A predefined vocabulary size was used to convert text into numerical sequences, and padding was applied for consistency. The dataset was split into 90% training and 10% validation, with shuffling to avoid ordering bias and ensure balanced class distribution. PyTorch was used to prepare data, converting it into tensors, creating TensorDataset objects, and using DataLoader with a batch size of 64 for efficient batching and shuffling.

We used pre-trained FastText embeddings to represent words as dense vectors, leveraging its subword-based approach to capture semantic meanings effectively. FastText was chosen over GloVe due to its ability to handle rare or unseen words and its training on diverse corpora like Wikipedia and Common Crawl.

During training, the model alternated between training and validation phases. In training, gradients were reset, forward passes and backpropagation updated the weights, and accuracy was tracked. For validation, dropout and batch normalization were disabled, and performance metrics were calculated. Epoch summaries included average loss and accuracy for both phases, allowing us to monitor trends and detect issues like overfitting or underfitting.

Key hyperparameters included 100 LSTM units, 7 output classes, a batch size of 64, a learning rate of 0.001, and 10 epochs. Training accuracy exceeded 86% by Epoch 3 and stabilized at ~86.7%, while validation accuracy reached ~86.8%. The validation loss stabilized at 0.6925 from Epoch 2 onwards, indicating convergence. The consistent accuracy and stable loss demonstrate the model’s reliable performance on both training and validation data.

## Conclusion

The test dataset we have, has only the text on which our trained model has to classify the text under what categories it does belong to. As we do not have labels of the test set, we cannot measure the accuracy. We measure the performance of our model by submitting the predictions on Kaggle Public leaderboard.  This public leaderboard is calculated with approximately 72% of the test data. The final results will be based on the other 28%. 

For 72% of the test dataset, the accuracy was approx. 95.732, which is fairly good performance of our trained BiLSTM model. 

<img width="1201" alt="image" src="https://github.com/user-attachments/assets/c837af17-6585-40a6-85d6-5af856bc6de1" />


Several parameters were tuned to enhance model performance:
-  Embedding Dimension: Set to 300 based on experiments with different dimensions.
-  LSTM Units: Optimal number found to be 100 units.
-  Dropout Rate: A dropout rate of 0.1 was found to balance regularization without significant loss of information.
-  Batch Size and Epochs: Batch size of 256 and training for up to 10 epochs with early stopping based on validation loss improvements.

The BiLSTM model with FastText embeddings proved to be effective for multi-label classification of toxic comments which captured both forward and backward contextual dependencies. Through extensive experimentation and parameter tuning, the model achieved satisfactory performance, demonstrating the importance of each architectural element and the impact of hyperparameter choices. Challenges such as overfitting and imbalanced data were addressed through dropout, early stopping, and careful data preprocessing, resulting in a robust model for toxic comment classification. 

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
