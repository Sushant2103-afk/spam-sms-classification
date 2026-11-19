# Spam SMS Classification

An end-to-end exploratory data analysis, feature engineering, and supervised text classification workflow in Python for distinguishing spam SMS messages from legitimate (`ham`) messages[cite: 41].

## Dataset
- **Source:** `SMSSpamCollection`[cite: 41]
- **Initial Size:** 5,572 SMS messages across `label` and `message` columns (4,825 `ham` vs. 747 `spam`)[cite: 16, 41]
- **Class Handling:** Spam messages were oversampled in the notebook prior to feature extraction and model training[cite: 16, 41].

## Exploratory Data Analysis & Feature Engineering
- Mapped target labels (`ham` → `0`, `spam` → `1`) and visualized class distributions using countplots[cite: 41].
- Engineered structural features from raw SMS text[cite: 13, 41]:
  - `word_count`: Token count per message and distribution comparison across `ham` and `spam` messages[cite: 41].
  - `contaning_currency_symbol`: Binary flag checking for currency symbols (`₹`, `$`, `€`, `£`, `¥`)[cite: 41].
  - `contains_number`: Binary flag checking for ASCII digits (`0`–`9`)[cite: 41].

## Text Preprocessing & Vectorization
- Cleaned text using regular expressions (`re.sub('[^a-zA-Z]', ' ', sms)`) and lowercasing[cite: 41].
- Removed English stopwords (`nltk.corpus.stopwords`) and applied lemmatization (`WordNetLemmatizer`) to build the processed text corpus[cite: 41].
- Transformed cleaned messages into numerical features using `TfidfVectorizer(max_features=500)` and split the data into train and test sets (`test_size=0.2`, `random_state=42`)[cite: 41].

## Models & Results
- **Multinomial Naive Bayes (`MultinomialNB`):**[cite: 41]
  - 10-fold cross-validation F1-score: **0.943** (std: `0.004`)[cite: 41]
  - Test set precision, recall, and F1-score: **0.94** across both classes[cite: 41]
- **Decision Tree (`DecisionTreeClassifier`):**[cite: 41]
  - 10-fold cross-validation F1-score: **0.981** (std: `0.003`)[cite: 41]
  - Test set precision, recall, and F1-score: **0.98** across both classes[cite: 41]
- Evaluated model predictions using classification reports and confusion matrix heatmaps, plus a `predict_spam()` helper function for sample messages[cite: 41].

## Repository Contents
- `SmsSpamClassifier.ipynb` — Complete Jupyter notebook with EDA, preprocessing, model training, evaluation, and sample inference[cite: 41].
- `README.md` — Project documentation.

## How to Run Locally
1. Clone the repository and open `SmsSpamClassifier.ipynb` in Jupyter Notebook or VS Code[cite: 13, 41].
2. Place `SMSSpamCollection` in the same working directory as the notebook[cite: 41].
3. Install dependencies (`numpy`, `pandas`, `matplotlib`, `seaborn`, `nltk`, `scikit-learn`) and run the cells in order[cite: 41].