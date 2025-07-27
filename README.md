# Sentiment-classification
# 🧠 Twitter US Airline Sentiment Classification using TF-IDF

## 📘 Project Summary

This project involves feature engineering and sentiment classification on Twitter data about U.S. airlines. The tweets are categorized into **positive**, **neutral**, and **negative** sentiments. The goal is to convert raw, unstructured text data into informative numeric features—especially using **TF-IDF vectorization**—to support machine learning models in predicting sentiment accurately.

---

## 📁 Dataset Used

- **Dataset Name:** Twitter US Airline Sentiment
- **File:** `Tweets.csv`
- **Important Columns:**  
  - `text`: Tweet text  
  - `airline_sentiment`: Sentiment label (positive, neutral, negative)

---

## 🔧 Libraries and Tools

| Library    | Purpose |
|------------|---------|
| `pandas`   | Data loading and manipulation |
| `numpy`    | Numerical operations |
| `spaCy`    | Text preprocessing (lemmatization, stopword removal) |
| `TextBlob` | Sentiment polarity scoring |
| `sklearn`  | TF-IDF vectorization |
| `seaborn`  | Data visualization |
| `matplotlib` | Plotting |

---

## 🚀 Project Workflow

### 1. **Load Dataset**
- Only `text` and `airline_sentiment` columns are used.

### 2. **Text Preprocessing**
- Convert to lowercase
- Remove stopwords and non-alphabetic tokens
- Lemmatize using **spaCy**

### 3. **Feature Extraction**
- `word_count`: Number of words in each tweet
- `char_count`: Number of characters
- `avg_word_len`: Average word length
- `exclam_count`: Number of exclamation marks
- `quest_count`: Number of question marks
- `has_hashtag`: Presence of hashtags
- `has_mention`: Presence of user mentions
- `polarity`: Sentiment score using **TextBlob**
- **TF-IDF Vectors**: Convert cleaned text into numeric vectors using `TfidfVectorizer` (top 500 features)

### 4. **Visualization**
- Countplot of sentiment classes
- Pairplot to observe feature relationships with sentiment

---

## 📊 Output Example

- Cleaned tweet text
- Feature columns like `word_count`, `polarity`, etc.
- A DataFrame of **TF-IDF vectors**
- Graphs showing sentiment distribution and feature correlation

---

## 💡 Key Learning

- How to preprocess real-world noisy text
- How to use TF-IDF for feature extraction
- Extract and analyze custom features for classification

---

## ▶️ How to Run the Project

1. Install requirements:
   ```bash
   pip install pandas numpy matplotlib seaborn spacy textblob scikit-learn
   python -m spacy download en_core_web_sm
