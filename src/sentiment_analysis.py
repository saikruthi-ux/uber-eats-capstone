from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "reviews.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "reports" / "figures"
DOCS_DIR = ROOT / "docs"

for directory in [PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, DOCS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SENTIMENT_ORDER = ["negative", "neutral", "positive"]

# ---------------------------------------------------------
# 2. Download required NLTK resources
# ---------------------------------------------------------

print("Checking required NLTK resources...")

for resource in ["stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
    try:
        nltk.download(resource, quiet=True, raise_on_error=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download NLTK resource '{resource}'. "
            "Check your internet connection and run the script again."
        ) from exc

# ---------------------------------------------------------
# 3. Load and validate the review dataset
# ---------------------------------------------------------

print("\nLoading review dataset...")
reviews = pd.read_csv(RAW_PATH)

required_columns = {"review_id", "order_id", "review_text", "rating"}
missing_columns = required_columns - set(reviews.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

if reviews["review_id"].duplicated().any():
    raise ValueError("Duplicate review_id values were found.")

if reviews["rating"].isna().any() or reviews["review_text"].isna().any():
    raise ValueError(
        "Missing ratings or review text found. Handle these before modeling."
    )

reviews["review_text"] = reviews["review_text"].astype(str).str.strip()
reviews = reviews[reviews["review_text"].ne("")].copy()

# Ratings become proxy labels for supervised learning.
def rating_to_sentiment(rating):
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"

reviews["sentiment"] = reviews["rating"].apply(rating_to_sentiment)

print(f"Rows loaded: {len(reviews):,}")
print(f"Unique review texts: {reviews['review_text'].nunique():,}")
print("\nSentiment label distribution:")
print(reviews["sentiment"].value_counts().reindex(SENTIMENT_ORDER))

# ---------------------------------------------------------
# 4. Clean the review text
# ---------------------------------------------------------

print("\nCleaning review text...")

tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
lemmatizer = WordNetLemmatizer()

# Preserve negation terms because they can reverse sentiment.
stop_words = set(stopwords.words("english")) - {
    "no", "nor", "not", "never", "neither"
}

def clean_review(text):
    """Lowercase, tokenize, remove stop words, and lemmatize."""
    text = str(text).lower()
    tokens = tokenizer.tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    ]
    return " ".join(tokens)

reviews["clean_text"] = reviews["review_text"].apply(clean_review)

# Keep original text for VADER, which uses punctuation and emoticons.
cleaned_columns = [
    "review_id",
    "order_id",
    "review_text",
    "clean_text",
    "rating",
    "sentiment",
]
cleaned_path = PROCESSED_DIR / "cleaned_reviews.csv"
reviews[cleaned_columns].to_csv(cleaned_path, index=False)

print("Saved:", cleaned_path.relative_to(ROOT))
print("\nOriginal and cleaned examples:")
print(
    reviews[["review_text", "clean_text", "sentiment"]]
    .head(8)
    .to_string(index=False)
)

# ---------------------------------------------------------
# 5. Split by unique review text to reduce template leakage
# ---------------------------------------------------------

print("\nPreparing train/test split...")

# A repeated review template should not appear in both sets.
template_labels = reviews.groupby("review_text")["sentiment"].nunique()

if (template_labels > 1).any():
    print(
        "Warning: at least one identical review text has multiple labels. "
        "Using a stratified row split instead."
    )
    train_idx, test_idx = train_test_split(
        reviews.index,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=reviews["sentiment"],
    )
else:
    templates = (
        reviews[["review_text", "sentiment"]]
        .drop_duplicates(subset="review_text")
        .reset_index(drop=True)
    )

    template_counts = templates["sentiment"].value_counts()

    if (
        templates["sentiment"].nunique() == len(SENTIMENT_ORDER)
        and template_counts.min() >= 2
    ):
        train_templates, test_templates = train_test_split(
            templates["review_text"],
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=templates["sentiment"],
        )

        train_idx = reviews.index[
            reviews["review_text"].isin(set(train_templates))
        ]
        test_idx = reviews.index[
            reviews["review_text"].isin(set(test_templates))
        ]
    else:
        print(
            "Not enough distinct templates per class for a stratified "
            "template split. Using a stratified row split instead."
        )
        train_idx, test_idx = train_test_split(
            reviews.index,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=reviews["sentiment"],
        )

X_train = reviews.loc[train_idx, "clean_text"]
X_test = reviews.loc[test_idx, "clean_text"]
y_train = reviews.loc[train_idx, "sentiment"]
y_test = reviews.loc[test_idx, "sentiment"]

print(f"Training rows: {len(X_train):,}")
print(f"Testing rows: {len(X_test):,}")
print("Training sentiment distribution:")
print(y_train.value_counts().reindex(SENTIMENT_ORDER, fill_value=0))
print("Testing sentiment distribution:")
print(y_test.value_counts().reindex(SENTIMENT_ORDER, fill_value=0))

# ---------------------------------------------------------
# 6. TF-IDF and Naive Bayes classifier
# ---------------------------------------------------------

print("\nTraining TF-IDF + Multinomial Naive Bayes model...")

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                max_features=3000,
                sublinear_tf=True,
            ),
        ),
        ("classifier", MultinomialNB(alpha=0.5)),
    ]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

model_path = MODEL_DIR / "sentiment_naive_bayes.joblib"
joblib.dump(model, model_path)
print("Saved:", model_path.relative_to(ROOT))

# Save the full TF-IDF matrix using the vocabulary learned from training data.
tfidf = model.named_steps["tfidf"]
full_tfidf = tfidf.transform(reviews["clean_text"])

from scipy.sparse import save_npz

tfidf_path = PROCESSED_DIR / "reviews_tfidf_matrix.npz"
save_npz(tfidf_path, full_tfidf)

vocabulary_path = PROCESSED_DIR / "tfidf_feature_names.csv"
pd.DataFrame({"feature": tfidf.get_feature_names_out()}).to_csv(
    vocabulary_path, index=False
)

print("Saved TF-IDF matrix:", tfidf_path.relative_to(ROOT))
print("TF-IDF matrix shape:", full_tfidf.shape)
print("Saved vocabulary:", vocabulary_path.relative_to(ROOT))

# ---------------------------------------------------------
# 7. Evaluate the trained model
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    predictions,
    labels=SENTIMENT_ORDER,
    average="weighted",
    zero_division=0,
)

print("\n" + "=" * 60)
print("NAIVE BAYES MODEL EVALUATION")
print("=" * 60)
print(f"Accuracy:          {accuracy:.4f}")
print(f"Weighted precision:{precision:.4f}")
print(f"Weighted recall:   {recall:.4f}")
print(f"Weighted F1-score: {f1:.4f}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=SENTIMENT_ORDER,
        zero_division=0,
    )
)

report_dict = classification_report(
    y_test,
    predictions,
    labels=SENTIMENT_ORDER,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report_dict).transpose().to_csv(
    PROCESSED_DIR / "sentiment_classification_report.csv"
)

model_metrics = {
    "model": "TF-IDF + Multinomial Naive Bayes",
    "accuracy": accuracy,
    "weighted_precision": precision,
    "weighted_recall": recall,
    "weighted_f1": f1,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "unique_review_texts": reviews["review_text"].nunique(),
}

# Confusion matrix
cm = confusion_matrix(y_test, predictions, labels=SENTIMENT_ORDER)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=SENTIMENT_ORDER,
)
display.plot(values_format="d")
plt.title("Naive Bayes Sentiment Confusion Matrix")
plt.tight_layout()
cm_path = FIGURE_DIR / "sentiment_confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print("Saved:", cm_path.relative_to(ROOT))

# ---------------------------------------------------------
# 8. VADER sentiment analysis and comparison
# ---------------------------------------------------------

print("\nRunning VADER sentiment analysis...")

vader = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    compound = vader.polarity_scores(str(text))["compound"]
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"

reviews["vader_sentiment"] = reviews["review_text"].apply(vader_sentiment)

# Compare VADER with rating-derived labels on the same held-out rows.
vader_predictions = reviews.loc[test_idx, "vader_sentiment"]
vader_accuracy = accuracy_score(y_test, vader_predictions)

v_precision, v_recall, v_f1, _ = precision_recall_fscore_support(
    y_test,
    vader_predictions,
    labels=SENTIMENT_ORDER,
    average="weighted",
    zero_division=0,
)

print("\nVADER evaluation on the held-out reviews:")
print(f"Accuracy:          {vader_accuracy:.4f}")
print(f"Weighted precision:{v_precision:.4f}")
print(f"Weighted recall:   {v_recall:.4f}")
print(f"Weighted F1-score: {v_f1:.4f}")
print("\nVADER classification report:")
print(
    classification_report(
        y_test,
        vader_predictions,
        labels=SENTIMENT_ORDER,
        zero_division=0,
    )
)

vader_report_dict = classification_report(
    y_test,
    vader_predictions,
    labels=SENTIMENT_ORDER,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(vader_report_dict).transpose().to_csv(
    PROCESSED_DIR / "vader_classification_report.csv"
)

vader_cm = confusion_matrix(y_test, vader_predictions, labels=SENTIMENT_ORDER)
vader_display = ConfusionMatrixDisplay(
    confusion_matrix=vader_cm,
    display_labels=SENTIMENT_ORDER,
)
vader_display.plot(values_format="d")
plt.title("VADER Sentiment Confusion Matrix")
plt.tight_layout()
vader_cm_path = FIGURE_DIR / "vader_confusion_matrix.png"
plt.savefig(vader_cm_path, dpi=150, bbox_inches="tight")
plt.close()

# Save per-review predictions for later inspection.
reviews["model_sentiment"] = model.predict(reviews["clean_text"])
reviews["vader_compound"] = reviews["review_text"].apply(
    lambda text: vader.polarity_scores(str(text))["compound"]
)
reviews[
    [
        "review_id",
        "order_id",
        "review_text",
        "rating",
        "sentiment",
        "model_sentiment",
        "vader_sentiment",
        "vader_compound",
    ]
].to_csv(PROCESSED_DIR / "sentiment_predictions.csv", index=False)

# ---------------------------------------------------------
# 9. Important TF-IDF terms and complaint-theme exploration
# ---------------------------------------------------------

print("\nExtracting important terms and complaint themes...")

# Mean TF-IDF scores for features in the cleaned negative reviews.
negative_texts = reviews.loc[
    reviews["sentiment"] == "negative", "clean_text"
]
negative_matrix = tfidf.transform(negative_texts)

if negative_matrix.shape[0] > 0 and negative_matrix.shape[1] > 0:
    mean_scores = np.asarray(negative_matrix.mean(axis=0)).ravel()
    feature_names = tfidf.get_feature_names_out()

    top_indices = mean_scores.argsort()[::-1][:20]
    negative_terms = pd.DataFrame(
        {
            "term_or_phrase": feature_names[top_indices],
            "mean_tfidf": mean_scores[top_indices],
        }
    )
else:
    negative_terms = pd.DataFrame(
        columns=["term_or_phrase", "mean_tfidf"]
    )

negative_terms.to_csv(
    PROCESSED_DIR / "negative_review_top_terms.csv", index=False
)

# Phrase frequency is an exploratory signal, not proof of a causal issue.
complaint_vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_features=1000,
)
complaint_matrix = complaint_vectorizer.fit_transform(negative_texts)
complaint_counts = np.asarray(complaint_matrix.sum(axis=0)).ravel()
complaint_features = complaint_vectorizer.get_feature_names_out()

if len(complaint_features):
    top_complaint_indices = complaint_counts.argsort()[::-1][:20]
    complaint_terms = pd.DataFrame(
        {
            "term_or_phrase": complaint_features[top_complaint_indices],
            "count": complaint_counts[top_complaint_indices],
        }
    )
else:
    complaint_terms = pd.DataFrame(columns=["term_or_phrase", "count"])

complaint_terms.to_csv(
    PROCESSED_DIR / "negative_complaint_themes.csv", index=False
)

# Save a small chart of frequent negative-review terms.
if not complaint_terms.empty:
    chart_data = complaint_terms.head(10).sort_values("count")
    plt.figure(figsize=(9, 5))
    plt.barh(chart_data["term_or_phrase"], chart_data["count"])
    plt.xlabel("Frequency in negative-rated reviews")
    plt.ylabel("Term or phrase")
    plt.title("Frequent Terms in Negative-Rated Reviews")
    plt.tight_layout()
    complaint_chart_path = FIGURE_DIR / "negative_review_complaint_terms.png"
    plt.savefig(complaint_chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", complaint_chart_path.relative_to(ROOT))

# ---------------------------------------------------------
# 10. Save model comparison and summary report
# ---------------------------------------------------------

comparison = pd.DataFrame(
    [
        {
            "model": "TF-IDF + Multinomial Naive Bayes",
            "accuracy": accuracy,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1": f1,
        },
        {
            "model": "VADER",
            "accuracy": vader_accuracy,
            "weighted_precision": v_precision,
            "weighted_recall": v_recall,
            "weighted_f1": v_f1,
        },
    ]
)

comparison.to_csv(
    PROCESSED_DIR / "sentiment_model_comparison.csv", index=False
)

report_path = DOCS_DIR / "day3_sentiment_report.md"

report_text = f"""# Day 3 — Review Sentiment Analysis

## 1. Dataset overview

- Review records analyzed: {len(reviews)}
- Unique review texts: {reviews['review_text'].nunique()}
- Missing review text or ratings after validation: 0
- Exact duplicate rows in the original inspection: 0
- Duplicate review IDs: 0

## 2. Sentiment-label definition

Ratings were converted to proxy sentiment labels:

- Ratings 1–2: negative
- Rating 3: neutral
- Ratings 4–5: positive

These are rating-derived labels, not independent human annotations.

## 3. Text preprocessing

Review text was lowercased and tokenized. English stop words were removed while
retaining negation words such as "not", "no", and "never". Remaining tokens were
lemmatized. The original review text was retained for VADER because punctuation
and other surface cues can affect its sentiment scores.

## 4. Machine-learning approach

A TF-IDF vectorizer with unigram and bigram features was combined with a
Multinomial Naive Bayes classifier. The TF-IDF vocabulary was learned from the
training data only. A template-aware split was used when possible to reduce
identical review texts appearing in both training and test sets.

### TF-IDF + Multinomial Naive Bayes

- Accuracy: {accuracy:.4f}
- Weighted precision: {precision:.4f}
- Weighted recall: {recall:.4f}
- Weighted F1-score: {f1:.4f}

## 5. VADER comparison

- Accuracy: {vader_accuracy:.4f}
- Weighted precision: {v_precision:.4f}
- Weighted recall: {v_recall:.4f}
- Weighted F1-score: {v_f1:.4f}

VADER uses a rule-based sentiment lexicon and scores the original review text.
The model comparison should be interpreted as agreement with rating-derived
labels, not proof that either approach captures true customer sentiment.

## 6. Complaint-theme exploration

The files `negative_review_top_terms.csv` and `negative_complaint_themes.csv`
contain frequent TF-IDF terms and word/phrase counts from negative-rated reviews.
These terms are exploratory indicators and should be checked against actual
review examples before making operational recommendations.

## 7. Limitations

1. The dataset contains only {reviews['review_text'].nunique()} unique review
   texts across {len(reviews)} rows. Repeated templates limit linguistic variety.
2. Sentiment labels are derived from ratings, so evaluation measures agreement
   with ratings rather than independently verified sentiment.
3. Short, templated reviews provide limited context for nuanced sentiment.
4. Complaint-term frequencies do not establish root causes or causation.
5. Results should not be assumed to generalize to real-world customer reviews
   without testing on a larger, more diverse, independently labeled dataset.

## 8. Output files

- `data/processed/cleaned_reviews.csv`
- `data/processed/reviews_tfidf_matrix.npz`
- `data/processed/tfidf_feature_names.csv`
- `data/processed/sentiment_classification_report.csv`
- `data/processed/vader_classification_report.csv`
- `data/processed/sentiment_predictions.csv`
- `data/processed/sentiment_model_comparison.csv`
- `data/processed/negative_review_top_terms.csv`
- `data/processed/negative_complaint_themes.csv`
- `models/sentiment_naive_bayes.joblib`
- `reports/figures/sentiment_confusion_matrix.png`
- `reports/figures/vader_confusion_matrix.png`
- `docs/day3_sentiment_report.md`
"""

report_path.write_text(report_text, encoding="utf-8")

print("\n" + "=" * 60)
print("DAY 3 PIPELINE COMPLETED")
print("=" * 60)
print("\nModel comparison:")
print(comparison.to_string(index=False))
print("\nTop negative-review terms:")
print(negative_terms.head(10).to_string(index=False))
print("\nFrequent complaint terms:")
print(complaint_terms.head(10).to_string(index=False))
print("\nReport saved:", report_path.relative_to(ROOT))
