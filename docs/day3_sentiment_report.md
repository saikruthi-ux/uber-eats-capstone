# Day 3 — Review Sentiment Analysis

## 1. Dataset overview

- Review records analyzed: 1871
- Unique review texts: 15
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

- Accuracy: 0.4359
- Weighted precision: 0.6440
- Weighted recall: 0.4359
- Weighted F1-score: 0.4785

## 5. VADER comparison

- Accuracy: 0.8486
- Weighted precision: 0.7764
- Weighted recall: 0.8486
- Weighted F1-score: 0.7997

VADER uses a rule-based sentiment lexicon and scores the original review text.
The model comparison should be interpreted as agreement with rating-derived
labels, not proof that either approach captures true customer sentiment.

## 6. Complaint-theme exploration

The files `negative_review_top_terms.csv` and `negative_complaint_themes.csv`
contain frequent TF-IDF terms and word/phrase counts from negative-rated reviews.
These terms are exploratory indicators and should be checked against actual
review examples before making operational recommendations.

## 7. Limitations

1. The dataset contains only 15 unique review
   texts across 1871 rows. Repeated templates limit linguistic variety.
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
