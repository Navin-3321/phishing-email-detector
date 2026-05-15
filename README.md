# Phishing Email Detection Model

A machine learning model built with Scikit-learn that classifies emails as **Phishing** or **Safe**.

## Features
- Trains on phishing + legitimate email dataset
- Extracts URL count, suspicious keywords, uppercase ratio, and more
- Combines **TF-IDF** text features with custom-engineered features
- Uses **Random Forest** classifier
- Displays accuracy score and confusion matrix
- Interactive email testing mode

## How to Run

```bash
pip install scikit-learn pandas numpy
python phishing_detector.py
```

## Feature Engineering
| Feature | Description |
|---------|-------------|
| URL count | Number of links in email |
| Phishing keywords | Words like "urgent", "verify", "winner", etc. |
| Exclamation count | Excessive punctuation is a red flag |
| Uppercase ratio | Phishing emails often SHOUT |
| Mentions attachment | `.exe`, `.zip`, `.pdf` attachments |
| IP in URL | Direct IP links are suspicious |

## Model
- **Algorithm:** Random Forest (100 trees)
- **Text Vectorizer:** TF-IDF (bigrams, 500 features)
- **Split:** 75% train / 25% test

## Using Your Own Dataset
Place a CSV with columns `text` and `label` (0=safe, 1=phishing) and pass the path in `load_dataset()`.

Recommended public datasets:
- [CEAS 2008 Spam Dataset](http://ceas.cc/2008/)
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/publiccorpus/)

## Tech Used
- Python 3
- Scikit-learn
- Pandas, NumPy
