import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────

def extract_features(text):
    """Extract numeric features from email text."""
    features = {}

    # URL count
    features["url_count"] = len(re.findall(r'http[s]?://\S+', text))

    # Suspicious keywords
    phishing_keywords = [
        "click here", "verify", "account suspended", "urgent",
        "confirm your", "update your", "login", "password",
        "bank", "credit card", "prize", "winner", "free",
        "limited time", "act now", "expires", "suspended",
        "unauthorized", "security alert", "dear customer"
    ]
    features["phishing_keyword_count"] = sum(
        1 for kw in phishing_keywords if kw in text.lower()
    )

    # Exclamation marks
    features["exclamation_count"] = text.count("!")

    # Length
    features["text_length"] = len(text)

    # Uppercase ratio
    alpha = [c for c in text if c.isalpha()]
    features["uppercase_ratio"] = (
        sum(1 for c in alpha if c.isupper()) / len(alpha) if alpha else 0
    )

    # Has attachments mentioned
    features["mentions_attachment"] = int(
        bool(re.search(r'\.(pdf|doc|xls|zip|exe)', text.lower()))
    )

    # Has IP address in URL
    features["has_ip_url"] = int(
        bool(re.search(r'http[s]?://\d+\.\d+\.\d+\.\d+', text))
    )

    return features


# ─────────────────────────────────────────────
# Dataset Generation (Sample)
# ─────────────────────────────────────────────

def generate_sample_dataset():
    """
    Generates a sample dataset for demo purposes.
    In real use, replace with a CSV like CEAS, SpamAssassin, etc.
    """
    phishing_emails = [
        "Dear Customer, Your account has been suspended. Click here to verify: http://bank-update.xyz/login",
        "URGENT: Your credit card has been compromised! Update your password immediately at http://192.168.1.1/secure",
        "Congratulations! You are a WINNER! Claim your free prize now. Act NOW, limited time offer!",
        "Security Alert: Unauthorized login detected. Confirm your account at http://paypa1-secure.com",
        "Dear user, your account expires today! Login at http://123.45.67.89/verify to keep access",
        "You have won $1,000,000! Click here to claim your reward. Limited time only!!!",
        "Your bank account needs verification. Provide your details here: http://secure-bank-update.net",
        "ACCOUNT SUSPENDED: Verify your identity at http://amaz0n-verify.com/login now!",
        "Exclusive offer for you! Download free software now: http://free-download.xyz/setup.exe",
        "Your PayPal account is at risk. Confirm your password: http://paypa1.security-check.com",
        "FINAL WARNING: Your subscription expires in 24 hours. Pay now to avoid suspension!",
        "Dear valued customer, click here to receive your refund: http://refund-center.biz/claim",
        "Your package could not be delivered. Update your address: http://fedex-update.info/track",
        "Congratulations! Your email was selected for a $500 gift card. Claim here!!!",
        "Alert: Someone tried to access your account. Verify at http://192.0.2.1/account-verify",
    ]

    legitimate_emails = [
        "Hi team, please find attached the Q3 report. Let me know if you have any questions.",
        "Meeting rescheduled to 3 PM tomorrow. Please update your calendars accordingly.",
        "Your order #45231 has been shipped. Expected delivery: 3-5 business days.",
        "Thank you for your purchase. Your receipt is attached for your records.",
        "Weekly newsletter: Check out our latest blog posts and product updates.",
        "Reminder: Team standup at 10 AM. Agenda: sprint planning and blockers.",
        "Your GitHub pull request has been reviewed. Two comments added by reviewer.",
        "Invoice #INV-2024-089 from Vendor Corp. Payment due by end of month.",
        "Hi John, great presentation today! The client loved the proposal.",
        "New comment on your post: 'Really useful article, thanks for sharing!'",
        "Your flight booking confirmation: PNR XYZ123. Departure: June 15, 8:00 AM",
        "Monthly statement available. Log in to your account to view details.",
        "Project deadline extended to next Friday. Please update your tasks.",
        "Happy Birthday! Wishing you a wonderful day from all of us at the office.",
        "Lunch today at 12:30? We're going to the new cafe on Main Street.",
    ]

    data = []
    for email in phishing_emails:
        data.append({"text": email, "label": 1})  # 1 = Phishing
    for email in legitimate_emails:
        data.append({"text": email, "label": 0})  # 0 = Safe

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Load or generate dataset
# ─────────────────────────────────────────────

def load_dataset(filepath=None):
    if filepath and os.path.exists(filepath):
        print(f"[+] Loading dataset from: {filepath}")
        df = pd.read_csv(filepath)
        # Expect columns: 'text' and 'label' (0=safe, 1=phishing)
        if "text" not in df.columns or "label" not in df.columns:
            print("[-] CSV must have 'text' and 'label' columns. Using sample data.")
            df = generate_sample_dataset()
    else:
        print("[*] No external dataset found. Using built-in sample data.")
        print("    (For better results, use a real dataset like CEAS or SpamAssassin)")
        df = generate_sample_dataset()

    print(f"[+] Dataset loaded: {len(df)} samples")
    print(f"    Phishing: {df['label'].sum()} | Safe: {(df['label']==0).sum()}")
    return df


# ─────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────

def build_feature_matrix(df):
    """Combine TF-IDF features with custom extracted features."""
    print("\n[*] Extracting features...")

    # TF-IDF on text
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["text"]).toarray()

    # Custom features
    custom_features = df["text"].apply(extract_features)
    custom_df = pd.DataFrame(list(custom_features))

    # Combine
    X = np.hstack([tfidf_matrix, custom_df.values])
    y = df["label"].values

    return X, y, tfidf


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print("[*] Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n" + "="*50)
    print("  MODEL EVALUATION")
    print("="*50)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy : {acc * 100:.2f}%")

    print("\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    return model


# ─────────────────────────────────────────────
# Interactive Email Checker
# ─────────────────────────────────────────────

def check_email(model, tfidf, email_text):
    """Predict if a given email is phishing or safe."""
    tfidf_vec = tfidf.transform([email_text]).toarray()
    custom = pd.DataFrame([extract_features(email_text)]).values
    X = np.hstack([tfidf_vec, custom])

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    result = "🚨 PHISHING" if prediction == 1 else "✅ SAFE"
    confidence = proba[prediction] * 100

    print(f"\n  Result     : {result}")
    print(f"  Confidence : {confidence:.1f}%")
    print(f"  Safe probability    : {proba[0]*100:.1f}%")
    print(f"  Phishing probability: {proba[1]*100:.1f}%")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("\n╔════════════════════════════════════╗")
    print("║  Phishing Email Detection Model     ║")
    print("╚════════════════════════════════════╝")

    # Load data
    df = load_dataset()  # Pass a CSV path here if you have one

    # Build features and train
    X, y, tfidf = build_feature_matrix(df)
    model = train_model(X, y)

    # Interactive testing
    print("\n" + "="*50)
    print("  TEST THE MODEL")
    print("="*50)

    while True:
        print("\nOptions:")
        print("  1. Check a custom email")
        print("  2. Run sample test emails")
        print("  3. Exit")

        choice = input("\nChoice (1/2/3): ").strip()

        if choice == "1":
            text = input("Paste email text: ").strip()
            if text:
                check_email(model, tfidf, text)

        elif choice == "2":
            test_emails = [
                ("URGENT: Your account has been suspended. Click here http://bank-verify.xyz", "Phishing"),
                ("Hi, reminder about tomorrow's 3 PM meeting. Agenda sent separately.", "Safe"),
                ("You won a FREE iPhone! Claim now before it expires!!!", "Phishing"),
                ("Your order has been shipped. Tracking number: TRK789456.", "Safe"),
            ]
            print("\n[Sample Email Tests]")
            for email, expected in test_emails:
                print(f"\n  Email   : {email[:60]}...")
                print(f"  Expected: {expected}")
                check_email(model, tfidf, email)

        elif choice == "3":
            print("Exiting.")
            break


if __name__ == "__main__":
    main()
