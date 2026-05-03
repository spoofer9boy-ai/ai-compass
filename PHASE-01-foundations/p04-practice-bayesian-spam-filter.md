# Practice: Bayesian Spam Filter

**Phase:** PHASE-01-foundations  
**Subjects Required:** 21 (Probability Axioms), 22 (Conditional Probability), 23 (Bayes Theorem)  
**Estimated Time:** 120 minutes  
**Difficulty:** Intermediate

## Industry Context

You are a data engineer at a mid-sized email security startup. The product team wants a lightweight spam filter that can run on the edge (inside a browser extension and a mobile SDK) without calling a cloud API for every incoming message. They need a baseline classifier trained from a small labeled dataset of user-reported spam, and it must explain its decisions in plain English so users trust it.

## The Problem

Build a Naive Bayes spam classifier from scratch using only Python standard library and NumPy. You are given a small labeled dataset of emails (provided as a Python dictionary). Your classifier must:

1. Parse the text into word tokens (lowercase, split on whitespace, strip punctuation).
2. Compute class priors $P(\text{spam})$ and $P(\text{ham})$ from the training data.
3. Compute likelihoods $P(\text{word} \mid \text{spam})$ and $P(\text{word} \mid \text{ham})$ using Laplace (add-1) smoothing.
4. Classify a new email by computing the posterior log-probability for each class and picking the larger one.
5. Return the predicted label **and** the top 3 words that most influenced the decision (highest contribution to the log-posterior difference).

You must implement the training and prediction logic yourself. Do not use scikit-learn, NLTK, or any other ML library for the core algorithm.

## Constraints

- Only Python standard library and NumPy are allowed.
- Use Laplace smoothing with $\alpha = 1$ to avoid zero probabilities.
- Work in log-space to prevent underflow: $\log P(y \mid \text{email}) \propto \log P(y) + \sum_i \log P(w_i \mid y)$.
- Tokenization must be case-insensitive and punctuation-stripped.
- The solution must handle words seen only in one class gracefully (smoothing covers this).

## Starter Code

```python
import numpy as np
import math
import re
from collections import defaultdict, Counter

# --- Training data -----------------------------------------------------------
train_data = [
    ("spam", "Congratulations you won a free iPhone click here now"),
    ("spam", "Buy cheap viagra pills online free offer limited time"),
    ("spam", "You have won a lottery claim your prize money now"),
    ("ham",  "Hey are we still on for lunch tomorrow"),
    ("ham",  "Can you send me the report by end of day"),
    ("ham",  "Meeting rescheduled to 3pm in conference room B"),
    ("ham",  "Thanks for the quick turnaround on the bug fix"),
    ("spam", "Free entry to win a brand new car click below"),
    ("ham",  "Please review the attached document and let me know"),
    ("spam", "Claim your free gift now limited supply hurry"),
]

# --- Your task ---------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    # TODO: implement
    pass

class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        # TODO: store whatever you need for training

    def fit(self, texts: list[str], labels: list[str]):
        """Train on parallel lists of texts and labels."""
        # TODO: compute priors and word likelihoods with Laplace smoothing
        pass

    def predict(self, text: str) -> dict:
        """
        Return a dict:
        {
            "label": "spam" or "ham",
            "log_spam": <float>,
            "log_ham": <float>,
            "top_words": [(word, contribution), ...]  # top 3 by abs(contribution)
        }
        """
        # TODO: implement log-posterior computation and top-word extraction
        pass

# --- Tests -------------------------------------------------------------------

def test_classifier():
    texts = [t for (_, t) in train_data]
    labels = [l for (l, _) in train_data]

    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(texts, labels)

    result = clf.predict("Free lottery win claim your prize now")
    assert result["label"] == "spam", f"Expected spam, got {result['label']}"
    assert result["log_spam"] > result["log_ham"]
    assert len(result["top_words"]) == 3

    result2 = clf.predict("See you at the meeting tomorrow thanks")
    assert result2["label"] == "ham", f"Expected ham, got {result2['label']}"
    assert result2["log_ham"] > result2["log_spam"]

    print("All tests passed.")

if __name__ == "__main__":
    test_classifier()
```

## Evaluation Criteria

1. **Correctness:** `test_classifier()` passes without modification.
2. **No external ML libraries:** Only `numpy`, `collections`, `re`, and `math` are used for the algorithm.
3. **Log-space stability:** No raw probability multiplication; all scoring is done with log-probabilities.
4. **Top-word explainability:** The `top_words` list correctly identifies the words that most shifted the decision toward the winning class.
5. **Laplace smoothing:** Every unseen word gets a non-zero probability.

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
import numpy as np
import math
import re
from collections import defaultdict, Counter

# --- Training data -----------------------------------------------------------
train_data = [
    ("spam", "Congratulations you won a free iPhone click here now"),
    ("spam", "Buy cheap viagra pills online free offer limited time"),
    ("spam", "You have won a lottery claim your prize money now"),
    ("ham",  "Hey are we still on for lunch tomorrow"),
    ("ham",  "Can you send me the report by end of day"),
    ("ham",  "Meeting rescheduled to 3pm in conference room B"),
    ("ham",  "Thanks for the quick turnaround on the bug fix"),
    ("spam", "Free entry to win a brand new car click below"),
    ("ham",  "Please review the attached document and let me know"),
    ("spam", "Claim your free gift now limited supply hurry"),
]

# --- Utilities ---------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_counts = Counter()
        self.word_counts = defaultdict(Counter)   # class -> Counter(word)
        self.vocab = set()
        self.total_words = defaultdict(int)         # class -> total tokens
        self.priors = {}
        self.log_priors = {}

    def fit(self, texts: list[str], labels: list[str]):
        """Train on parallel lists of texts and labels."""
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            tokens = tokenize(text)
            for w in tokens:
                self.word_counts[label][w] += 1
                self.total_words[label] += 1
                self.vocab.add(w)

        n_total = sum(self.class_counts.values())
        for cls in self.class_counts:
            self.priors[cls] = self.class_counts[cls] / n_total
            self.log_priors[cls] = math.log(self.priors[cls])

    def _log_likelihood(self, word: str, cls: str) -> float:
        """Log P(word | cls) with Laplace smoothing."""
        count = self.word_counts[cls].get(word, 0)
        # Laplace smoothing: (count + alpha) / (total_words_in_class + alpha * |V|)
        denom = self.total_words[cls] + self.alpha * len(self.vocab)
        return math.log((count + self.alpha) / denom)

    def predict(self, text: str) -> dict:
        tokens = tokenize(text)
        scores = {}
        word_contributions = defaultdict(float)

        for cls in self.class_counts:
            score = self.log_priors[cls]
            for w in tokens:
                ll = self._log_likelihood(w, cls)
                score += ll
                # Store contribution of this word for this class
                word_contributions[(w, cls)] = ll
            scores[cls] = score

        # Determine winning class
        label = max(scores, key=scores.get)
        other = "ham" if label == "spam" else "spam"

        # Top words: those whose contribution to the winning class
        # most exceeds their contribution to the losing class.
        diffs = {}
        for w in set(tokens):
            diffs[w] = word_contributions[(w, label)] - word_contributions[(w, other)]

        top_words = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

        return {
            "label": label,
            "log_spam": scores["spam"],
            "log_ham": scores["ham"],
            "top_words": top_words,
        }

# --- Tests -------------------------------------------------------------------

def test_classifier():
    texts = [t for (_, t) in train_data]
    labels = [l for (l, _) in train_data]

    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(texts, labels)

    result = clf.predict("Free lottery win claim your prize now")
    assert result["label"] == "spam", f"Expected spam, got {result['label']}"
    assert result["log_spam"] > result["log_ham"]
    assert len(result["top_words"]) == 3

    result2 = clf.predict("See you at the meeting tomorrow thanks")
    assert result2["label"] == "ham", f"Expected ham, got {result2['label']}"
    assert result2["log_ham"] > result2["log_spam"]

    print("All tests passed.")

if __name__ == "__main__":
    test_classifier()
```

</details>

## What You Actually Learned

- **Probability Axioms:** You built a classifier on the foundation that probabilities sum to 1 and must be non-negative. The priors $P(\text{spam})$ and $P(\text{ham})$ came straight from the empirical frequency of each class in the training set.
- **Conditional Probability:** You estimated $P(\text{word} \mid \text{class})$ by counting how often each word appeared within each class. This is the core conditional relationship that lets the model learn class-specific vocabulary.
- **Bayes Theorem:** You flipped the conditional direction: instead of knowing the class and guessing the words, you observed the words and inferred the class. The posterior $P(\text{class} \mid \text{email})$ is proportional to the prior times the product of likelihoods — exactly Bayes' theorem in action.
- **Laplace Smoothing:** You handled unseen words by adding a pseudocount, which prevents zero probabilities and keeps the model robust on small datasets.
- **Log-space arithmetic:** You learned why multiplying many tiny probabilities underflows in floating point, and why summing log-probabilities is the standard engineering fix in every production Naive Bayes implementation.

## Resources

- [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html) — Official docs on the family of NB classifiers (reference only; you built it yourself).
- [Wikipedia: Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) — Clean mathematical derivation and the independence assumption.
- [3Blue1Brown: Bayes' Theorem](https://www.3blue1brown.com/lessons/bayes-theorem) — Visual intuition for why Bayes' theorem flips conditionals.
- [NLTK Book: Learning to Classify Text](https://www.nltk.org/book/ch06.html) — Classic walkthrough of text classification with Naive Bayes (conceptual reference).
- [Google Developers: Machine Learning Recipes #7](https://developers.google.com/machine-learning/guides/text-classification) — Practical text-classification guidance from Google's ML education team.
