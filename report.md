# Machine Learning and Neural Network Comparison

## 1. Introduction

This project uses machine learning to predict whether a passenger survived the Titanic disaster. The Titanic dataset was chosen because it is a well-known, beginner-friendly dataset that provides a clear binary prediction task — survived or did not survive — using real, structured passenger data such as age, gender, ticket class, and fare. The goal was to train and compare several models to find which one predicts survival most accurately.

---

## 2. Dataset Description

The dataset contains **891 passenger records** and 12 original features. The target variable is **Survived** (0 = Did Not Survive, 1 = Survived).

Before training, the following preprocessing steps were applied to clean and prepare the data:

- Missing **Age** values were filled with the median age, since it is a more reliable estimate than the average when outliers are present.
- Missing **Embarked** values were filled with the most common category, as only two entries were missing.
- **Cabin** was removed entirely because more than 75% of its values were missing, making it unreliable for predictions.
- **PassengerId**, **Name**, and **Ticket** were removed because they are unique identifiers that carry no predictive value.
- Categorical columns such as **Sex** and **Embarked** were converted to numbers using one-hot encoding so that the models could process them.

The dataset had a mild class imbalance: 61.6% of passengers did not survive and 38.4% survived. Because of this imbalance, the **F1-score** was used as the primary evaluation metric, as it accounts for both false positives and false negatives more fairly than accuracy alone. The data was split into **80% training** and **20% testing**.

---

## 3. Models Used

Three machine learning models were trained and compared:

- **Logistic Regression** — Accuracy: 0.804 | F1-Score: 0.724. A simple model that estimates the probability of survival based on a weighted combination of the input features.
- **Random Forest** — Accuracy: 0.816 | F1-Score: 0.752 *(best)*. An ensemble model that builds many decision trees and combines their outputs, which makes it more robust and less sensitive to noise in the data.
- **K-Nearest Neighbors (KNN)** — Accuracy: 0.816 | F1-Score: 0.744. A model that predicts survival by looking at the closest passengers in the dataset and using their outcomes.

**Random Forest** achieved the highest F1-score and was selected as the best traditional machine learning model.

---

## 4. Neural Network

A feed-forward neural network was built using Keras with the following baseline architecture: two hidden layers with 64 and 32 neurons respectively, both using ReLU activation to introduce non-linearity, a Dropout layer after each hidden layer to reduce overfitting, and a final output layer with a single neuron using Sigmoid activation to produce a probability between 0 and 1.

The baseline model was trained for **10 epochs** using the Adam optimiser and binary cross-entropy loss, achieving an F1-score of **0.750**.

An improved version was then tested with two targeted changes:

**Dropout was increased from 0.3 to 0.5.** A higher dropout rate means that more neurons are randomly switched off during each training step, which prevents the network from relying too heavily on any one path and forces it to learn more general patterns. This helps reduce overfitting on a small dataset.

**Early Stopping was added with a maximum of 50 epochs.** Rather than training for a fixed number of steps, Early Stopping monitors performance on the validation set and halts training automatically when it stops improving. This prevents the model from memorising the training data past the point of usefulness. The model stopped well before the 50-epoch limit, which confirms that longer training would have caused overfitting.

The improved model achieved: **Accuracy: 0.793 | Precision: 0.786 | Recall: 0.638 | F1-Score: 0.704** — slightly *worse* than the baseline rather than better. The heavier dropout (0.5) appears to have removed too much capacity for such a small dataset, so the extra regularisation reduced performance instead of improving it.

---

## 5. Results and Comparison

| Model                   | Accuracy | Precision | Recall | F1-Score |
| ----------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression     | 0.804    | 0.793     | 0.667  | 0.724    |
| Random Forest           | 0.816    | 0.781     | 0.725  | 0.752    |
| KNN                     | 0.816    | 0.800     | 0.696  | 0.744    |
| Baseline Neural Network | 0.821    | 0.814     | 0.696  | 0.750    |
| Improved Neural Network | 0.793    | 0.786     | 0.638  | 0.704    |

**Random Forest** achieved the highest F1-score of **0.752**, making it the strongest overall model — though the baseline neural network came extremely close (0.750) and had the highest accuracy of any model (0.821). The improved neural network performed *worse* than the baseline (F1 dropped to 0.704), so the attempted tuning did not pay off on this dataset.

This result is expected. Neural networks are most powerful when trained on large amounts of data, because they need many examples to detect complex, non-obvious patterns. With only 891 records, there is not enough data for the network to develop a significant advantage. Random Forest, on the other hand, performs well on small, structured datasets because it makes decisions based on straightforward rules derived directly from the input features — and requires far less data and tuning to do so effectively.

---

## 6. What I Learned

This project provided several important lessons:

- **Preprocessing has a large impact.** Cleaning the data and handling missing values carefully was just as important as choosing the right model.
- **F1-score is the right metric for imbalanced data.** Accuracy alone would have been misleading here, since a model that always predicted "did not survive" would still achieve 61.6% accuracy.
- **More complex does not always mean better.** The neural network required significantly more setup, tuning, and careful training than Random Forest, yet it still produced weaker results on this dataset.
- **Dropout and Early Stopping are essential tools.** Understanding not just how to use them, but why they work, helped make more informed decisions about how to improve the model.
- **Dataset size and structure matter when choosing a model.** For small, well-structured datasets, tree-based models are often a stronger and more practical choice than deep learning.

Overall, **Random Forest** provided the best balance across all metrics and was the most effective model for this project.
