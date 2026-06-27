<!-- Slide number: 1 -->
# The Problem of Imbalanced Data in Machine Learning. Algorithm-Level Approach
Marina M. Lukashevich

### Notes:

<!-- Slide number: 2 -->
# Imbalanced Problem in Machine Learning

<!-- Slide number: 3 -->
# ML Pipeline

![Top 9 ML Pipeline Platforms for AI Lifecycle Management in 2024](Picture2.jpg)

### Notes:

<!-- Slide number: 4 -->
# Imbalanced Datasets in Machine Learning
A dataset is considered imbalanced if "the distribution of instances across known classes is uneven" .
While there is no universal consensus in the literature on criteria for classifying imbalance severity, Google’s Machine Learning Course proposes the following thresholds:
Mild imbalance: Minority class = 20–40% of total instances.
Moderate imbalance: Minority class = 1–20%.
Severe imbalance: Minority class < 1%.

Majority = Negative
Minority = Positive

![](Рисунок1.jpg)
The imbalance ratio (IR) is commonly defined as the ratio between the number of instances in the minority class and the majority class.

<!-- Slide number: 5 -->
# Imbalanced Problem

![](Рисунок6.jpg)
Class imbalance: Ratio of classes is significantly different.
Consequence: Undesirable predictive behavior for smaller class.

Often, the minority class is the more important class.

<!-- Slide number: 6 -->
# Machine Learning Tasks

![Machine learning tasks | Download Scientific Diagram](Picture4.jpg)
Binary Classification: A classification predictive modeling problem where all examples belong to one of two classes (2 classes).
Multiclass Classification: A classification predictive modeling problem where all examples belong to one of three or more classes (>2 classes).

![](Рисунок6.jpg)
A supervised learning task where objects are assigned to predefined classes based on their features.

2 classes
>2 classes

<!-- Slide number: 7 -->
# Approaches for class imbalanced learning
Data-Level Approaches for Handling Class Imbalance
Data-level approaches to the problem of class imbalance involve techniques and methods that directly modify the training data.

OversamplingIncreases the number of minority class instances by duplicating existing samples or generating synthetic data. Pros: Preserves original data. Cons: May lead to overfitting.
UndersamplingReduces the majority class by randomly removing samples or using clustering techniques. Pros: Improves training speed. Cons: Loss of potentially useful information.
Hybrid SamplingCombines oversampling (for minority class) and undersampling (for majority class), balances data while mitigating drawbacks of standalone methods.

![](Рисунок3.jpg)
Undersampling involves reducing instances from the majority class.
Oversampling involves increasing instances from the minority class.

<!-- Slide number: 8 -->
# ML Pipeline

![Top 9 ML Pipeline Platforms for AI Lifecycle Management in 2024](Picture2.jpg)

### Notes:

<!-- Slide number: 9 -->
# Metrics

<!-- Slide number: 10 -->
# Performance Evaluation in Machine Learning
The Ideal vs. The Reality

The Accuracy Trap

Ideal Goal: Maximize correct predictions for all classes (100% accuracy).
Imbalanced Data Reality:
High accuracy on majority class(es).
Poor performance on minority class(es).
Why? Models favor majority predictions to artificially inflate accuracy.

Example:
Disease prevalence = 0.5% → Always predicting "no disease" yields 99.5% accuracy.
But: Fails to detect any actual cases (0% recall for minority class).
Key Insight:
Accuracy is misleading with class imbalance.
Optimizing for it encourages harmful bias toward majority classes.

<!-- Slide number: 11 -->
# Confusion matrix for a binary class problem
Majority Class: Negative outcome, class 0.
Minority Class: Positive outcome, class 1.

![](Рисунок7.jpg)
TP: True positives. These are the instances of class 1, that the classifier correctly predicts.
TN: True negatives. These are the instances of class 0, that the classifier correctly predicts.
FP: False positives. These are the instances of class 0, that the classifier incorrectly predicts.
FN: False negatives. These are the instances of class 1, that the classifier incorrectly predicts.

<!-- Slide number: 12 -->
# Threshold Metrics

![](Рисунок3.jpg)

![](Рисунок4.jpg)

<!-- Slide number: 13 -->
# Threshold Metrics: Accuracy

![](Объект3.jpg)

![](Рисунок4.jpg)
Three confusion matrices with the same accuracy

<!-- Slide number: 14 -->
# Threshold Metrics: Sensitivity-Specificity Metrics

![](Рисунок3.jpg)
Sensitivity refers to the true positive rate and summarizes how well the positive class was predicted.

![](Рисунок4.jpg)
Specificity is the complement to sensitivity, or the true negative rate, and summarizes how well the negative class was predicted.

![](Рисунок7.jpg)
Sensitivity and Specificity can be combined into a single score that balances both concerns, called the G-mean.

<!-- Slide number: 15 -->
# Threshold Metrics: Balanced Accuracy

![](Объект3.jpg)

![Classification matrix for a binary classification problem.](Picture2.jpg)
Accuracy = (TP + TN) / (TP+FN+FP+TN) = 20+5000 / (20+70+30+5000)
Accuracy = ~98.05%

Sensitivity = TP / (TP + FN) = 20 / (20+70) = 22.2%
Specificity = TN / (TN + FP) = 5000 / (5000 +30) = ~99.4%

Balanced accuracy = (sensitivity + specificity) / 2 = (22.2 + 99.4) / 2 = 60.80%

<!-- Slide number: 16 -->
# Threshold Metrics: Balanced Accuracy
Accuracy = TP + TN / (TP+FP+FN+TN)

TP = 10 + 545 + 11 + 3 = 569
FP = 175 + 104 + 39 + 50 = 368
TN = 695 + 248 + 626 + 874 = 2443
FN = 57 + 40 + 261 + 10 = 368

Accuracy = 569 + 2443 / (569 + 368 + 368 + 2443)
Accuracy = 0.803

Balanced accuracy = (RecallP + RecallQ + RecallR + RecallS) / 4

RecallP = 10 / (10+57) = 0.054
RecallQ = 545 / (545 + 40) = 0.932
RecallR = 11 / (11 + 261) = 0.040
RecallS = 3 / (3 + 10) = 0.231

Balanced accuracy = (0.054 + 0.932 + 0.040 + 0.231) / 4 = 1,257 / 4 = 0.3143

![A sample confusion matrix for a multi-class imbalanced classification problem. There are four classes present: P, Q, R, and S. The matrix also shows row- and column-wise sum of metrics.](Picture2.jpg)

<!-- Slide number: 17 -->
# Threshold Metrics: Precision-Recall Metrics

![](Рисунок3.jpg)
Precision summarizes the fraction of examples assigned the positive class that belong to the positive class.

![](Рисунок4.jpg)
Recall summarizes how well the positive class was predicted and is the same calculation as sensitivity.
Precision and recall can be combined into a single score that seeks to balance both concerns, called the F-score or the F-measure.

![](Рисунок5.jpg)

![](Рисунок9.jpg)
The Fbeta-measure measure is an abstraction of the F-measure where the balance of precision and recall in the calculation of the harmonic mean is controlled by a coeficient called beta.

<!-- Slide number: 18 -->
# Threshold Metrics: advanced

![](Объект3.jpg)
This measure takes into account the relative balance of the classifier’s performance on both the positive and the negative classes.

![](Рисунок6.jpg)
Another version of the G-mean was, therefore, also suggested, which focuses solely on the positive class.

![](Рисунок8.jpg)
The macro-averaged accuracy (MAA) is calculated as the
arithmetic average of the partial accuracies of each class. Its formula is given as follows for the binary case.

### Notes:
Because the two classes are given equal importance in this formulation, the Gmean, while more sensitive to class imbalances than accuracy, remains close, in some sense, to the multi-class focus category of metrics.

<!-- Slide number: 19 -->
# Example

![](Рисунок10.jpg)

![](Рисунок9.jpg)
False Positive (type I error)
null hypothesis was wrongly rejected
spam was classified as no spam
False Negative (type II error)
null hypothesis was wrongly accepted
no spam was classified as spam

<!-- Slide number: 20 -->
# Example

![](Объект5.jpg)

![](Рисунок6.jpg)
In this case we have:
true negatives = 616 (no spam labeled as no spam)
false negatives = 59 (spam labeled as no spam)
true positives = 71 (spam labeled as spam)
false positives = 89 (no spam labeled as spam)

<!-- Slide number: 21 -->
# Example
Accuracy Score for evaluation: to divide number of correct predictions by total number of predictions and get the percentage of samples were predicted correctly

![](Рисунок3.jpg)
Example: TN = 95 with spam and FN=5 with no spam as spam.

Looks like the classifier has great accuracy – 95%, but it do absolutely nothing.

We need to move from calculating the common metric for all classes to separate performance metrics for each class.

![](Рисунок4.jpg)

<!-- Slide number: 22 -->
# Example

![](Объект3.jpg)

![](Рисунок5.jpg)

![](Рисунок4.jpg)
This clearly illustrates the essence of this model: it’s pretty good at finding spam, but terrible at finding no spam.

<!-- Slide number: 23 -->
# Example

![](Объект3.jpg)

![](Рисунок4.jpg)

<!-- Slide number: 24 -->
# Example
FP = 5, TP = 5, TN = 90, FN = 0

![](Рисунок3.jpg)
the classifier was wrong as many times as it was right in rare case.

![](Рисунок5.jpg)
Precision = 0.5 means that letter with no spam, in half of the cases we’ll find spam there.

![](Рисунок6.jpg)

<!-- Slide number: 25 -->
# Example
Case 1: TP = FP = 0, TN = 95, FN = 5

Case 2: TP = 0, FP = 10, FN = 5, TN = 85

Case 3: TP = 5, FP = 5, FN = 0, TN = 90

![](Рисунок3.jpg)

![](Рисунок5.jpg)

![](Рисунок6.jpg)

### Notes:

<!-- Slide number: 26 -->
# Example

![](Объект3.jpg)

<!-- Slide number: 27 -->
# Example. Some conclusions.
Accuracy
the fraction of correct predictions
depends on the balance between classes, is not applicable to imbalanced datasets
for balanced datasets, is equal to Balanced Accuracy
Balanced Accuracy
the average of Sensitivity and Specificity
immune to class imbalance, can be applied to imbalanced datasets
for balanced datasets, is equal to Accuracy
Precision (Positive Predictive Value)
the fraction of true positive predictions from all positive predictions
shows if the classifier is able to differ one class from all others
immune to class imbalance, can be applied to imbalanced datasets
should be balanced with Recall
Recall (True Positive Rate, Sensitivity, Hit Rate)
the fraction of true positive predictions from all positive samples in dataset
shows if the classifier is able to detect a giving class at all
immune to class imbalance, can be applied to imbalanced datasets
should be balanced with Precision
F1
harmonic mean of Precision and Recall
doesn’t take into account True Negatives
immune to class imbalance, can be applied to imbalanced datasets

<!-- Slide number: 28 -->
# Ranking Metrics

![](Рисунок4.jpg)
The most commonly used ranking metric is the ROC Curve or ROC Analysis (Receiver Operating Characteristic).
It summarizes a field of study for analyzing binary classifiers based on their ability to discriminate classes.

A ROC curve is a diagnostic plot for summarizing the behavior of a model by calculating the false positive rate and true positive rate for a set of predictions by the model under difierent thresholds.

![](Рисунок5.jpg)

![](Рисунок6.jpg)
The ROC Curve is a helpful diagnostic for one model. The area under the ROC curve can be calculated and provides a single score to summarize the plot that can be used to compare models. A no skill classier will have a score of 0.5, whereas a perfect classier will have a score of 1.0.

### Notes:

<!-- Slide number: 29 -->
# Ranking Metrics

![](Рисунок4.jpg)
Precision-Recall Curve is a helpful diagnostic tool for evaluating
a single classier but challenging for comparing classifiers. Like the ROC AUC, we can calculate the area under the curve as a score and use that score to compare classifiers.

<!-- Slide number: 30 -->
# Probabilistic Metrics
The most common metric for evaluating predicted probabilities is log loss for binary classification (or the negative log likelihood), or known more generally as cross-entropy. For a binary classification dataset where the expected values are y and the predicted values are yhat, this can be calculated as follows:

![](Рисунок4.jpg)

![](Рисунок6.jpg)
The Brier score is that it is focused on the positive class, which for imbalanced classification is the minority class.
A perfect classier has a Brier score of 0.0.

![](Рисунок7.jpg)
Using the reference score, a Brier Skill Score, or BSS, can be calculated where 0.0 represents no skill, worse than no skill results are negative, and the perfect skill is represented by a value of 1.0.

<!-- Slide number: 31 -->
# How to Choose a Metric for Imbalanced Classification

![](Объект3.jpg)

<!-- Slide number: 32 -->
# Recommendations
No Universal Metric
Different evaluation measures capture distinct aspects of model performance (e.g., recall vs. precision).
There is no single "best" metric — choice depends on:
Application requirements (e.g., fraud detection prioritizes recall).
Cost of errors (FP vs. FN trade-offs).
Accuracy is Misleading
Never use accuracy for imbalanced data (e.g., 99% accuracy with 1% minority class is trivial).
Preferred alternatives:
F1-score (harmonic mean of precision/recall).
Balanced Accuracy (BAC) or MCC (Matthews Correlation Coefficient).
Metric Comparison Pitfalls
Avoid direct numerical comparisons between metrics with different scales/ranges (e.g., MCC ∈ [-1,1] vs. BAC ∈ [0,1]).
Solution: Normalize metrics or use domain-specific cost matrices.

<!-- Slide number: 33 -->
# Cost-Sensitive Learning. Class Weighting
Assigns higher misclassification penalties to minority classes during training

<!-- Slide number: 34 -->
# Cost-Sensitive Learning
Cost-sensitive learning is a subfield of machine learning that addresses classification problems where the misclassification costs are not equal.
Cost-sensitive problems occur in many disciplines such as medicine (e.g., disease detection), engineering (e.g., machine failure detection), transport (e.g., traffic-jam detection), finance (e.g., fraud detection), and so forth.
They are often related to the class-imbalance problem since in most of these problems, the goal is to detect events that are rare. The training datasets therefore typically contain fewer examples of the event of interest.

<!-- Slide number: 35 -->
# Cost Matrix

![](Объект3.jpg)
Denoting by  the cost matrix, its entries  quantify the cost of predicting class  when the true class is. For a binary classification problem, the cost matrix is a  matrix, 2×2.

Correct classifications have a cost of zero, that is,

![](Рисунок5.jpg)

![](Рисунок6.jpg)
the majority and minority class, respectively.

![](Рисунок9.jpg)

![](Рисунок11.jpg)
Cost matrix for imbalanced data. The cost of a false negative is 1, and the cost of a false positive is the imbalance ratio (IR).

<!-- Slide number: 36 -->
# Cost Matrix

![](Объект4.jpg)
In datasets where some classes have significantly fewer samples than others, a model might become biased towards the majority class, leading to poor performance on the minority class. class_weight  helps mitigate this by giving more importance to samples from underrepresented classes during model training.

![](Рисунок7.jpg)

<!-- Slide number: 37 -->
# Cost-Sensitive Learning
By assigning higher weights to minority classes, the model's loss function is adjusted to penalize errors on these classes more severely, compelling the model to learn features that better distinguish them. This can lead to improved recall for the minority class, often at the cost of some precision.

The scikit-learn Python machine learning library provides examples of these cost-sensitive extensions via the class weight argument on the following classifiers:
SVC,
DecisionTreeClassifier,
RandomForest,
GradinerBoosting Classifier,
etc.

<!-- Slide number: 38 -->
# Example 1. Synthetic Dataset

![](Объект3.jpg)

<!-- Slide number: 39 -->

![](Объект3.jpg)
# Example 1. Model Development

<!-- Slide number: 40 -->
# Example 1. No class_weight vs ‘balanced’

![](Объект3.jpg)

<!-- Slide number: 41 -->
# Example 1. No class_weight vs ‘balanced’

![](Рисунок9.jpg)

![](Рисунок6.jpg)

![](Рисунок7.jpg)

![](Рисунок8.jpg)

### Notes:

<!-- Slide number: 42 -->
# Example 1. No class_weight vs ‘balanced’

![](Рисунок5.jpg)

![](Рисунок6.jpg)

![](Рисунок4.jpg)

![](Рисунок3.jpg)

<!-- Slide number: 43 -->
# Example 1. No class_weight vs ‘balanced’ vs dict

![](Объект3.jpg)

<!-- Slide number: 44 -->
# Example 1. No class_weight vs ‘balanced’ vs dict

![](Рисунок4.jpg)

![](Рисунок6.jpg)

![](Рисунок8.jpg)

![](Рисунок5.jpg)

![](Рисунок7.jpg)

![](Рисунок9.jpg)

<!-- Slide number: 45 -->
# Example 1. No class_weight vs ‘balanced’ vs dict

![](Рисунок3.jpg)

![](Рисунок4.jpg)

![](Рисунок5.jpg)

![](Рисунок8.jpg)

![](Рисунок7.jpg)

![](Рисунок9.jpg)

<!-- Slide number: 46 -->
# Example 2. Decision Tree

![](Объект3.jpg)

<!-- Slide number: 47 -->

![](Объект3.jpg)
# Example 2. Decision Tree

<!-- Slide number: 48 -->
# Example 2. Decision Tree

![](Объект3.jpg)

<!-- Slide number: 49 -->
# Example 2. Decision Tree

![](Рисунок9.jpg)

![](Рисунок6.jpg)

![](Рисунок5.jpg)

![](Рисунок7.jpg)

![](Объект3.jpg)

![](Рисунок10.jpg)

![](Рисунок8.jpg)

![](Рисунок4.jpg)

![](Рисунок11.jpg)

<!-- Slide number: 50 -->
# Example 1 and 2. Logistic Regression and Decision Tree. Saving Results

![](Рисунок4.jpg)

![](Рисунок5.jpg)

<!-- Slide number: 51 -->
# Example 3. Keras model

![](Объект3.jpg)

<!-- Slide number: 52 -->
# Example 3. Keras model

![](Объект3.jpg)

<!-- Slide number: 53 -->
# Example 3. Keras model

![](Объект3.jpg)

<!-- Slide number: 54 -->
# Example 3. Keras model

![](Рисунок5.jpg)

![](Объект3.jpg)

![](Рисунок4.jpg)

![](Рисунок6.jpg)

<!-- Slide number: 55 -->
# Example 3. Keras model

![](Рисунок5.jpg)

![](Объект3.jpg)

![](Рисунок4.jpg)

![](Рисунок6.jpg)

<!-- Slide number: 56 -->
# Example 3. Keras model

![](Рисунок5.jpg)

![](Объект3.jpg)

![](Рисунок4.jpg)

![](Рисунок6.jpg)

<!-- Slide number: 57 -->
# Ensemble Methods
Algorithms like Balanced Random Forest or EasyEnsemble focus on underrepresented samples

<!-- Slide number: 58 -->
# Ensemble Methods
Ensemble methods consist in training multiple prediction models for the same prediction task, and in combining their outputs to make the final prediction.
Ensembles of models very often allow to provide better prediction performances than single models, since combining the predictions from multiple models usually allows to reduce the overfitting phenomenon. The prediction models making up an ensemble are referred to as baseline learners.
Both the way with which the baseline learners are constructed and how their predictions are combined are key factors in the design of an  ensemble.

<!-- Slide number: 59 -->
# Ensemble Methods
Ensemble methods can be broadly divided into parallel-based and iterative-based ensembles.

In parallel-based ensembles, each baseline learner is trained in parallel, using either a subset of the training data, a subset of the training features, or a combination of both. The two most popular techniques for parallel-based ensembles are bagging and random forest.

In iterative-based ensembles, also referred to as boosting, the baseline classifiers are trained in sequence, with each learner in the sequence aiming at minimizing the prediction errors of the previous learner. The currently most widely-used implementations for boosting are XGBoost, CatBoost and LightGBM.

<!-- Slide number: 60 -->
# Ensemble Methods

![](Объект3.jpg)
The image illustrates a machine learning workflow designed to handle imbalanced datasets through ensemble methods. It starts by separating the majority and minority classes, applying different sampling techniques to each: under-sampling or bootstrapping for the majority class and over-sampling or bootstrapping for the minority class. The sampled data is then divided into multiple subsets (subset 1 to subset n), each undergoing feature selection before forming individual training sets.

<!-- Slide number: 61 -->
# Ensemble Methods

![](Объект3.jpg)
This approach ensures diversity in the data used for training multiple classifiers.
The bottom part of the image depicts a multiple classifier system where each classifier (Classifier 1 to Classifier n) is trained on its respective training set. When new data (X) is introduced, all classifiers contribute to the prediction through a majority voting mechanism. This ensemble technique combines the strengths of individual models to improve overall accuracy and robustness, particularly in scenarios with class imbalance, by leveraging diverse perspectives derived from different data subsets and feature selections.

<!-- Slide number: 62 -->
# Ensemble Methods
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of their predictions for classification tasks. Each tree is trained on a random subset of the data and features, which introduces diversity and reduces overfitting. While highly effective for balanced datasets, Random Forest tends to favor the majority class in imbalanced scenarios, as its objective is to maximize overall accuracy. This can lead to poor performance on minority classes, as the algorithm may overlook their patterns due to their underrepresentation in the training data.
Balanced Random Forest addresses this limitation by explicitly incorporating class imbalance into the training process. It combines the principles of Random Forest with techniques like under-sampling the majority class or over-sampling the minority class within each bootstrap sample. This ensures that every tree in the ensemble receives a balanced subset of data, improving sensitivity to minority class patterns. As a result, Balanced Random Forest achieves better recall for underrepresented classes while maintaining robust performance, making it particularly useful for applications like fraud detection or medical diagnosis where minority class identification is critical.

<!-- Slide number: 63 -->
# Example 4. RandomForest vs BalancedRandomForest

![](Объект3.jpg)

<!-- Slide number: 64 -->
Example 4. RandomForest vs BalancedRandomForest

![](Объект3.jpg)

<!-- Slide number: 65 -->
# Example 4. RandomForest vs BalancedRandomForest

![](Объект3.jpg)

<!-- Slide number: 66 -->
# Example 4. RandomForest vs BalancedRandomForest

![](Объект4.jpg)

![](Рисунок5.jpg)

<!-- Slide number: 67 -->
# Example 4. RandomForest vs BalancedRandomForest

![](Объект4.jpg)

![](Рисунок5.jpg)

<!-- Slide number: 68 -->
# Anomaly Detection
Treats minority classes as outliers using one-class classifiers (e.g., Isolation Forest)

<!-- Slide number: 69 -->
# Class Imbalance as Anomaly Detection
"Rare Classes = Anomalies in Disguise"
Parallel Objectives:
Both aim to identify rare events (fraud, faults, diseases)
Minority class = "anomalies" in normal data distribution
Shared Challenges:
High false negatives (missing anomalies)
Evaluation via precision/recall (not accuracy)
Need specialized sampling/metrics
Key Insight:"Anomaly detection is extreme class imbalance (99.9% vs 0.1%)"

<!-- Slide number: 70 -->
# One-Class SVM

![](Рисунок3.jpg)

<!-- Slide number: 71 -->
# One-Class SVM

![](Рисунок4.jpg)

![](Рисунок5.jpg)

![](Рисунок6.jpg)

<!-- Slide number: 72 -->
# Advantages, Limitations, Recommendations

<!-- Slide number: 73 -->
# Class Weighting
What: Adjusts loss/decision thresholds to penalize misclassifications of the minority class more heavily.
Methods:
class_weight in scikit-learn
custom weights
Pros:
Simple to implement (built into many algorithms like SVM, Logistic Regression).
No data modification needed (preserves original distribution).
Cons:
May overfit to noise in the minority class.
Less effective for extreme imbalance (e.g., 99:1).
When to Use:
Moderate imbalance (e.g., 80:20 to 95:5).
Algorithms that natively support weighting (e.g., trees, linear models).

<!-- Slide number: 74 -->
# Ensemble Methods
What: Combines multiple models to improve minority class detection.Methods:
Balanced Random Forest: Under-samples majority class per tree.
RUSBoost: Combines random under-sampling with boosting.
EasyEnsemble: Multiple balanced subsets + ensemble voting.
Pros:
Handles extreme imbalance effectively.
Built-in diversity reduces overfitting.
Cons:
Computationally expensive.
Hyperparameter tuning complexity.
When to Use:
Severe imbalance (e.g., 99:1).
When interpretability is less critical than performance.

<!-- Slide number: 75 -->
# Anomaly Detection
What: Treats the minority class as "anomalies" in a majority-class world.Methods:
One-Class SVM: Learns a boundary around normal data.
Isolation Forest: Isolates anomalies using random splits.
Autoencoders: Reconstructs normal data; flags outliers.
Pros:
No need for minority class samples during training.
Works for extreme imbalance (e.g., 99.9:0.1).
Cons:
Hard to tune
May miss semantic patterns in minority class.
When to Use:
Very rare anomalies (e.g., fraud, defects).
When minority class is poorly defined or missing during training.

<!-- Slide number: 76 -->
# Comparison Summary

![](Объект6.jpg)

<!-- Slide number: 77 -->
# Recommendations
Start simple: Try class_weight first for moderate imbalance.
Use ensembles for severe imbalance with sufficient compute resources.
Switch to anomaly detection if:
Minority class is extremely rare (<1%).
You lack labeled anomalies for training.
Always validate with metrics like:
Precision-Recall AUC (better for imbalance than ROC AUC).
F1-score (harmonic mean of precision/recall).

<!-- Slide number: 78 -->
# Some Research Results

<!-- Slide number: 79 -->
# Datasets
| # | Dataset | IR | Totalsamples | Classdistribution |
| --- | --- | --- | --- | --- |
| 1 | ecoli | 0.1163 | 336 | 0:301, 1:35 |
| 2 | abalone | 0.1033 | 4177 | 0:3786, 1:391 |
| 3 | car\_eval\_34 | 0.0841 | 1728 | 0:1594, 1:134 |
| 4 | arrhythmia | 0.0585 | 452 | 0:427, 1:25 |
| 5 | oil | 0.0458 | 937 | 0:896, 1:41 |
| 6 | car\_eval\_4 | 0.0391 | 1728 | 0:1663, 1:65 |
| 7 | ozone\_level | 0.0296 | 2536 | 0:2463, 1:73 |
| 8 | abalone\_19 | 0.0077 | 4177 | 0:4145, 1:32 |

<!-- Slide number: 80 -->
# Approaches for class weighting

![](Рисунок3.jpg)
Inverse Frequency

Logarithmic Weighting

Effective Number of Samples (ENN)

Adaptive weighted

![](Рисунок4.jpg)

![](Рисунок5.jpg)

![](Рисунок6.jpg)

<!-- Slide number: 81 -->
# Experimental results for class_weight=’balanced’ and Inverse Frequency

![](Объект3.jpg)

<!-- Slide number: 82 -->
# Experimental results for Logarithmic weighting and Effective number of samples (ENN)

![](Объект3.jpg)

<!-- Slide number: 83 -->
# Experimental results for F1 adaptive weighted

![](Объект3.jpg)

<!-- Slide number: 84 -->
# Discussion
The experimental results demonstrate the efficacy of class weighting methods. The analysis reveals that while the commonly used Inverse Frequency approach improves model performance, it fails to achieve maximum accuracy values compared to Logarithmic Weighting. This finding suggests that practitioners should evaluate multiple weighting approaches and select the optimal one based on its synergistic effects with the chosen classification algorithm and other hyperparameters. Furthermore, comprehensive exploratory data analysis must precede model development.
This critical phase enables:
identification of class imbalance,
quantification of imbalance severity,
determination of necessary mitigation strategies, including class weighting implementation.

<!-- Slide number: 85 -->
# Discussion
These findings yield specific recommendations for machine learning pipeline construction.
(1) Conduct comprehensive exploratory data analysis to assess dataset characteristics.
(2) Quantify class imbalance ratios when detected during initial analysis.
(3) Include class weighting methods in hyperparameter optimization procedures.
(4) For severe imbalance cases, evaluate alternative weighting approaches beyond standard library implementations, particularly Logarithmic Weighting.
(5) Perform joint optimization of classification algorithms and weighting methods to maximize overall model performance.

<!-- Slide number: 86 -->
# Thank you for your attention!