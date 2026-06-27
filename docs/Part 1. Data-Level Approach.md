<!-- Slide number: 1 -->
# The Problem of Imbalanced Data in Machine Learning. Data-Level Approach
Marina M. Lukashevich

### Notes:

<!-- Slide number: 2 -->
# Machine Learning and Classification Task

<!-- Slide number: 3 -->
# AI vs ML vs DL

![Differences Between AI vs. Machine Learning vs. Deep Learning | Simplilearn](Picture6.jpg)

<!-- Slide number: 4 -->
# Machine Learning vs. Traditional Programming

![](Picture8.jpg)
Traditional programming: you express rules in a programming language. They act on data and your program provides answers.

Machine learning: you provide the answers (typically called labels) along with the data, and the machine infers the rules that determine the relationship between the answers and data.

![](Picture2.jpg)

<!-- Slide number: 5 -->
# Data and Knowledge

![иконки таблица, table,](Picture14.jpg)
Table
Text
Signal
Sound
Image
Video

![Картинки по запросу knowledge png](Picture4.jpg)

![data](Picture10.jpg)

![иконка текст, text, файл, документ, document, редактирование, редактировать, edit,](Picture10.jpg)

![иконка graph, статистика,](Picture12.jpg)

![иконка звук, динамик, громкость, sound,](Picture2.jpg)

![иконка image, изображение,](Picture6.jpg)

![иконки video, видео,](Picture8.jpg)

<!-- Slide number: 6 -->
# ML Pipeline

![Top 9 ML Pipeline Platforms for AI Lifecycle Management in 2024](Picture2.jpg)

### Notes:

<!-- Slide number: 7 -->
# Algorithm vs Model
Algorithm A method, function, or series of instructions used to generate a machine learning model. Examples include linear regression, decision trees, support vector machines, and neural networks.

Model A data structure that stores a representation of a dataset (weights and biases). Models are created/learned when you train an algorithm on a dataset.

<!-- Slide number: 8 -->
# Machine Learning Tasks

![Machine learning tasks | Download Scientific Diagram](Picture4.jpg)
Binary Classification: A classification predictive modeling problem where all examples belong to one of two classes (2 classes).
Multiclass Classification: A classification predictive modeling problem where all examples belong to one of three or more classes (>2 classes).

![](Рисунок6.jpg)
A supervised learning task where objects are assigned to predefined classes based on their features.

2 classes
>2 classes

<!-- Slide number: 9 -->
# Imbalanced Data

### Notes:
Привет

<!-- Slide number: 10 -->
# EDA

![](Объект4.jpg)

![](Рисунок5.jpg)

![](Рисунок6.jpg)

### Notes:

<!-- Slide number: 11 -->
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

<!-- Slide number: 12 -->
# Why Imbalance Matters

![Gender balance in the Nordic forest sector](Picture2.jpg)
Medical diagnosis (1% cancer cases)
Fraud detection (<0.1% fraud)
Claim Prediction (~1-5% claims)
Default Prediction (~2-10% defaults)
Churn Prediction (~5-20% churn)
Spam Detection (~50-90% spam)
Anomaly Detection (~0.1-5% anomalies)
Outlier Detection (~0.1-5% outliers)
Intrusion Detection (<1% attacks)
Conversion Prediction (~1-10% conversions)

Key Challenge: Models bias toward majority class

Majority Class: The class (or classes) in an imbalanced classification predictive modeling. problem that has more examples.
Minority Class: The class in an imbalanced classification predictive modeling problem that has less examples.

When working with an imbalanced classification problem, the minority class is typically of the most interest. This means that a model's skill in correctly predicting the class label or probability for the minority class is more important than the majority class or classes.

### Notes:
Fraud Detection (<0.1% fraud) – Мошеннические операции крайне редки.
Claim Prediction (~1-5% claims) – Зависит от отрасли (страхование, гарантии и т. д.).
Default Prediction (~2-10% defaults) – Зависит от кредитного риска и экономической ситуации.
Churn Prediction (~5-20% churn) – Зависит от бизнес-модели и конкурентной среды.
Spam Detection (~50-90% spam) – В некоторых почтовых системах спам преобладает.
Anomaly Detection (~0.1-5% anomalies) – Сильно зависит от контекста (финансы, IoT, производство).
Outlier Detection (~0.1-5% outliers) – Аналогично аномалиям, зависит от данных.
Intrusion Detection (<1% attacks) – Большинство сетевого трафика легитимно.
Conversion Prediction (~1-10% conversions) – Зависит от продукта и рекламной кампании.

<!-- Slide number: 13 -->
# Imbalanced Datasets in Machine Learning
Balanced distribution with an almost equal number of examples for each class
An imbalanced dataset with five classes and a varying number of samples

![](Объект17.jpg)

![](Объект13.jpg)

![](Рисунок18.jpg)

<!-- Slide number: 14 -->
# Imbalanced Datasets in Machine Learning

![](Объект8.jpg)
Scatter Plot of a Binary Classification Dataset with Different Class Distributions

<!-- Slide number: 15 -->
# Example

![](Объект3.jpg)

![](Рисунок4.jpg)

![](Рисунок7.jpg)

<!-- Slide number: 16 -->
# Causes of Class Imbalance in Datasets
Domain-Specific Rarity
Certain events/states are inherently rare (e.g., rare diseases, fraudulent transactions).
Data Collection & Labeling Errors
Human bias, equipment flaws, or misconfigured sampling.
Biased Sampling
Limited geographic/time coverage skews class distribution (e.g., data from a single region).
Mitigation Challenges: Re-collecting data or relabeling may resolve biases but is often costly, time-consuming, or impractical, forcing reliance on existing datasets.

<!-- Slide number: 17 -->
# Why Imbalanced Classification Is Hard?

![](Рисунок7.jpg)

![](Рисунок5.jpg)

![](Рисунок11.jpg)
Scatter Plots of an Imbalanced Classification Dataset With Different Numbers of Clusters
Scatter Plots of an Imbalanced Classification Dataset With Different Label Noise
Scatter Plots of an Imbalanced Classification Dataset With Different Dataset Sizes

### Notes:

<!-- Slide number: 18 -->
# Some conclusions and recommendation
Unbalanced datasets often arise in practical tasks in medicine, business, security, engineering and related fields, and pose a serious problem in machine learning, as most standard algorithms perform poorly on classification tasks when one or more classes significantly dominate over the rest.
Among the reasons for class imbalance are subject domain specificity, data collection errors, and sampling bias. It is often more appropriate to work with the existing data set due to financial, resource and time constraints on obtaining a new data set.
Classical classification algorithms such as SVM, naive Bayesian classifier, decision trees, k-nearest neighbors algorithm are sensitive to class imbalance and show insufficient efficiency when dealing with unbalanced data.
To evaluate the quality of models in classification tasks with unbalanced data set, it is important to select appropriate metrics: either insensitive to class imbalance (FPR and FNR) or based on balanced accuracy (balanced accuracy, weighted average of classical metrics for multiclass classification).

<!-- Slide number: 19 -->
# Data-Level Approach

<!-- Slide number: 20 -->
Data-Level Methods (Preprocessing/Sampling): These modify the dataset's class distribution directly.
◦
Oversampling: Increases the number of minority class examples by duplicating existing ones or synthesizing new ones (e.g., SMOTE, Borderline-SMOTE, ADASYN, ROSE, Safe-Level-SMOTE, DBSMOTE).
◦
Undersampling: Reduces the number of majority class examples (e.g., Near-Miss, Condensed Nearest Neighbor Rule (CNN), Tomek Links, Edited Nearest Neighbors Rule (ENN), One-Sided Selection (OSS), Neighborhood Cleaning Rule (NCL)).
◦
Combined Methods: Hybrid approaches that use both oversampling and undersampling techniques (e.g., SMOTETomek, SMOTEENN).
# Approaches for class imbalanced learning
Data-Level Approaches for Handling Class Imbalance
Data-level approaches to the problem of class imbalance involve techniques and methods that directly modify the training data.

OversamplingIncreases the number of minority class instances by duplicating existing samples or generating synthetic data. Pros: Preserves original data. Cons: May lead to overfitting.
UndersamplingReduces the majority class by randomly removing samples or using clustering techniques. Pros: Improves training speed. Cons: Loss of potentially useful information.
Hybrid SamplingCombines oversampling (for minority class) and undersampling (for majority class), balances data while mitigating drawbacks of standalone methods.

![](Рисунок3.jpg)
Undersampling involves reducing instances from the majority class.
Oversampling involves increasing instances from the minority class.

<!-- Slide number: 21 -->
# Oversampling and Undersampling

![](Объект3.jpg)

![](Рисунок4.jpg)

<!-- Slide number: 22 -->
# Oversampling
Oversampling is a technique used to address class imbalance by increasing the number of instances in the minority class, typically through duplication or synthetic generation (e.g., SMOTE - Synthetic Minority Over-sampling Technique). This approach helps prevent model bias toward the majority class by providing more training examples for underrepresented categories. While effective, oversampling can sometimes lead to overfitting, especially if synthetic samples are not representative of real-world variability. It is particularly useful when data collection is expensive or limited, as it maximizes the utility of existing minority-class samples.

![](Рисунок5.jpg)
Random oversampling process

<!-- Slide number: 23 -->
# Balanced Approach:Oversampling Minority Class
SMOTE (Synthetic Minority Over-sampling Technique)
SMOTE is a synthetic oversampling technique that generates new minority-class examples by interpolating between existing instances. For each minority sample, the algorithm identifies its *k* nearest neighbors, randomly selects one, and creates a synthetic point along the line connecting the original instance and its neighbor. This helps balance the dataset without simple duplication, reducing the risk of overfitting while improving classifier performance on minority classes.

Borderline-SMOTE (Boundary-Focused Oversampling)
Borderline-SMOTE is an enhanced version of SMOTE that specifically targets minority instances near the decision boundary. First, it identifies "borderline" minority samples (those surrounded mostly by majority-class neighbors). Then, it applies SMOTE only to these critical instances, generating synthetic data where the classifier struggles the most. This makes oversampling more effective by reinforcing the most challenging regions of the feature space.

ADASYN (Adaptive Synthetic Sampling)
ADASYN is an adaptive oversampling method that automatically adjusts the number of synthetic samples generated for each minority instance based on local density. Harder-to-classify minority examples (those with more majority-class neighbors) receive more synthetic samples, while easier cases get fewer. This adaptively shifts focus toward ambiguous regions, improving classifier.

![Exploring Oversampling Techniques for Imbalanced Datasets ...](Picture2.jpg)

### Notes:
Увеличение меньшего класса. Подход с увеличением количества объектов меньшего класса особенно актуален для работы с небольшими наборами данных, где удаление объектов большего класса может существенно сказаться на точности классификации. Выделяют следующие основные алгоритмы:
1) Алгоритм случайного увеличения (Random Oversampling) заключается в дублировании случайно выбранных объектов меньшего класса.
2) Алгоритм SMOTE создает синтетические примеры для меньшего класса, основываясь на существующими экземплярами, что увеличивает количество объектов редкого класса без создания копий существующего.
3) Алгоритм Borderline-SMOTE генерирует синтетические примеры только на границе классов, что улучшает способность модели различать классы, близкие к границе решения.
4) Алгоритм ADASYN создает синтетические примеры для меньшего класса с учетом их распределения и весов классов.

<!-- Slide number: 24 -->
# SMOTE (Synthetic Minority Over-sampling Technique)

![https://miro.medium.com/v2/resize:fit:745/1*47QrcrXtnkelH5nUHKkq7g.png](Picture2.jpg)
The SMOTE algorithm operates through a straightforward four-step process:
Select a minority class as the input vector.
Identify its k nearest neighbors (where the value of k_neighbors is defined as an argument in the SMOTE() function).
Pick one of these neighbors and generate a synthetic point along the line connecting the current point and the chosen neighbor.
Iteratively execute these steps until achieving a balanced distribution of the data.

![https://miro.medium.com/v2/resize:fit:875/0*fwc5Fo-sHaboXAl_.png](Picture4.jpg)

### Notes:
SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picingk a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.

<!-- Slide number: 25 -->
# Borderline-SMOTE
Borderline-SMOTE is a popular extension of the Synthetic Minority Oversampling Technique (SMOTE). Its main idea is to selectively oversample only those minority class instances that are considered "borderline" or "limit examples", rather than oversampling all minority examples or a random subset. This approach addresses the problem of overgeneralization that can occur with the original SMOTE algorithm, which generates synthetic samples without considering neighboring examples, potentially increasing overlapping between classes.

Borderline-SMOTE1: Generates synthetic examples by interpolating between a borderline minority instance and its positive (minority class) nearest neighbors.

Borderline-SMOTE2: Generates synthetic examples not only from borderline minority instances and their positive nearest neighbors but also by interpolating with their nearest negative (majority class) neighbors, aiming to push the boundary further into the majority class region. The new examples are generated closer to the minority class by multiplying the difference between the instance and its nearest negative neighbor by a random number between 0 and 0.5.

![](Объект5.jpg)

<!-- Slide number: 26 -->
# ADASYN (Adaptive Synthetic Sampling)

![The use of ADASYN to balance the dataset 2.4. Weighted SMOTE(W-SMOTE). | Download Scientific Diagram](Picture2.jpg)
Core Idea: An advanced oversampling technique that generates synthetic minority-class samples adaptively, focusing on regions where class separation is ambiguous.
Key Mechanism:
Density-Based Weighting:
Assigns higher synthesis priority to minority samples near decision boundaries.
Adaptive Generation:
Creates more synthetic data for "hard-to-learn" minority instances.

![](Рисунок4.jpg)

<!-- Slide number: 27 -->

# SMOTE vs Borderline SMOTE vs ADASYN
| Method | Advantages | Disadvantages | When to Use |
| --- | --- | --- | --- |
| SMOTE | - Simple, effective for linear separability. | - Blind to class overlap/noise. | Baseline imbalance; low computational budget. |
| (Chawla, 2002) | - Preserves minority class distribution. | - May create unrealistic synthetic samples. |  |
| Borderline SMOTE | - Focuses on "dangerous" minority samples near boundaries. | - Still sensitive to outliers. | High class overlap; noisy datasets. |
| (Han, 2005) | - Better than SMOTE for non-linear problems. | - Complex to tune (requires k-NN parameter). |  |
| ADASYN | - Adaptive: more samples in hard-to-learn regions. | - Computationally expensive. | Severe imbalance with complex distributions. |
| (Han, 2008) | - Reduces bias more aggressively than SMOTE. | - Risk of overfitting due to noise amplification. |  |

<!-- Slide number: 28 -->
| Algorithm | Authors | Year | Key Improvement |
| --- | --- | --- | --- |
| SMOTE | Chawla et al. | 2002 | Basic synthetic minority oversampling. |
| Borderline-SMOTE | Han et al. | 2005 | Focuses on minority samples near decision boundaries. |
| ADASYN | He et al. | 2008 | Adaptive synthesis based on local imbalance severity. |
| Safe-Level-SMOTE | Bunkhumpornpat et al. | 2009 | Generates samples only in "safe" regions (avoiding noise). |
| SMOTE-ENN | Batista et al. | 2004 | Combines SMOTE with Edited Nearest Neighbors (ENN) for cleaning noisy samples. |
| SMOTE-Tomek Links | Batista et al. | 2004 | Integrates Tomek Links for majority-class undersampling post-SMOTE. |
| Cluster-Based SMOTE | Cieslak et al. | 2008 | Applies SMOTE within minority-class clusters to preserve substructures. |
| SVM-SMOTE | Nguyen et al. | 2011 | Uses SVM to identify boundary regions for synthetic sample generation. |
| MWMOTE | Barua et al. | 2014 | Weighted synthesis based on both majority and minority densities. |
| Kernel-SMOTE | Sandhan & Choi | 2014 | Kernel-based feature space transformation for non-linear data. |
| G-SMOTE | Douzas & Bacao | 2019 | Geometric SMOTE: generates samples in latent space via geometric transformations. |
| L-SMOTE | Liang et al. | 2020 | Leverages local distributions for more realistic synthetic samples. |

<!-- Slide number: 29 -->
| Algorithm | Authors | Year | Key Innovation | Use Case |
| --- | --- | --- | --- | --- |
| G-SMOTE | Douzas & Bacao | 2019 | Geometric synthesis in latent space using interpolation weights. | High-dimensional data (e.g., images). |
| DBSMOTE | Zhang et al. | 2020 | Density-based SMOTE with kernel density estimation for minority clusters. | Non-uniformly distributed minority data. |
| L-SMOTE | Liang et al. | 2020 | Local distribution-aware synthesis using Gaussian mixtures. | Imbalanced regression tasks. |
| GraphSMOTE | Zhao et al. | 2021 | Graph neural network (GNN)-based oversampling for node classification. | Graph-structured data. |
| Diff-SMOTE | Kim et al. | 2022 | Diffusion model-based synthetic sample generation. | Complex data (e.g., medical imaging). |
| SMOOTH-GAN | Sampath et al. | 2022 | GAN-guided SMOTE with gradient penalty for realistic synthesis. | Tabular and image data. |
| FedSMOTE | Xu et al. | 2023 | Federated learning-compatible SMOTE for decentralized data. | Privacy-sensitive applications (e.g., healthcare). |
| TimeSMOTE | Torgo et al. | 2023 | SMOTE variant for temporal data with sequence-aware interpolation. | Time-series imbalance (e.g., IoT/sensors). |
| LLM-SMOTE | Chen & Liu | 2024 | Leverages LLM embeddings (e.g., BERT) for text data oversampling. | NLP tasks with rare classes. |

<!-- Slide number: 30 -->
# Undersampling
Undersampling balances imbalanced datasets by reducing the number of instances in the majority class, either randomly or via strategic selection (e.g., removing redundant or noisy samples). This method mitigates model bias by preventing the classifier from being overwhelmed by the dominant class. However, it risks losing valuable information if critical majority-class examples are discarded. Undersampling is computationally efficient and ideal for large-scale datasets, but performance depends on the retention of meaningful patterns in the reduced subset.

![](Рисунок6.jpg)
Random undersampling process

<!-- Slide number: 31 -->
# Balanced Approach:Undersampling Majority Class
Random Oversampling balances class distribution by randomly duplicating minority-class instances, while Random Undersampling randomly removes majority-class instances. Both are simple but have drawbacks: oversampling can lead to overfitting, and undersampling may discard useful information. They work best when combined with other techniques or when data redundancy is high.

Tomek Links (Boundary Cleaning)
Tomek Links identifies and removes noisy or borderline majority-class instances. A pair of nearest neighbors from opposite classes is called a Tomek Link—the majority-class instance in such a pair is deleted, sharpening the boundary between classes. Unlike random undersampling, Tomek Links selectively cleans the dataset, improving classifier generalization.

ENN (Edited Nearest Neighbours – Noise Removal)
ENN is an undersampling method that removes instances misclassified by their *k* nearest neighbors (e.g., a majority-class sample surrounded by minority-class points). By eliminating such noisy or ambiguous examples, ENN simplifies the decision boundary and enhances model robustness. It is often applied to the majority class but can also clean minority-class outliers.

### Notes:
Random Oversampling / Undersampling (случайная передискретизация)
Random Oversampling — это простейший метод балансировки, который случайным образом дублирует примеры миноритарного класса, пока распределение не станет сбалансированным. Random Undersampling, наоборот, удаляет случайные примеры мажоритарного класса. Оба метода просты в реализации, но могут приводить к переобучению (oversampling) или потере важной информации (undersampling).
Tomek Links (очистка границы между классами)
Tomek Links — это метод undersampling, который находит пары ближайших соседей из разных классов и удаляет пример мажоритарного класса, если он образует такую пару. Это помогает "очистить" границу между классами, убирая шумные или противоречивые точки. В отличие от случайного undersampling, Tomek Links действует более избирательно, улучшая качество разделения классов.
ENN (Edited Nearest Neighbours — удаление шумных примеров)
ENN — это метод undersampling, который удаляет те объекты, класс которых не совпадает с классом большинства его k ближайших соседей. Таким образом, алгоритм убирает выбросы и шумные примеры, упрощая классификацию. ENN может применяться как к мажоритарному, так и к миноритарному классу, но чаще используется для очистки majority-класса от примеров, мешающих корректному обучению модели.

<!-- Slide number: 32 -->
# Random Undersampling

![Illustration of random undersampling technique | Download Scientific Diagram](Picture4.jpg)
Random undersampling is the simplest method. The idea of undersampling, generally, is reducing the number of examples in the majority class by that the balance between minority and majority classes is reached in the data distribution. Random Undersampling (RUS) works by randomly selecting a subset of the majority class samples. This selected subset is combined with the minority class to form the balanced training dataset.

<!-- Slide number: 33 -->
# Tomek Links (Boundary Cleaning)
Tomek links identify and remove pairs of data points where one point comes from the majority class and the other from the minority class, and they are nearest neighbors to each other.
Tomek links effectively removed instances from the majority class (non-churners) that were closest to minority class instances (churners)
This cleaning of the decision boundary created a more balanced dataset while preserving the most informative instances
The technique removed approximately 5% of the majority class samples, specifically those creating ambiguity at the class boundaries

![](Объект3.jpg)
Tomek links represent a mathematically elegant approach to cleaning decision boundaries in classification problems. Formally defined, a pair of instances (x₁, x₂) is considered a Tomek link if:
1). x₁ belongs to class C₁ and x₂ belongs to class C₂ (different classes)
2). The distance d(x₁, x₂) is minimal
3). There exists no example x₃ such that d(x₁, x₃) < d(x₁, x₂) or d(x₂, x₃) < d(x₁, x₂)

<!-- Slide number: 34 -->
# ENN (Edited Nearest Neighbours – Noise Removal)
The ENN technique operates in the following steps:
1. Compute the k nearest neighbors for each instance in the dataset.2. Compare the class label of each instance with the class label of its k nearest neighbors.3. Remove instances from the majority class that are misclassified by their neighbors, thereby refining the dataset.

The Edited Nearest Neighbour (ENN) technique offers several benefits for addressing imbalanced classification challenges:
1. Noise Reduction: By eliminating noisy and misclassified instances, ENN contributes to a cleaner and more accurate dataset.2. Enhanced Generalization: The removal of redundant instances enhances the generalization capacity of classifiers trained on the refined dataset.3. Preservation of Decision Boundaries: ENN maintains the integrity of decision boundaries, thus ensuring that critical patterns are retained.

![](Рисунок3.jpg)

<!-- Slide number: 35 -->
# RUS vs Tomek Links vs ENN
| Method | Advantages | Disadvantages | When to Use |
| --- | --- | --- | --- |
| Random Undersampling | - Simple and fast to implement.- Reduces computational costs. | - High risk of losing important majority class information.- May remove useful patterns. | Quick baseline solutions.When majority class is extremely large. |
| Tomek Links | - Zero information loss from original data.- Effectively cleans class boundary regions.- Works synergistically with SMOTE.- Simple distance-based approach. | - May amplify noise if present in minority class.- Doesn't actually balance class distribution.- May remove useful majority samples.- Sensitive to distance metric choice. | Preprocessing step before other methods.Datasets with ambiguous boundaries. |
| (Tomek, 1976) |  |  |  |
| Edited Nearest Neighbors (ENN) | - Aggressively removes noisy samples from both classes.- Improves model generalization.- Works well with various classifiers. | - Can be too aggressive, removing valid patterns.- Computationally intensive (k-NN based).- Sensitive to k parameter selection. | Noisy datasets requiring clean decision boundaries.High-precision applications. |
| (Wilson, 1972) |  |  |  |
Key features of this format:
Clear separation of methods with original authors and years
Bullet-point advantages/disadvantages for quick scanning
Specific use-case recommendations in right column
Consistent structure matching your SMOTE/Borderline/ADASYN example
Would you like me to add any of these elements:
Performance metrics (F1-score ranges)
Code implementation examples
Visual workflow diagrams
Specific library references (e.g., imblearn)
New chat

<!-- Slide number: 36 -->
| Algorithm | Authors | Year | Key Improvement |
| --- | --- | --- | --- |
| Random Undersampling | - | - | Basic random reduction of majority class samples. |
| Tomek Links | Tomek | 1976 | Removes overlapping majority samples near class boundaries. |
| ENN | Wilson | 1972 | Removes samples misclassified by k-NN from both classes. |
| CNN | Hart | 1968 | Condensed Nearest Neighbor: keeps only informative boundary samples. |
| NCR | Laurikkala | 2001 | Neighborhood Cleaning Rule: aggressive ENN variant for noisy data. |
| OSS | Kubat & Matwin | 1997 | One-Sided Selection: combines Tomek Links and CNN. |
| NearMiss | Zhang & Mani | 2003 | Three versions with different majority sample selection strategies. |
| Instance Hardness | Smith et al. | 2014 | Removes samples based on classification difficulty. |
| Cluster Centroids | - | - | Replaces majority cluster with centroids to preserve topology. |
| Random Prototypes | - | - | Generates synthetic majority prototypes for reduction. |

<!-- Slide number: 37 -->
| Algorithm | Authors | Year | Key Improvement |
| --- | --- | --- | --- |
| GNG-U | Douzas et al. | 2019 | Uses Growing Neural Gas to preserve majority class topology during reduction. |
| SUNDO | Krawczyk & Woźniak | 2020 | Stochastic UNDersampling with Oversampling for dynamic imbalance handling. |
| IDM | Zhou et al. | 2021 | Information Density Maximization: keeps samples with highest local density. |
| GraphUndersampling | Zhao et al. | 2022 | Graph-based selection using node centrality for relational data. |
| DBSMOTE-U | Zhang et al. | 2022 | Density-Based SMOTE extension for majority class reduction. |
| Auto-Encoder Undersampling | Li & Liu | 2023 | Uses auto-encoder reconstruction error to identify redundant majority samples. |
| Focal Undersampling | Wang et al. | 2023 | Applies focal loss principles to remove "easy" majority samples. |
| DiffUndersample | Chen & Wong | 2024 | Diffusion model-guided selection of majority samples. |

<!-- Slide number: 38 -->
# Oversampling vs Undersampling
| Parameter / Method | Oversampling | Undersampling |
| --- | --- | --- |
| Definition | Increasing the number of samples in the underrepresented class. | Reducing the number of samples in the overrepresented class. |
| Goal | To improve the representation of rare classes. | To reduce the dominance of majority classes. |
| Methods | Synthetic Minority Over-sampling Technique (SMOTE), Adaptive Synthetic (ADASYN) | Random Undersampling, Tomek Links, Cluster Centroids |
| Advantages | Prevents loss of important information. Increases diversity of training examples. | Reduces training time. Lowers the risk of overfitting on dominant classes. |
| Disadvantages | May lead to overfitting. Increases training time. | May result in loss of important information. Reduces the amount of available training data. |
| Application | Better suited when data quantity is limited. | Preferred when dealing with large datasets. |
| Use Cases | Small datasets, imbalanced medical data. | Large datasets, such as in credit scoring. |
| Risks | Synthetic data may not always adequately reflect reality. | Potential loss of important data patterns. |

<!-- Slide number: 39 -->
# Hybrid Sampling: Combining Oversampling & Undersampling
When to Use Hybrid?
Moderate-to-severe imbalance (e.g., 1:10 class ratio)
Noisy datasets requiring cleaning
Critical applications (e.g., fraud detection, medical diagnosis)
Why Hybrid?
Mitigates weaknesses of standalone methods
Preserves key patterns while balancing classes
Improves generalization vs. pure over/undersampling

Popular Hybrid Techniques
| Method | Oversampling | Undersampling | Key Feature |
| --- | --- | --- | --- |
| SMOTE-ENN | SMOTE | Edited NN | Aggressive noise cleaning |
| SMOTE-Tomek | SMOTE | Tomek Links | Boundary refinement |
| SMOTE-RSB\* | SMOTE | Rough Set Theory | Handles vague boundaries |
| ADASYN-CC | ADASYN | Cluster Centroids | Density-aware reduction |

<!-- Slide number: 40 -->
# Hybrid resampling process

![](Объект3.jpg)

<!-- Slide number: 41 -->
# Experiments and Results

<!-- Slide number: 42 -->
# Description of selected datasets
| Dataset Name | Dataset Size | Classification Type | Class Ratio |
| --- | --- | --- | --- |
| Fetal Health Classification | ~2,000 | Multiclass | 8:100 and 13:100 |
| Cerebral Stroke Prediction | ~43,000 | Binary | 2:100 |
| Credit Risk Assessment | ~32,000 | Binary | 22:100 |
| Credit Rating Classification | 100,000 | Multiclass | 17:100 and 29:100 |
| Diabetes Health Indicators | ~254,000 | Multiclass | 2:100 and 15:100 |

<!-- Slide number: 43 -->
# Tech Stack
Python was selected as the primary programming language due to the availability of the imbalanced-learn library, which implements data balancing techniques. Supporting libraries included:
Pandas, NumPy, and Scikit-learn for dataset processing
Matplotlib and Seaborn for visualization
XGBoost and LightGBM for ensemble modeling
Optuna framework for hyperparameter tuning

<!-- Slide number: 44 -->
# Metric, classification algorithms

![](Объект3.jpg)
Ensembles
RandomForest
Granient Boosting
XGBoost
AdaBoost
LightGBM
Classical algorithms
Decision Tree
MLP
k-nearest neighbors
Naïve Bayes

<!-- Slide number: 45 -->
# Isolated Balancing Usage
| Classes Distribution | Initial dataset |  | Balanced dataset |  | Best model and balancing technic |
| --- | --- | --- | --- | --- | --- |
|  | Classical model | Ensemble | Classical model | Ensemble |  |
| 8:100 and 13:100 | 0,873 | 0,920 +5,38% | 0,880 +0,80% | 0,934 +6,99% | RF – Random Oversampling RF – SMOTE RF – B-SMOTE GB – Random Undersampling GB – OSS XGBoost – Random Oversampling LightGBM – TomekLinks LightGBM – OSS |
| 2:100 | 0,650 | 0,502 -22,77% | 0,753 +15,85% | 0,779 +19,85% | AdaBoost – Random Oversampling |
| 22:100 | 0,839 | 0,864 +2,98% | 0,839 +0% | 0,878 +4,65% | XGBoost – Random Oversampling |
| 17:100 and 29:100 | 0,732 | 0,804 +9,84% | 0,770 +5,19% | 0,825 +12,70% | RF – Random Oversampling RF – SMOTE RF – B-SMOTE RF – B-SMOTE SVM RF – ADASYN RF – Random Undersampling |
| 2:100 and 15:100 | 0,453 | 0,391 -13,69% | 0,490 +8,17% | 0,468 +3,31% | GaussianNB – SMOTE GaussianNB – B-SMOTE SVM GaussianNB – ADASYN |

<!-- Slide number: 46 -->
# Determining the optimal class ratio

![график 3_1](Picture3.jpg)
Results depending on the class ratio: a – «Cerebral stroke» dataset; b – «Credit risk» dataset

<!-- Slide number: 47 -->
# Determining the optimal class ratio

![график 4_1](Picture2.jpg)
Results depending on class ratio and class to balance in dataset: a – «Fetal Health», b – «Credit Rating»

<!-- Slide number: 48 -->
# The contribution of balancing and model settings to the final result

![график 5](Picture3.jpg)
Contribution of the stages of working with a dataset to the final result

<!-- Slide number: 49 -->
# Some Results
The use of methods for increasing a smaller class using classical classification methods improved the quality of models by an average of 11%, while non-negative dynamics was observed for sets. The methods of reducing the larger class gave unstable results: both a deterioration of up to 41% and an improvement of up to 54% were observed, the average value for all sets was close to zero. Without balancing, the ensemble methods showed results on average 5% worse than the classical ones; with balancing, the improvement reached 13-18% compared to the classical models on the initial set.
The methods of increasing the smaller class had a more pronounced effect, this is typical for all the data sets considered. Combining the method of increasing a smaller class and the method of decreasing a larger class is advisable in cases where there are time and computing resources, and the best possible quality of the model is required. Moreover, those algorithms should be chosen for the combination, which in isolation give the best increase in the quality of the model. Otherwise, even the use of widespread and well-known combinations can lead to a deterioration in the quality of the model if these algorithms do not produce a significant positive result in isolation from the combination.

<!-- Slide number: 50 -->
# Some Results
The optimal class ratio after balancing for most of the data sets considered is in the range between 1:1 and 2:1, where the first number corresponds to the number of objects of the initially smaller class. Balancing with a ratio of less than 1:1 turned out to be impractical, and balancing only one smaller class instead of all in a multiclass classification. At the same time, the increase in model quality when selecting the class ratio was not very pronounced and amounted to 1-3%, therefore, in conditions of limited time resources, balancing to the level of 1:1 or 2:1 can be recommended without more careful selection of the exact ratio.
For all the data sets considered, the lack of balancing was not compensated by the selection of hyperparameters. The quality of the models when combining the method of increasing the smaller class, the method of decreasing the larger class, the ensemble algorithm and the selection of hyperparameters turned out to be the best. The largest contribution was made by increasing the smaller class and using the ensemble model instead of the classical one.

<!-- Slide number: 51 -->
# Thank you for your attention!