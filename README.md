Task 6: K-Nearest Neighbors (KNN) Classification

This project is part of my AI/ML Internship, aimed at implementing the **K-Nearest Neighbors (KNN)** algorithm for classification tasks using the classic **Iris dataset**.

---

##  Objective

To understand and implement **KNN for classification**, exploring its:
- Distance-based decision-making
- Sensitivity to feature scaling
- Behavior with different values of **K**

Tools & Libraries Used

- Python 
- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`

---

##  Files in This Repository

| File Name                  | Description |
|---------------------------|-------------|
| `task6_knn_classifier.py` | Full classification with accuracy and confusion matrix |
| `task6_knn_boundaries.py` | Visualizes decision boundaries using first two features |

---

# Dataset: Iris Flower Dataset

- 3 classes: Setosa, Versicolor, Virginica  
- 4 features: sepal length, sepal width, petal length, petal width  
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)

---

What I Explored

- Normalization before distance-based learning
- Choosing **K** using performance observations
- Understanding bias vs variance in KNN
- Plotting **decision boundaries** with 2D feature sets

- Achieved **high classification accuracy** for multiple K values  
- Visualized how the model separates the classes in 2D space  
- Understood KNN's sensitivity to scaling and noise
How does KNN work?
   It finds the 'K' nearest data points (neighbors) and uses majority voting for classification.
Why normalize?
   Because distance-based models are sensitive to scale — larger ranges dominate.
How to choose K?
   Trial-and-error with performance metrics or cross-validation. Odd K preferred to avoid ties.

Time Complexity
   Training: O(1) (lazy learner); Prediction: O(n × d)

# How to Run
# For training and evaluation
python task6_knn_classifier.py

# For decision boundary visualization
python task6_knn_boundaries.py

