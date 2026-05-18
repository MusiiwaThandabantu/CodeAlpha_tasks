# Iris Dataset Classification with Decision Trees and Ensemble Models
## Project Overview
This project demonstrates how to build, visualize, and evaluate Decision Tree, Random Forest, and Gradient Boosting models using the classic Iris dataset. It includes exploratory data analysis (EDA), visualization, model training, and performance comparison.

## Installation
Before running the notebook, install the required libraries:

bash
pip install pydotplus
pip install graphviz   # install via command line
Libraries Used
pandas & numpy → Data handling

matplotlib → Visualization

scikit-learn → Machine learning models & evaluation metrics

pydotplus & graphviz → Decision tree visualization

## Workflow
Load dataset → Import Iris data from CSV.

Exploratory Data Analysis → Inspect dataset shape, class distribution, and missing values.

Visualization → Scatter plots of sepal length vs sepal width by species.

Preprocessing → Split into features and target, then train/test sets.

Decision Tree → Train, visualize, and evaluate with accuracy, precision, recall, and F1 score.

Random Forest → Train ensemble model and compare metrics.

Gradient Boosting → Train boosting model and compare metrics.

Model Comparison → Evaluate performance across all models.

## Evaluation Metrics
Accuracy → Overall correctness

Precision → Avoiding false positives

Recall → Capturing true positives

F1 Score → Balance between precision and recall

## Decision Tree Visualization
The decision tree is exported and saved as a PNG file using Graphviz and pydotplus, allowing clear interpretation of splits and decision rules.

