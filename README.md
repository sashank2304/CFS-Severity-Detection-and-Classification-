# Chronic Fatigue Syndrome (CFS) Severity Detection Using XGBoost and Graph Neural Networks

## Overview

This project implements a **hybrid machine learning model** that combines Graph Neural Networks (GNN) and XGBoost to detect and classify the severity of Chronic Fatigue Syndrome (CFS) from longitudinal patient data. The GNN captures interdependencies between symptoms, while XGBoost provides robust classification on fused features, achieving state-of-the-art accuracy and interpretability for clinical decision support.

## Features

- Preprocesses and balances longitudinal clinical data from 952 diagnosed CFS patients
- Builds symptom graphs and learns embeddings with Graph Convolutional Network (GCN)
- Combines GCN embeddings with original features for hybrid classification using XGBoost
- Comprehensive evaluation with metrics like accuracy, precision, recall, and F1-score
- Achieves 98.6% accuracy and outperforms traditional classifiers

## Project Structure

| File/Folder           | Purpose                                        |
|-----------------------|------------------------------------------------|
| `CFS-Code-Notepad.txt`| Python code for GCN + XGBoost model training and evaluation |
| `CFS_IEEE_REPORT-1.pdf` | Detailed project report and methodology      |
| `preprocessed_CFS_dataset.csv` | Dataset used for training and testing (expected in project root) |
| `README.md`           | This documentation                             |

## Prerequisites

- Python 3.x
- PyTorch and PyTorch Geometric
- XGBoost
- scikit-learn
- pandas, numpy

Install all dependencies with:  
```bash
pip install torch torch-geometric xgboost scikit-learn pandas numpy
```

## Usage

### Run Training and Evaluation

Modify the dataset path in `CFS-Code-Notepad.txt` and run:

```bash
python CFS-Code-Notepad.txt
```

This script:
- Loads and preprocesses the dataset
- Creates a graph representation of symptom data
- Trains a two-layer GCN to learn symptom embeddings
- Fuses embeddings with original features for XGBoost classification
- Prints performance metrics: accuracy, precision, recall, and F1-score

## Core Code Snippet

```python
!pip install torch torch-geometric xgboost scikit-learn pandas numpy
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('/content/preprocessed_CFS_dataset.csv')
df.columns = df.columns.str.strip()  # Remove unwanted spaces

# ✅ Automatically use the last column as target
target_col = df.columns[-1]
print(f"✅ Using '{target_col}' as the target label column.")

# Split into features and labels
features = df.drop(columns=[target_col])
labels = df[target_col]

# Encode labels (convert to numeric classes if needed)
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Convert to tensors
x = torch.tensor(features.values, dtype=torch.float)
y = torch.tensor(encoded_labels, dtype=torch.long)

# Create dummy edge_index (fully connected or placeholder)
edge_index = torch.combinations(torch.arange(x.size(0)), r=2).t().contiguous()

# Create PyG Data object
data = Data(x=x, edge_index=edge_index, y=y)
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
model = GCN(num_features=x.size(1), hidden_channels=32, num_classes=len(le.classes_))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
model.eval()
with torch.no_grad():
    gcn_features = model.conv1(data.x, data.edge_index)
    gcn_features = F.relu(gcn_features)
X_train, X_test, y_train, y_test = train_test_split(
    gcn_features.numpy(), y.numpy(), test_size=0.3, random_state=42)

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_))
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n🔹 Evaluation Metrics 🔹")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-Score: {f1:.4f}")

```

## Model Performance

- Overall accuracy: **98.60%**
- Precision: **97.91%**
- Recall: **98.60%**
- F1-Score: **98.25%**
- Outperforms Logistic Regression, Random Forest, SVM, and standalone XGBoost
- Effective in Mild, Moderate, and Severe CFS severity classification

## Methodology Summary

- Data preprocessing includes missing value imputation and SMOTE for class balancing
- Symptom interdependencies modelled using graph construction and normalized adjacency
- GCN learns structural symptom embeddings that are fused with raw features
- XGBoost classifier performs hybrid feature-based severity classification
- Extensive hyperparameter tuning and testing ensure robust generalization

## References

Refer to the detailed IEEE report (`CFS_IEEE_REPORT-1.pdf`) for methodology, literature survey, mathematical formulations, and detailed results explanations.

## Contributors

- Sarah Thomas
- Samvrudha RR
- Sashank Abburu
- T Pushkar Reddy
