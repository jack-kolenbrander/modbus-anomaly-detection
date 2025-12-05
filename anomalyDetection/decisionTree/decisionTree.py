from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Import Datasets
df_train = pd.read_csv('../ModbusTCP_Dataset/Modbus_TCP_Cybersecurity_Dataset_Training.csv', sep=';', low_memory=False)
df_val = pd.read_csv('../ModbusTCP_Dataset/Modbus_TCP_Cybersecurity_Dataset_Validation.csv', sep=';', low_memory=False)

# Extract feature columns from 48 total features
feature_columns = [
    # IP and Network Features
    # 'IP_src', # Source IP
    # 'IP_dst', # Destination IP
    'IP_len', # IP packet length
    'IP_ttl', # Packet TTL

    #TCP Features
    'TCP_window', # TCP window size
    'TCP_flags', # TCP flag

    # Modbus Features
    'ModbusTCPRequest_func_code', # Modbus request function code 
    'ModbusTCPRequest_unit_id', # Device Identification
    'ModbusTCPRequest_trans_id', # Transaction ID
    'ModbusTCPResponse_func_code', # Modbus response function code
    'ModbusReadDiscreteInputsRequest_reference_number',
    'ModbusWriteMultipleCoilsRequest_reference_number',
    'ModbusTCPRequest_length', 
    'ModbusTCPResponse_length',  
    'ModbusTCPResponse_trans_id'  
]

# Create X and y
X_train = df_train[feature_columns].copy()
y_train = df_train['Classification']

X_val = df_val[feature_columns].copy()
y_val = df_val['Classification']

# Split into training set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size = 0.2)

# Create decision tree classifier
dtc = DecisionTreeClassifier(max_depth=None, random_state=17)
dtc.fit(X_train, y_train)

# Predict validation set
y_pred = dtc.predict(X_val)

# Create classification report
print(classification_report(y_val, y_pred))

# Create confusion matrix
class_labels = dtc.classes_
cm = confusion_matrix(y_val, y_pred, labels=class_labels)
# Plot confusion matrix 
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Modbus Attack Detection - Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Create feature importance table
# Feature importance source: https://www.geeksforgeeks.org/machine-learning/understanding-feature-importance-and-visualization-of-tree-models/#1-decision-tree-feature-importance
importance = dtc.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)
print(importance_df.to_string(index=False))
