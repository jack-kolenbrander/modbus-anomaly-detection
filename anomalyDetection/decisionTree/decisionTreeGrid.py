"""
Anomaly Detection - Decision Tree Model
Detects anomalies within Modbus ICS dataset
"""
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import Datasets
df_train = pd.read_csv('../../ModbusTCP_Dataset/Modbus_TCP_Cybersecurity_Dataset_Training.csv', sep=';', low_memory=False)
df_val = pd.read_csv('../../ModbusTCP_Dataset/Modbus_TCP_Cybersecurity_Dataset_Validation.csv', sep=';', low_memory=False)

# Extract feature columns from 48 total features
feature_columns = [
    # IP and Network Features
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
dtc = DecisionTreeClassifier(random_state=17)

# Create grid parameters
# Source: https://medium.com/biased-algorithms/grid-search-for-decision-tree-ababbfb89833
param_grid = {
    'max_depth': [10, 20, None],
    'min_samples_split': [10, 30, 50],
    'min_samples_leaf': [5, 10, 20],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None]  
}
# Create grid search
grid_search = GridSearchCV(estimator=dtc, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)
# Fit grid search
grid_search.fit(X_train, y_train)

# Output best parameters:
print(f"Best grid search parameters: {grid_search.best_params_}")
# Using the best parameters, predict validation set
best_dtc = grid_search.best_estimator_
y_pred = best_dtc.predict(X_val)

# Output accuracy
accuracy = accuracy_score(y_val,y_pred)
errors = int((1 - accuracy) * len(y_val))
print(f"Accuracy: {accuracy:.8f} Total Errors: {errors} out of {len(y_val)}")
# Create classification report
classificationReport = classification_report(y_val,y_pred)
print(classificationReport)

# Create confusion matrix
class_labels = best_dtc.classes_
cm = confusion_matrix(y_val, y_pred, labels=class_labels)
# Plot confusion matrix 
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Modbus Attack Detection - Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
plt.show()

# Create feature importance table
# Feature importance source: https://www.geeksforgeeks.org/machine-learning/understanding-feature-importance-and-visualization-of-tree-models/#1-decision-tree-feature-importance
importance = best_dtc.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)
print(importance_df.to_string(index=False))
