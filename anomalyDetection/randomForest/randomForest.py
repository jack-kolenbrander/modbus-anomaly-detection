"""
Anomaly Detection - Random Forest Classifier Model
Detects anomalies within Modbus ICS dataset
"""
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

# Create X and y datasets
X_train = df_train[feature_columns].copy()
y_train = df_train['Classification']

X_val = df_val[feature_columns].copy()
y_val = df_val['Classification']

# Create parameter grid

param_grid = {
    'n_estimators': [100, 200], # Number of decision trees
    'max_depth': [15, 20, 30], # Maximum splits from root to leaf per tree
    'min_samples_split': [15, 25], # Minimum samples required to split
    'min_samples_leaf': [2, 5], # Mimimum samples required in a leaf node
    'bootstrap': [True, False], # Bootstrap
    'class_weight': ['balanced', None]  
}

# Create random forest classifier
rfc = RandomForestClassifier(random_state=42)

# Create grid search
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)
# Fit grid search
grid_search.fit(X_train, y_train)
rfc.fit(X_train, y_train)
# Output best parameters:
print(f"Best grid search parameters: {grid_search.best_params_}")
# Using the best parameters, predict validation set
best_rfc = grid_search.best_estimator_
y_pred = best_rfc.predict(X_val)

# Output accuracy
accuracy = accuracy_score(y_val,y_pred)
errors = int((1 - accuracy) * len(y_val))
print(f"Accuracy: {accuracy:.8f} Total Errors: {errors} out of {len(y_val)}")
# Create classification report
classificationReport = classification_report(y_val,y_pred)
print(classificationReport)

# Create confusion matrix
class_labels = best_rfc.classes_
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
importance = best_rfc.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)
print(importance_df.to_string(index=False))
