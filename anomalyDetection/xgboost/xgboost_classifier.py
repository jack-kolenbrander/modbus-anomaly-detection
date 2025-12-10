from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

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

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)
# Create xgboost model
xgbClassifier = XGBClassifier(objective='multi:softmax', random_state=42, num_class=len(le.classes_), eval_metric='mlogloss', n_jobs=-1, tree_method='hist')

# Randomized search space for XGBoost
param_dist = {
    'max_depth': [6, 8],              
    'learning_rate': [0.05, 0.1],     
    'n_estimators': [200, 500],      
    'subsample': [0.8],               
    'colsample_bytree': [0.8],        
    'gamma': [0, 1],                  
    'min_child_weight': [1, 3],       
}

# Create grid search
random_search = RandomizedSearchCV(estimator=xgbClassifier, param_distributions=param_dist, cv=3, n_iter=20, scoring='f1_macro', n_jobs=-1, verbose=2)
# Fit grid search
random_search.fit(X_train, y_train_encoded)


# Use the best model
best_xgb = random_search.best_estimator_
# Print best parameters
print(best_xgb.get_params())
#Predict
y_pred = best_xgb.predict(X_val)

accuracy = accuracy_score(y_val_encoded,y_pred)
errors = int((1 - accuracy) * len(y_val))
print(f"Accuracy: {accuracy:.8f} Total Errors: {errors} out of {len(y_val)}")
# Create classification report
classificationReport = classification_report(y_val_encoded,y_pred,target_names=le.classes_)
print(classificationReport)

# Create confusion matrix
cm = confusion_matrix(y_val_encoded, y_pred)
# Plot confusion matrix 
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Modbus Attack Detection - Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Create feature importance table
# Feature importance source: https://www.geeksforgeeks.org/machine-learning/understanding-feature-importance-and-visualization-of-tree-models/#1-decision-tree-feature-importance
importance = best_xgb.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)
print(importance_df.to_string(index=False))