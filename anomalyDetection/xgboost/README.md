# XGBoost Classifier

## Project Overview
A XGBoost classifier leveraged to detect anomalies within Modbus TCP network traffic logs. 

## Model Overview


### Key Features
- 
## Dataset
- **Training samples**: ~3.3M Network Packets
- **Validation samples**: ~2.6M Network Packets
- **Features**: 13 Features selected from 48 available
- **Classes**: 6 (Normal traffic + 5 Attack Classifications)

## Feature Selection

**IP Layer**
- `IP_len` - Packet length
- `IP_ttl` - Time to live

**TCP Layer**
- `TCP_window` - Window size
- `TCP_flags` - Control flags

**Modbus Protocol**
- `ModbusTCPRequest_func_code` - Function code
- `ModbusTCPRequest_unit_id` - Device identifier
- `ModbusTCPRequest_trans_id` - Transaction ID
- `ModbusTCPRequest_length` - Request length
- `ModbusTCPResponse_func_code` - Response function code
- `ModbusTCPResponse_length` - Response length
- `ModbusTCPResponse_trans_id` - Response transaction ID
- `ModbusReadDiscreteInputsRequest_reference_number` - Starting address (reference number) for reading discrete inputs (Modbus Function 2)
- `ModbusWriteMultipleCoilsRequest_reference_number` - Starting address (reference number) for writing multiple coils (Modbus Function 15)

## Hyperparameter Training
RandomSearchCV with 3-fold cross-validation and F1-macro scoring:
- **max_depth**: [6, 8]              
- **learning_rate**: [0.05, 0.1]     
- **n_estimators**: [200, 500]      
- **subsample**: [0.8]               
- **colsample_bytree**: [0.8]        
- **gamma**: [0, 1]                  
- **min_child_weight**: [1, 3]   

## Model Architecture
```python
XGBClassifier(
    objective='multi:softmax',
    random_state=42, 
    num_class=len(le.classes_), 
    eval_metric='mlogloss', 
    n_jobs=-1, 
    tree_method='hist')

```

## Results



### Classification Report


### Feature Importance


### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

## Key Findings
- 

### Lessons Learned
- 
## References

### Technical Resources

- **Feature Importance**: GeeksforGeeks. ["Understanding Feature Importance and Visualization of Tree Models"](https://www.geeksforgeeks.org/machine-learning/understanding-feature-importance-and-visualization-of-tree-models/#1-decision-tree-feature-importance). Accessed December 2024.

### Dataset
- **FARAONIC Dataset**: [Modbus TCP Cybersecurity Dataset](https://www.kaggle.com/datasets/dataset-name). Kaggle, 2024.
https://medium.com/@fraidoonomarzai99/xgboost-classification-in-depth-979f11ef4bf9