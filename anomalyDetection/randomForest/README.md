# Random Forest Classifier

## Overview
A random forest classifier leveraged to detect anomalies within Modbus TCP network traffic logs. 

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
GridSearchCV with 5-fold cross-validation and F1-macro scoring:
- **max_depth**: [10, 20, None]
- **min_samples_split**: [10, 30, 50]
- **min_samples_leaf**: [5, 10, 20]
- **criterion**: ['gini', 'entropy']
- **class_weight**: ['balanced', None]  

## Model Architecture
```python
DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=17
)
```