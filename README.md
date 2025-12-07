# Modbus Protocol Anomaly Detection

Machine learning approaches for detecting cyberattacks in industrial control system networks.

## Overview

This repository explores various machine learning techniques for identifying anomalies in Modbus TCP network traffic. The project compares multiple classification and anomaly detection approaches to address the challenge of detecting rare but critical attack patterns in industrial control systems.

## Repostitory Structure
```
modbusAnomalyDetection/
├── anomalyDetection/
│   ├── decisionTree/          # Decision Tree Implementation             
├── ModbusTCP_Dataset/         # Dataset documentation
└── README.md
```

## Models

Each model implementation includes:
- Source code
- Performance metrics
- Confusion matrices
- Feature importance analysis
- Documentation

**Current Status:**
- ✅ Decision Tree - Complete

## Dataset

This project uses the **FARAONIC Modbus TCP Cybersecurity Dataset**, developed as part of the FARAONIC framework (Framework for Anomaly Recognition and Analysis in Operational Networks for Industrial Cybersecurity).

The dataset contains Modbus TCP network traffic with normal operations and five attack classifications:
- **DDOS** - Denial of Service
- **FAKE_REG** - Fake Register Manipulation
- **FUNC_TAMPER** - Function Code Tampering
- **MASQ** - Masquerade Attacks
- **UNIT_ENUM** - Unit Enumeration

See `ModbusTCP_Dataset/` for full dataset documentation and source.

