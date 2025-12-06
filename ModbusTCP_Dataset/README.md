# Modbus Dataset 
This project uses the Modbus TCP Cybersecurity Dataset, licensed under CC BY-NC 4.0. The dataset was published on Kaggle by Fabio Araujo Fabres. 
Dataset source: https://www.kaggle.com/datasets/fabioaraujofabres/modbustcp-cybersecurity-dataset?resource=download&select=Modbus_TCP_+Cybersecurity_Dataset_Training.csv
Github Project: https://github.com/f4bio89/FARAONIC

## Dataset Description
The dataset consists of a training and validation set of Modbus/TCP data. The dataset was developed as part of the FARAONIC framework (Framework for Anomaly Recognition and Analysis in Operational Networks for Industrail Cybersecurity.)

### Dataset Generation
The datasets were generated within a virtualized industrial environment. The data is classified into six different classes: NORMAL, FAKE_REG, FUNC_TAMPER, UNIT_ENUM, DDOS, and MASQ.

From the dataset description:
| Class | Description |
|-------|-------------|
| NORMAL | Legitimate communication between OpenPLC and Factory I/O. |
| FAKE_REG | Injection of fake or manipulated Modbus registers. |
| FUNC_TAMPER | Unauthorized alteration of the Function Code. |
| UNIT_ENUM | Forced enumeration of Unit IDs. |
| DDOS | SYN Flood attack directed at the Modbus server. |
| MASQ | IP masquerading (spoofing) with Modbus/TCP field manipulation, especially `reference_number`. |