# Modbus client
from pymodbus.client import ModbusTcpClient as ModbusClient
import time

# Create Modbus TCP Client on localhost port 502 and connect
print('Start Modbus Client')
client = ModbusClient(host='127.0.0.1', port=502)
client.connect()

# Initalize starting register address to zero and device ID to 1
reg = 0
device_id = 1

# Initialize data
data = [0.1, 1.1, 2.1, 3.1, 4.1]

# Create loop where each iteration is one complete write-read cycle
for i in range(10):
    print('-'*5, 'Cycle', i, '-'*30)
    time.sleep(1.0)
    
    # Increment data by one
    for idx, d in enumerate(data):
        data[idx] = d + 1
    
    # Display values that will be written holding registers (40001 to 40005)
    print('Write', data)
    
    # Convert float data to integers for writing
    int_data = [int(d) for d in data]
    
    # Write multiple registers - pymodbus 3.x handles encoding automatically
    # Send Modbus Write Multiple Registers command (0x10)
    result = client.write_registers(address=reg, 
                                    values=int_data,
                                    device_id=device_id)
    
    # Read holding registers
    # Send Modbus Read Holding Registers command (0x03)
    # Read 5 registers starting from address 0 from Device ID 1
    rd = client.read_holding_registers(address=reg, 
                                       count=len(data), 
                                       device_id=device_id).registers
    print('Read', rd)

client.close()