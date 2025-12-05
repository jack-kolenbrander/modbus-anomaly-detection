from pymodbus.client import ModbusTcpClient

# Creates ModbusTCPClient that connects to localhost port 502
client = ModbusTcpClient('127.0.0.1', port=502)
# Establish TCP Connection, performs TCP handshake
client.connect()

# Send Modbus write single coil command (0x05), writes to coil at address 1, and sets value to 1 (ON)m targetting device ID 1
client.write_coil(address=1, value=True, device_id=1)

# Send Modbus read coils command (0x01) to read from coil at address 1 from device ID 1
result = client.read_coils(address=1, count=1, device_id=1)
# Check for Modbus Error Message
if not result.isError():
    print(f"Coil value: {result.bits[0]}")
else:
    print(f"Error: {result}")
# Close TCP Connection
client.close()