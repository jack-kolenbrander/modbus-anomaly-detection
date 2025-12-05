from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSequentialDataBlock, 
    ModbusDeviceContext,
    ModbusServerContext
)

def run_async_server():
    
    nreg = 200
    
    # Create a proper device context (this is the slave context wrapper)
    device = ModbusDeviceContext(
        # Creates 200 discrete input registers starting at address zero and initailized to 15
        di=ModbusSequentialDataBlock(0, [15]*nreg),
        # Creates 200 coil registers startting at address zero and intialized to 16
        co=ModbusSequentialDataBlock(0, [16]*nreg),
        # Creates 200 holding registers and intialized to 17
        hr=ModbusSequentialDataBlock(0, [17]*nreg),
        # Creates 200 input registers intialized to 18
        ir=ModbusSequentialDataBlock(0, [18]*nreg)
    )
    
    # Create server context, Map device_id 1 to this context
    context = ModbusServerContext(devices={1: device}, single=False)
    # Start modbus TCP server using context hosted on localhost port 502
    StartTcpServer(context=context, address=("127.0.0.1", 502))

if __name__ == "__main__":
    print("Modbus server started on 127.0.0.1 port 502")
    run_async_server()