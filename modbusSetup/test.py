# Test what we can import
from pymodbus import datastore
import inspect

print("Available in datastore:")
print([item for item in dir(datastore) if 'Context' in item or 'Slave' in item])