# By Nick Erickson
# Visualizing Data Testing

import sys
import time

print('hello', end="")
sys.stdout.flush()
time.sleep(2)
print('\rhey', end="")
sys.stdout.flush()

for i in range(1000):
    if i % 10 == 0:
        print('\r', 'Learning', '(', str(i), '/', str(1000), ')', end="")
        sys.stdout.flush()
    time.sleep(0.01)
print('\r', 'Learning Complete                                           ')
