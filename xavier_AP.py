import subprocess
import os
import time

while True:
    time.sleep(1)
    result = subprocess.run(['lsusb'], stdout=subprocess.PIPE)
    print(result.stdout)
    
    if ("Intel Corp." in str(result.stdout)):
        print('here')
        time.sleep(3)
        os.system("sudo create_ap wlan0 eth0 SYL1 11111111 --freq-band 2.4 --no-virt -w 2")
        
        break
