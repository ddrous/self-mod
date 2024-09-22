## Run the "main_ncf copy i.py" script for each i in the range of 0 to 7.

import os

for i in range(8):
    print("==== Running main_ncf_copy" + str(i) + ".py ====", flush=True)
    os.system("python main_ncf_copy" + str(i) + ".py > nohup.log")
