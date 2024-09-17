## Run the "main_ncf copy i.py" script for each i in the range of 0 to 7.

import os

for i in range(8):
    print("Currently running main_ncf_copy" + str(i) + ".py", flush=True)
    os.system("python main_ncf_copy" + str(i) + ".py > nohup.log")

    print("Currently running main_ncf_copy" + str(i+8) + ".py", flush=True)
    os.system("python main_ncf_copy" + str(i+8) + ".py > nohup.log")
