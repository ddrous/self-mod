#%%
import os

Ks = [10, 100, 1000]
taylor_orders = [0, 1, 2, 3]

## Run the files in succession. 
for to in taylor_orders:
    for K in Ks:
        print(f"\n\nCurrently running CAVIA with taylor_order={to} envs and K={K} shots ...", flush=True)
        current_file = f"CAVIA-TO{to}-K{K}.py"
        os.system(f"echo k_shots={K} > {current_file}")
        os.system(f"echo taylor_orders=\({to},0\) >> {current_file}")
        os.system(f"cat base_cavia.py >> {current_file}")
        # os.system(f"python {current_file} > nohup.log")

        print(f"\n\nCurrently running NCF_PAML with taylor_order={to} envs and K={K} shots ...", flush=True)
        current_file = f"NCF_PAML-TO{to}-K{K}.py"
        os.system(f"echo k_shots={K} > {current_file}")
        os.system(f"echo taylor_orders=\({to},0\) >> {current_file}")
        os.system(f"cat base_ncf_paml.py >> {current_file}")
        # os.system(f"python {current_file} > nohup.log")

        print(f"\n\nCurrently running NCF_NOALM with taylor_order={to} envs and K={K} shots ...", flush=True)
        current_file = f"NCF_NOALM-TO{to}-K{K}.py"
        os.system(f"echo k_shots={K} > {current_file}")
        os.system(f"echo taylor_orders=\({to},0\) >> {current_file}")
        os.system(f"cat base_ncf_noalm.py >> {current_file}")
        # os.system(f"python {current_file} > nohup.log")

    #     break
    # break
