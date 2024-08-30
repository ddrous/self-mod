#%%
import os

# Ns = [6, 12, 60, 252, 1020]
# Ks = [10, 50, 100, 500, 1000]

Ns = [1020]
Ks = [500, 1000]

## Run the files in succession. 
for N in Ns:
    for K in Ks:
        # if K != 500:
        print(f"\n\nCurrently running CAVIA with N={N} envs and K={K} shots ...", flush=True)
        current_file = f"CAVIA-N{N}-K{K}.py"
        os.system(f"echo k_shots={K} > {current_file}")
        os.system(f"echo envs_batch_size={N} >> {current_file}")
        os.system(f"cat base_cavia.py >> {current_file}")
        os.system(f"python {current_file} > nohup.log")

        print(f"\n\nCurrently running NCF-t2 with N={N} envs and K={K} shots ...", flush=True)
        current_file = f"NCF-N{N}-K{K}.py"
        os.system(f"echo k_shots={K} > {current_file}")
        os.system(f"echo envs_batch_size={N} >> {current_file}")
        os.system(f"cat base_ncf.py >> {current_file}")
        os.system(f"python {current_file} > nohup.log")

    #     break
    # break

