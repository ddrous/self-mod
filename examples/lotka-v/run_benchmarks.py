#%%

## Run the scripts in the runs folder
import os

# Get the current directory
scripts_names = ["main_mixer_3X_A.py", "main_mixer_3X_B.py", "main_mixer_1X_A.py", "main_mixer_1X_B.py"]
directories = ["runs/_Benchmark_NCF", "runs/_Benchmark_CoDA", "runs/_Benchmark_GEPS"]

# print("\n\n============ RUNNING HIGH-DIMENSIONAL CASES ... ============\n", flush=True)
# for d in directories:
#     for s in scripts_names:
#         # Run the script
#         print(f"Running {s} in {d} ...", flush=True)
#         os.system(f"cd {d} && python {s} > nohup.log")
#         print("\n============ SWITCHING TO NEXT HYPERPARAMETER SET ... ============\n", flush=True)
#         # exit(0)


## Run two late scripts for interpretability
print("\n\n============ RUNNING LOW-DIMENSIONAL NCF CASES ... ============\n", flush=True)
low_dim_scripts = ["main_mixer_3X_A_2D.py", "main_mixer_3X_B_2D.py"]
for s in low_dim_scripts:
    # Run the script
    print(f"Running {s} in {directories[0]} ...", flush=True)
    os.system(f"cd {directories[0]} && python {s} > nohup.log")
    print("\n============ SWITCHING TO NEXT HYPERPARAMETER SET ... ============\n", flush=True)
