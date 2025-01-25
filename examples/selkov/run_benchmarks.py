#%%

## Run the scripts in the runs folder
import os

# Get the current directory
scripts_names = ["main_mixer_3X_A.py", "main_mixer_3X_B.py", "main_mixer_1X_A.py", "main_mixer_1X_B.py"]
directories = ["runs/_Benchmark_NCF", "runs/_Benchmark_CoDA", "runs/_Benchmark_GEPS"]

print("\n\n============ RUNNING HIGH-DIMENSIONAL CASES ... ============\n", flush=True)
for d in directories:
    for s in scripts_names:
        # Run the script
        print(f"Running {s} in {d} ...", flush=True)
        os.system(f"cd {d} && python {s} > nohup.log")
        print("\n============ SWITCHING TO NEXT HYPERPARAMETER SET ... ============\n", flush=True)
        # exit(0)

