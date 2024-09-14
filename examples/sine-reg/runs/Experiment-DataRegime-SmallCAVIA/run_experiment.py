#%%
import os

GRADIENT_UPDATES = [1, 5, 100]          ## GU       ## Change the learning rate 0.1 and go again. (also CAVIA can scale grads, so who cares?)
TAYLOR_ORDERS = [0, 2]                  ## TO       ## Do [1 and 4 later on !]
# NUM_ENVS = [250, 1000, 12500]           ## NE
NUM_ENVS = [12500]           ## NE
CONTEXT_SIZE = [2, 3, 50]               ## CS

csv_file = "results.csv"
os.system(f"echo 'method,num_envs,taylor_order,context_size,gradient_updates,outer_steps,mse_ind,ci_ind,mse_ood,ci_ood' > {csv_file}")

## Run the files in succession. 
for gu, gradient_updates in enumerate(GRADIENT_UPDATES):
    for taylor_orders in TAYLOR_ORDERS:
        for num_envs in NUM_ENVS:
            for context_size in CONTEXT_SIZE:

                ## Run CAVIA
                print(f"\n\nCurrently running DataRegime-CAVIA with N={num_envs} envs, TO={taylor_orders}, CS={context_size}, GU={gradient_updates} ...", flush=True)
                current_file = f"CAVIA-NE{num_envs}-TO{taylor_orders}-CS{context_size}-GU{gradient_updates}.py"
                os.system("echo num_envs=\("+str(num_envs)+", 1000\)" + " > " + current_file)
                os.system("echo taylor_orders=\("+str(taylor_orders) + ", 0\)" + " >> " + current_file)
                os.system(f"echo context_size={context_size} >> {current_file}")
                os.system(f"echo nb_inner_steps={gradient_updates} >> {current_file}")
                os.system(f"cat base_cavia.py >> {current_file}")
                os.system(f"python {current_file} > nohup.log")

                # ## Run NCF which does not require tweaking gradient updates
                # if gu == 0:
                #     print(f"\n\nCurrently running DataRegime-NCF with N={num_envs} envs, TO={taylor_orders}, CS={context_size}, GU={gradient_updates} ...", flush=True)
                #     current_file = f"NCF-NE{num_envs}-TO{taylor_orders}-CS{context_size}.py"
                #     os.system("echo num_envs=\("+str(num_envs)+", 1000\)" + " > " + current_file)
                #     os.system("echo taylor_orders=\("+str(taylor_orders) + ", 0\)" + " >> " + current_file)
                #     os.system(f"echo context_size={context_size} >> {current_file}")
                #     os.system(f"cat base_ncf.py >> {current_file}")
                #     os.system(f"python {current_file} > nohup.log")

                print("\n============ SWITCHING TO NEXT HYPERPARAMETER SET ... ============\n", flush=True)

                # exit(0) # For testing purposes TODO: Remove this line





# ### Some scripts failed to run appropriately !

# methods = ["CAVIA", "NCF"]
# for method in methods:
#     print(f"\n\nCurrently running {method} ...", flush=True)
#     # for root, dirs, files in os.walk(f"examples/sine-reg/runs/Experiment-DataRegime/{method}"):
#     for root, dirs, files in os.walk(f"./{method}"):
#         ## Exclude any folder named "selfmod"
#         if "selfmod" in root:
#             continue

#         ## Now we want to open all folders for both methods, and run the .py file in them
#         # for file in files:
#         #     if file.endswith(".py"):
#         #         print(f"\tRunning {file} in {root} ...", flush=True)
#         #         ## Move to the parent folder and run the file (use the directory name)
#         #         # os.system(f"cd {root} && python {os.path.join(root, file)} > nohup_noshuffle.log")
#         #         os.system(f"cd {root} && python {file} > nohup_noshuffle.log")
#         #         print("\n============ SWITCHING TO NEXT HYPERPARAMETER SET ... ============\n", flush=True)
#         #         # exit(0)


#         ## For everile folder we find, open the results.csv file and appends its last line "results_all.csv"
#         for file in files:
#             if file == "results.csv":
#                 print(f"\tAppending results from {file} in {root} ...", flush=True)
#                 os.system(f"cat {os.path.join(root, file)} | tail -n 1 >> results_all.csv")
#                 print("\n============ SWITCHING TO NEXT HYPERPARAMETER SET ... ============\n", flush=True)
#                 # exit(0)