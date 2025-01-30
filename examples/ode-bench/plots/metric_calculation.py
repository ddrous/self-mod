#%%


#%%[markdown]
## 10 Families of ODEs, all with 5 environments

## A script to parse and calculate the mean OOD performance of a set of runs
# Look for lines line this: "Loss per OoD environment: [0.022641170769929886]"

import os

def get_mean_ood_performance(run_dir):
    log_file = os.path.join(run_dir, "nohup.log")
    with open(log_file, "r") as f:
        lines = f.readlines()
    ood_losses = []
    for line in lines:
        if "Loss per OoD environment" in line and "," not in line:
            loss = float(line.split("[")[1].split("]")[0])
            ood_losses.append(loss)
    mean_ood = sum(ood_losses) / len(ood_losses)
    return mean_ood

get_mean_ood_performance("../")

folders_ncf = ["_02_NCFRuns/250125-190838-10Fams-1Expert/", "_02_NCFRuns/250126-001748-10Fams-10Expts-GD/", "_02_NCFRuns/250126-090106-10Fams-20Expts/"]
folders_coda = ["_03_CoDARuns/250125-212602-10Fams-1Expert/", "_03_CoDARuns/250126-015635-10Fams-10Expts-GD/", "_03_CoDARuns/250126-162200-10Fams-20Experts/"]
folders_geps = ["_04_GEPSRuns/250125-225903-10Fams-1Expt/", "_04_GEPSRuns/250126-044755-10Fams-10Expts-GD/", "_04_GEPSRuns/250126-213733-10Fams-20Expts/"]

folders = folders_ncf + folders_coda + folders_geps
for folder in folders:
    print(f"Mean OOD performance for {folder} is \t {get_mean_ood_performance('../runs/'+folder)}")

print()
folders_10_experts = ["_02_NCFRuns/250122-041154-10Fams-T0*/", "_03_CoDARuns/250121-165622-10Fams-T0*/", "_04_GEPSRuns/250124-003842-10Fams-T0*/"]
for folder in folders_10_experts:
    print(f"Mean OOD performance for {folder} is \t {get_mean_ood_performance('../runs/'+folder)}")


#%%
## Ind Metrics are all in a line like this : "Loss InD (mean)  : 0.119"

def get_ind_performance(run_dir):
    log_file = os.path.join(run_dir, "nohup.log")
    with open(log_file, "r") as f:
        lines = f.readlines()
    ind_losses = []
    for line in lines:
        if "Loss InD (mean)" in line:
            loss = float(line.split(":")[1])
            ind_losses.append(loss)
    return ind_losses

for folder in folders:
    print(f"Ind performance for {folder} is \t {get_ind_performance('../runs/'+folder)}")

print()
for folder in folders_10_experts:
    print(f"Ind performance for {folder} is \t {get_ind_performance('../runs/'+folder)}")




#%%

## Now we want the percentage ot the OoD loses that is below a certain threshold
def get_ood_thresholded_perf(run_dir, threshold):
    log_file = os.path.join(run_dir, "nohup.log")
    with open(log_file, "r") as f:
        lines = f.readlines()
    ood_losses = []
    for line in lines:
        if "Loss per OoD environment" in line and "," not in line:
            loss = float(line.split("[")[1].split("]")[0])
            ood_losses.append(loss)
    below_threshold = [loss for loss in ood_losses if loss < threshold]
    return len(below_threshold) / len(ood_losses)

threshold = 0.1
print("Caclulating the percentage of OOD losses below ", threshold)
all_folders = folders + folders_10_experts
for folder in all_folders:
    print(f"    {folder}: \t\t {get_ood_thresholded_perf('../runs/'+folder, threshold)*100:0.1f}%")

#%%
## Do the same for the Ind losses: A line is like this: "Losses per InD environment: [0.04892430827021599, 0.0594823956489563, 0. ... ]" there are 160 values before the closing bracket

def get_ind_thresholded_perf(run_dir, threshold):
    log_file = os.path.join(run_dir, "nohup.log")
    with open(log_file, "r") as f:
        lines = f.readlines()
    ind_losses = []
    for line in lines:
        if "Losses per InD environment" in line:
            losses = line.split("[")[1].split("]")[0]
            losses = [float(loss) for loss in losses.split(",")]
            ind_losses.extend(losses)
    below_threshold = [loss for loss in ind_losses if loss < threshold]
    return len(below_threshold) / len(ind_losses)

threshold = 0.1
print("Caclulating the percentage of Ind losses below ", threshold)
all_folders = folders + folders_10_experts
for folder in all_folders:
    print(f"    {folder}: \t\t {get_ind_thresholded_perf('../runs/'+folder, threshold)*100:0.1f}%")



#%%[markdown]
## 10 Families of ODEs, all with 5 environments

#%%
### Now let's print the InD relative MSE
folders_ncf = ["_02_NCFRuns/250129-084602-5Envs-1Expert/", "_02_NCFRuns/250129-040905-5Envs-10Experts/", "_02_NCFRuns/250130-091142-5Envs-10Experts-GD/"]
folders_coda = ["_03_CoDARuns/250128-224851-5Envs-1Expert/", "_03_CoDARuns/250129-000048-5Envs-10Experts/", "_03_CoDARuns/250130-095818-5Envs-10Experts-GD/"]
folders_geps = ["_04_GEPSRuns/250128-232913-5Envs-1Expert/", "_04_GEPSRuns/250129-013232-5Envs-10Experts/", "_04_GEPSRuns/250130-091444-5Envs-10Experts-GD/"]

print("Relative MSE for InD")
for folder in folders_ncf + folders_coda + folders_geps:
    print(f"Ind performance for {folder} is \t {get_ind_performance('../runs/'+folder)}")

print("Relative MSE for OoD")
for folder in folders_ncf + folders_coda + folders_geps:
    print(f"Mean OOD performance for {folder} is \t {get_mean_ood_performance('../runs/'+folder)}")

# #%%
# print("Percentage success for InD")

# threshold = 0.1
# print("Caclulating the percentage of Ind losses below ", threshold)
# all_folders = folders_ncf + folders_coda + folders_geps
# for folder in all_folders:
#     print(f"    {folder}: \t\t {get_ind_thresholded_perf('../runs/'+folder, threshold)*100:0.1f}%")

# print("Percentage success for OoD")
# threshold = 0.1
# print("Caclulating the percentage of OOD losses below ", threshold)
# all_folders = folders_ncf + folders_coda + folders_geps
# for folder in all_folders:
#     print(f"    {folder}: \t\t {get_ood_thresholded_perf('../runs/'+folder, threshold)*100:0.1f}%")
