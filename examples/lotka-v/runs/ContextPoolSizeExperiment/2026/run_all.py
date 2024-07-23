import os

files = [f"{i}.py" for i in [0, 1, 2, 4, 6, 8]]

## Run the files in succession. One must finish before the other starts
for file in files:
    os.system(f"python {file} > nohup.log")
