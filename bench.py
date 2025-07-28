import subprocess

scripts = ["ladder.py", "merge.py", "quick.py", "tim.py"]

for script in scripts:
    print(f"\nRunning {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
