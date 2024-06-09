import subprocess
import os

def run_script(script_path):
    try:
        result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        print(f"Output of {script_path}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}:\n{e.stderr}")

def main():
    # List of Python scripts to run in sequence
    scripts = [
        'createJSON.py',
        'removeDuplicate.py',
        'combineJSON.py',
        'graph_cycle&util_perlayer.py',
        'graph_cycle&util_perconfig.py',
        'graph_readwrite.py',
        'compare_util.py'
    ]

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Run each script in the list
    for script in scripts:
        script_path = os.path.join(script_dir, script)
        run_script(script_path)

if __name__ == "__main__":
    main()