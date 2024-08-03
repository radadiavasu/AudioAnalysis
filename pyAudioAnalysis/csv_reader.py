import sys
import subprocess

def install_module(module_name):
    """
    Function to install a module using pip
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
    
def import_or_install(module_name):
    """
    Function to try to import a module, and if not found, prompt the user to install it
    """
    try:
        __import__(module_name)
        print(f"Module '{module_name}' is already installed.")
    except ImportError:
        user_input = input(f"Module '{module_name}' not found. Do you want to install it? (Y/n): ").strip().lower()
        if user_input in ('y', 'yes', ''):
            install_module(module_name)
            print(f"Module '{module_name}' installed successfully.")
        else:
            print(f"Module '{module_name}' not installed. Exiting.")
            sys.exit(1)


try:
    import_or_install('csv')
    import csv
    def read_csv_file(csv_path):
        start_times = []
        end_times = []
        labels = []
        with open(csv_path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            next(csv_reader)  # Skip header
            for row in csv_reader:
                if len(row) == 4:
                    start_times.append(float(row[1]))
                    end_times.append(float(row[2]))
                    labels.append(row[3])
                else:
                    if not start_times and end_times:
                        raise ValueError("No valid data found in the csv file.")
        return start_times, end_times, labels
except ModuleNotFoundError:
    print("Module has not been installed properly. Please prefer only `pypi` modules.")
# Example usage:
start_times, end_times, labels = read_csv_file('G:/pyAudioAnalysis/pyAudioAnalysis/data/ground_truth.csv')
print("Start times:", start_times)
print("End times:", end_times)
print("Labels:", labels)

