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

# Example usage:
start_times, end_times, labels = read_csv_file('G:/pyAudioAnalysis/pyAudioAnalysis/data/ground_truth.csv')
print("Start times:", start_times)
print("End times:", end_times)
print("Labels:", labels)

