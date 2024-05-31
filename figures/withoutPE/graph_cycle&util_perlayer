import json
import matplotlib.pyplot as plt
import os

def create_bar_and_line_plot(json_data, output_dir):
    # Dictionary to store unique combinations of 'topologies', 'gridsize', and 'gridconfig'
    unique_combinations = {}

    # Group JSON data by unique combinations of 'topologies', 'gridsize', and 'gridconfig'
    for entry in json_data:
        key = (entry['topologies'], entry['gridsize'], entry['gridconfig'])
        if key not in unique_combinations:
            unique_combinations[key] = []
        unique_combinations[key].append(entry)

    # Create a bar and line plot for each unique combination
    for key, data_list in unique_combinations.items():
        topologies, gridsize, gridconfig = key

        # Create subfolders for each 'topologies' and 'gridsize' combination
        output_subdir = os.path.join(output_dir, topologies, gridsize)
        os.makedirs(output_subdir, exist_ok=True)

        # Extract layerID, cycle, and util values
        layer_ids = [entry['layerID'] for entry in data_list]
        cycles = [entry['cycle'] for entry in data_list]
        utils = [entry['util'] for entry in data_list]

        # Create bar plot
        plt.bar(layer_ids, cycles, color='skyblue', label='Cycle')
        plt.xlabel('Layer ID')
        plt.ylabel('Cycle')

        # Create line plot for 'util'
        plt.twinx()  # Create a second y-axis
        plt.plot(layer_ids, utils, color='orange', label='Utilization')
        plt.ylabel('Utilization')

        plt.title(f'Cycle and Utilization vs. Layer ID ({topologies}, {gridsize}, {gridconfig})')

        # Add legend
        plt.legend(loc='upper left')

        # Save the plot as a PNG file in the subfolder
        output_file = os.path.join(output_subdir, f'bar_and_line_plot_{topologies}_{gridsize}_{gridconfig}.png')
        plt.savefig(output_file)
        plt.close()

        print(f"Bar and line plot saved as '{output_file}'")

# Load JSON data from file
with open('./figures/withoutPE/output.json', 'r') as file:
    json_data = json.load(file)

# Specify output directory
output_dir = './figures/withoutPE'  # Base directory to save the plots

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create bar and line plots
create_bar_and_line_plot(json_data, output_dir)
