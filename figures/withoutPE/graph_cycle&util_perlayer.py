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

        # Extract layerID, cycle, and util values and sort them by layerID
        data_list.sort(key=lambda x: x['layerID'])
        layer_ids = [entry['layerID'] for entry in data_list]
        cycles = [entry['cycle'] for entry in data_list]
        utils = [entry['util'] for entry in data_list]

        fig, ax1 = plt.subplots()

        # Create bar plot for cycles
        color = 'tab:blue'
        ax1.set_xlabel('Layer ID')
        ax1.set_ylabel('Cycle', color=color)
        ax1.bar(layer_ids, cycles, color=color, label='Cycle')
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis for utilization
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Utilization (%)', color=color)
        ax2.plot(layer_ids, utils, color=color, marker='o', label='Utilization')  # Adding markers for better visibility
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 100)  # Assuming utilization is a percentage

        fig.tight_layout()  # To ensure there is no overlap

        plt.title(f'Cycle and Utilization vs. Layer ID ({topologies}, {gridsize}, {gridconfig})')

        # Add legend
        fig.legend(loc='upper left')

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