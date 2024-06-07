import json
import matplotlib.pyplot as plt
import os

def create_bar_plot(json_file, output_dir):
    # Load JSON data from the big JSON file
    with open(json_file, 'r') as file:
        json_data = json.load(file)

    # Dictionary to store utilization data by gridconfig and topologies
    gridconfig_data = {}

    # Organize utilization data by gridconfig and topologies
    for entry in json_data:
        gridconfig = entry['gridconfig']
        topologies = entry['topologies'].split('.')[0]  # Remove .csv extension
        util = entry['totalutil']
        if gridconfig not in gridconfig_data:
            gridconfig_data[gridconfig] = {}
        if topologies not in gridconfig_data[gridconfig]:
            gridconfig_data[gridconfig][topologies] = []
        gridconfig_data[gridconfig][topologies].append(util)

    # Create a bar plot for each gridconfig
    for gridconfig, data in gridconfig_data.items():
        fig, ax = plt.subplots()
        topologies = list(data.keys())
        util_values = [value for sublist in [data[topology] for topology in topologies] for value in sublist]
        ax.bar(topologies, util_values, color='skyblue')
        ax.set_xlabel('Topologies')
        ax.set_ylabel('Utilization')
        ax.set_title(f'Utilization Comparison for Gridconfig: {gridconfig}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save the plot as a PNG file in the output directory
        output_file = os.path.join(output_dir, f'util_comparison_{gridconfig}.png')
        plt.savefig(output_file)
        plt.close()

        print(f"Bar plot saved as '{output_file}'")

# Specify the path to the big JSON file
big_json_file = './figures/withoutPE/totaloutput.json'  # Update with the path to your big JSON file

# Specify the output directory
output_dir = './figures/withoutPE/utilCompare'  # Update with your desired output directory

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create the bar plots
create_bar_plot(big_json_file, output_dir)
