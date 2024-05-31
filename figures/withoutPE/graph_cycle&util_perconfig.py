import json
import os
import matplotlib.pyplot as plt

# Load JSON data from file
with open('./figures/withoutPE/totaloutput.json', 'r') as file:
    combined_data = json.load(file)

# Group data by topologies and gridsize
grouped_data = {}
for entry in combined_data:
    key = (entry['topologies'], entry['gridsize'])
    if key not in grouped_data:
        grouped_data[key] = {'gridconfigs': [], 'totalcycles': [], 'totalutils': []}
    grouped_data[key]['gridconfigs'].append(entry['gridconfig'])
    grouped_data[key]['totalcycles'].append(entry['totalcycle'])
    grouped_data[key]['totalutils'].append(entry['totalutil'])

# Create plots for each unique combination of topologies and gridsize
for key, data in grouped_data.items():
    topologies, gridsize = key
    gridconfigs = data['gridconfigs']
    totalcycles = data['totalcycles']
    totalutils = data['totalutils']

    # Sort gridconfigs and corresponding totalutils
    sorted_indices = sorted(range(len(gridconfigs)), key=lambda k: gridconfigs[k])
    gridconfigs = [gridconfigs[i] for i in sorted_indices]
    totalutils = [totalutils[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))

    # Plot bar graph for total cycle
    plt.bar(gridconfigs, totalcycles, color='blue', label='Total Cycle')

    # Create a secondary y-axis for total utilization
    ax = plt.gca()
    ax2 = ax.twinx()

    # Plot line graph for total utilization
    ax2.plot(gridconfigs, totalutils, marker='o', color='red', label='Total Utilization')

    # Set y-axis limits for total utilization (0 to 100)
    ax2.set_ylim(0, 100)

    plt.xlabel('Grid Config')
    plt.ylabel('Total Cycle')
    ax2.set_ylabel('Total Utilization (%)')
    plt.title(f'Total Cycle and Utilization for {topologies} ({gridsize})')
    plt.xticks(rotation=45, ha='right')

    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()

    # Ensure the directory exists before saving the plot
    save_dir = os.path.join('.', 'figures', 'withoutPE', topologies, gridsize)
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{topologies}_{gridsize}_cycle_util.png'))

    # Close the plot to release memory
    plt.close()

print("Combined bar and line plots saved successfully.")