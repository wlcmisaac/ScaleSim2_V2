import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load JSON data from file
with open('./figures/withoutPE/totaloutput.json', 'r') as file:
    combined_data = json.load(file)

# Group data by topologies and gridsize
grouped_data = {}
for entry in combined_data:
    key = (entry['topologies'], entry['gridsize'])
    if key not in grouped_data:
        grouped_data[key] = {'gridconfigs': [], 'ifmapreads': [], 'filterreads': [], 'ofmapwrites': []}
    grouped_data[key]['gridconfigs'].append(entry['gridconfig'])
    grouped_data[key]['ifmapreads'].append(entry['ifmapread'])
    grouped_data[key]['filterreads'].append(entry['filterread'])
    grouped_data[key]['ofmapwrites'].append(entry['ofmapwrite'])

# Create stacked bar plots for each unique combination of topologies and gridsize
for key, data in grouped_data.items():
    topologies, gridsize = key
    gridconfigs = data['gridconfigs']
    ifmapreads = data['ifmapreads']
    filterreads = data['filterreads']
    ofmapwrites = data['ofmapwrites']

    # Sort gridconfigs
    sorted_indices = sorted(range(len(gridconfigs)), key=lambda k: gridconfigs[k])
    gridconfigs = [gridconfigs[i] for i in sorted_indices]
    ifmapreads = [ifmapreads[i] for i in sorted_indices]
    filterreads = [filterreads[i] for i in sorted_indices]
    ofmapwrites = [ofmapwrites[i] for i in sorted_indices]

    # Set the width of the bars
    bar_width = 0.35

    # Set position of bars on x-axis
    r = np.arange(len(gridconfigs))

    plt.figure(figsize=(10, 6))

    # Create stacked bars
    plt.bar(r, ifmapreads, color='b', edgecolor='white', width=bar_width, label='ifmapread')
    plt.bar(r, filterreads, color='g', edgecolor='white', width=bar_width, bottom=ifmapreads, label='filterread')
    plt.bar(r, ofmapwrites, color='r', edgecolor='white', width=bar_width, bottom=np.array(ifmapreads)+np.array(filterreads), label='ofmapwrite')

    # Add xticks on the middle of the group bars
    plt.xlabel('Grid Config', fontweight='bold')
    plt.ylabel('Values', fontweight='bold')
    plt.xticks(r, gridconfigs, rotation=45, ha='right')
    plt.title(f'Stacked Bar Plot for {topologies} ({gridsize})')
    plt.legend()

    # Ensure the directory exists before saving the plot
    save_dir = os.path.join('.', 'figures', 'withoutPE', topologies, gridsize)
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'{topologies}_{gridsize}_stacked_barplot.png'))

    # Close the plot to release memory
    plt.close()

print("Stacked bar plots saved successfully.")