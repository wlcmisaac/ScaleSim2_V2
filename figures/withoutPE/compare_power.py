import os
import json
import matplotlib.pyplot as plt
import numpy as np

def calculate_energy_consumption(data, calculation_factor, dramRW_factor, sramRW_factor):
    totalcycle_energy = data['totalcycle'] * calculation_factor
    ifmapread_dram_energy = data['ifmapread'] * dramRW_factor
    ifmapread_sram_energy = data['ifmapread'] * sramRW_factor
    filterread_dram_energy = data['filterread'] * dramRW_factor
    filterread_sram_energy = data['filterread'] * sramRW_factor
    ofmapwrite_dram_energy = data['ofmapwrite'] * dramRW_factor
    ofmapwrite_sram_energy = data['ofmapwrite'] * sramRW_factor

    total_energy = (
        totalcycle_energy +
        ifmapread_dram_energy +
        ifmapread_sram_energy +
        filterread_dram_energy +
        filterread_sram_energy +
        ofmapwrite_dram_energy +
        ofmapwrite_sram_energy
    )
    return total_energy

def create_topology_plots(aggregated_data, save_directory):
    for topology, topology_data in aggregated_data.items():
        if not topology_data:
            continue
        
        grid_configs = list(topology_data.keys())
        num_configs = len(grid_configs)
        
        bar_width = 0.2
        index = np.arange(num_configs)
        opacity = 0.8
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        plt.figure(figsize=(12, 8))

        # Plotting energy consumption for different grid configurations
        for i, (gridconfig, total_energy) in enumerate(topology_data.items()):
            plt.bar(index[i], total_energy, bar_width,
                    alpha=opacity,
                    color=colors[i % len(colors)],
                    label=f'Grid Config {gridconfig}')

        plt.xlabel('Grid Configurations')
        plt.ylabel('Total Energy Consumption')
        plt.title(f'Energy Consumption Analysis for {topology}')
        plt.xticks(index, grid_configs)
        plt.legend()

        save_path = os.path.join(save_directory, f'energy_consumption_{topology}.png')
        plt.savefig(save_path)
        plt.close()

        print(f"Plot for {topology} saved to {save_path}")

def main():
    # Define the directories containing JSON files
    directories = ['./figures/withoutPE/totaloutput.json']  # Update with your actual directory or list of directories

    # Define the energy factors
    calculation_factor = 0  # Example factor
    dramRW_factor = 150 * 32  # Example factor
    sramRW_factor = 15 * 32  # Example factor

    # Define the save directory for plots
    save_directory = './figures/withoutPE/energyCompare'  # Update this path to the directory where you want to save the plots

    # Filter criteria
    valid_topologies = ['Googlenet.csv', 'mobilenet.csv', 'AlphaGoZero.csv', 'Resnet18.csv']
    valid_gridsize = '0064'

    # Aggregate data by topology and gridconfig
    aggregated_data = {topology: {} for topology in valid_topologies}

    try:
        # Read and process each JSON file
        for directory in directories:
            with open(directory, 'r') as f:
                data = json.load(f)
                for entry in data:
                    if entry['topologies'] in valid_topologies and entry['gridsize'] == valid_gridsize:
                        total_energy = calculate_energy_consumption(
                            entry,
                            calculation_factor,
                            dramRW_factor,
                            sramRW_factor
                        )
                        if entry['topologies'] not in aggregated_data:
                            aggregated_data[entry['topologies']] = {}
                        if entry['gridconfig'] not in aggregated_data[entry['topologies']]:
                            aggregated_data[entry['topologies']][entry['gridconfig']] = 0
                        aggregated_data[entry['topologies']][entry['gridconfig']] += total_energy

        # Create individual plots for each topology
        create_topology_plots(aggregated_data, save_directory)

    except FileNotFoundError:
        print(f"Error: File not found at {directories}")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON file at {directories}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
