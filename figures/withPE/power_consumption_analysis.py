import os
import matplotlib.pyplot as plt

def compute_power_consumption(txt_file_path, comp_energy_factor, dram_read_energy_factor, dram_write_energy_factor):
    comp_energy = 0
    dram_read_energy = 0
    dram_write_energy = 0
    #need to add sram energy

    with open(txt_file_path, 'r') as file:
        for line in file:
            if line.startswith('Compute Cycles: '):
                comp_energy += int(float(line.split('Compute Cycles: ')[1].strip())) * comp_energy_factor
            elif line.startswith('IFMAP DRAM Reads: ') or line.startswith('Filter DRAM Reads: '):
                dram_read_energy += int(float(line.split(': ')[1].strip())) * dram_read_energy_factor
            elif line.startswith('OFMAP DRAM Writes: '):
                dram_write_energy += int(float(line.split('OFMAP DRAM Writes: ')[1].strip())) * dram_write_energy_factor

    total_power_consumption = comp_energy + dram_read_energy + dram_write_energy
    return total_power_consumption

def create_bar_plot(txt_files, labels, comp_energy_factor, dram_read_energy_factor, dram_write_energy_factor, save_directory):
    power_consumptions = []

    for txt_file in txt_files:
        power_consumption = compute_power_consumption(txt_file, comp_energy_factor, dram_read_energy_factor, dram_write_energy_factor)
        power_consumptions.append(power_consumption)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, power_consumptions, color='skyblue')
    plt.xlabel('Labels')
    plt.ylabel('Total Power Consumption')
    plt.title('Power Consumption Analysis')

    # Save the plot as a PNG file in the specified directory
    save_path = os.path.join(save_directory, 'power_consumption_analysis.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Bar plot saved to {save_path}")

def main():
    # Define the file paths and labels
    txt_files = [
        './ResSimulation/topologies/conv_nets/alexnet.csv/0064/0001&0064/LayerBasedInfo/merged_Layer00.txt',
        './ResSimulation/topologies/conv_nets/Googlenet.csv/0064/0001&0064/LayerBasedInfo/merged_Layer01.txt'
    ]
    labels = ['Alexnet 1&64 Layer 1', 'Googlenet 1&64 Layer 2']

    # Define the energy factors
    comp_energy_factor = 0  # Example factor
    dram_read_energy_factor = 150 * 32  # Example factor
    dram_write_energy_factor = 150 * 32  # Example factor

    # Define the save directory
    save_directory = './figures/withPE'

    # Ensure that the number of labels matches the number of txt files
    if len(labels) != len(txt_files):
        print("The number of labels must match the number of txt files.")
        return

    # Create the bar plot
    create_bar_plot(txt_files, labels, comp_energy_factor, dram_read_energy_factor, dram_write_energy_factor, save_directory)

if __name__ == "__main__":
    main()
