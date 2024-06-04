import os
import json

def extract_subfolders(root_dir, layer_info):
    # Extract subfolder names relative to the layer_info directory
    rel_path = os.path.relpath(root_dir, layer_info)
    subfolders = rel_path.split(os.path.sep)
    return subfolders

def process_txt_file(txt_file_path, subfolders):
    # Extract subfolder names
    topocategories, topologies, gridsize, gridconfig = subfolders

    # Read the contents of the text file
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    # Extract information from the text file
    layer_id = int(lines[0].split(': ')[1].strip())
    compute_cycles = float(lines[1].split(': ')[1].strip())
    overall_utilization = float(lines[2].split(': ')[1].strip())
    ifmap_dram_reads = float(lines[3].split(': ')[1].strip())
    filter_dram_reads = float(lines[4].split(': ')[1].strip())
    ofmap_dram_writes = float(lines[5].split(': ')[1].strip())

    # Create the JSON object
    json_data = {
        'topologies': topologies,
        'gridsize': gridsize,
        'gridconfig': gridconfig,
        'layerID': layer_id,
        'cycle': compute_cycles,
        'util': overall_utilization,
        'ifmapread': ifmap_dram_reads,
        'filterread': filter_dram_reads,
        'ofmapwrite': ofmap_dram_writes
    }

    return json_data

def traverse_directories(root_dir, layer_info):
    json_data_list = []

    # Traverse through all directories and subdirectories
    for root, dirs, files in os.walk(root_dir):
        # Extract subfolder names
        subfolders = extract_subfolders(root, layer_info)

        for filename in files:
            # Check if the file is a .txt file
            if filename.endswith('.txt'):
                # Process the text file
                txt_file_path = os.path.join(root, filename)
                json_data = process_txt_file(txt_file_path, subfolders)
                json_data_list.append(json_data)

    return json_data_list

def write_to_json(json_data_list, output_file):
    # Write the combined JSON data into a single file
    with open(output_file, 'w') as file:
        json.dump(json_data_list, file, indent=4)

# Main function
def main():
    layer_info = './ResLayer/topologies'  # Directory provided by you
    output_file = './figures/withoutPE/output.json'  # Output file to store combined JSON data

    # Traverse through directories and process text files
    all_json_data = []
    for root, dirs, files in os.walk(layer_info):
        # Iterate over subdirectories
        for directory in dirs:
            root_dir = os.path.join(layer_info, directory)
            # Process text files in the subdirectory
            json_data_list = traverse_directories(root_dir, layer_info)
            # Combine JSON data from all subdirectories
            all_json_data.extend(json_data_list)

    # Write the combined JSON data into a single file
    write_to_json(all_json_data, output_file)

if __name__ == "__main__":
    main()