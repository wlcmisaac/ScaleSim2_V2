import json

# Load JSON data from file
with open('./figures/withoutPE/output.json', 'r') as file:
    output_data = json.load(file)

# Dictionary to store aggregated data
aggregated_data = {}

# Iterate through JSON data and aggregate values
for entry in output_data:
    key = (entry['topologies'], entry['gridsize'], entry['gridconfig'])
    if key not in aggregated_data:
        aggregated_data[key] = {
            'totalcycle': 0,
            'totalutil': 0,
            'ifmapread': 0,
            'filterread': 0,
            'ofmapwrite': 0
        }
    aggregated_data[key]['totalcycle'] += entry['cycle']
    aggregated_data[key]['ifmapread'] += entry['ifmapread']
    aggregated_data[key]['filterread'] += entry['filterread']
    aggregated_data[key]['ofmapwrite'] += entry['ofmapwrite']

# Calculate average utilization after summing all cycles
for key, values in aggregated_data.items():
    total_util = 0
    for entry in output_data:
        if (entry['topologies'], entry['gridsize'], entry['gridconfig']) == key:
            total_util += entry['util'] * (entry['cycle'] / values['totalcycle'])
    aggregated_data[key]['totalutil'] = total_util

# Create a list of dictionaries for the aggregated data
total_output = [{'topologies': k[0], 'gridsize': k[1], 'gridconfig': k[2], **v} for k, v in aggregated_data.items()]

# Save aggregated data to a new JSON file
with open('./figures/withoutPE/totaloutput.json', 'w') as file:
    json.dump(total_output, file, indent=4)

print("Aggregated data saved to 'totaloutput.json'")
