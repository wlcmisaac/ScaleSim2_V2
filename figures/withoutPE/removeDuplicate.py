import json

def remove_duplicates(json_data):
    unique_data = []
    seen = set()

    for item in json_data:
        # Convert the dictionary to a hashable string
        item_str = json.dumps(item, sort_keys=True)
        # Check if the item has been seen before
        if item_str not in seen:
            # If not, add it to the unique data list
            unique_data.append(item)
            # Add the item to the set of seen items
            seen.add(item_str)

    return unique_data

# Load JSON data from file
with open('./figures/withoutPE/output.json', 'r') as file:
    json_data = json.load(file)

# Remove duplicates
unique_json_data = remove_duplicates(json_data)

# Write the unique JSON data back to file
with open('./figures/withoutPE/output.json', 'w') as file:
    json.dump(unique_json_data, file, indent=4)

print("Duplicates removed and saved to 'output.json'")