# process the data and generate a clean dataset in a format similar to k-Means.
# Here’s how data was processed for k-Means clustering:

import json

# Load the JSON file
file_path = "cleanData2.json"

with open(file_path, "r") as file:
    data = json.load(file)

# Display a sample of the data to understand its structure
data[:5] if isinstance(data, list) else data


print('data[:5] ==>> ', data[:5])