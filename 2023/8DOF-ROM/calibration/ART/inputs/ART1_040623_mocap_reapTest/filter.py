import sys
sub_folder = "r" + sys.argv[1]
file_ = sub_folder + "_r" +  sys.argv[2]


path = './' + sub_folder + '/' + file_ + '.txt'

# Open the file for reading
with open(path, 'r') as f:
    # Read all the lines in the file and store them in a list
    lines = f.readlines()

# Create a new list to store the filtered lines
filtered_lines = []

# Loop over the lines and check if the first column changes
previous_value = None
for line in lines:
    # Split the line into fields
    fields = line.split()
    
    # Check if the first field is different from the previous value
    current_value = float(fields[0])
    if previous_value is None or current_value != previous_value:
        # Add the line to the filtered lines list
        filtered_lines.append(line)
        
    # Store the current value as the previous value for the next iteration
    previous_value = current_value

# Open the file for writing and write the filtered lines
with open(path, 'w') as f:
    f.writelines(filtered_lines)