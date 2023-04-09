import sys


sub_folder = "r" + sys.argv[1]
file_ = sub_folder + "_r" +  sys.argv[2]

path = './' + sub_folder + '/' + file_ + '.txt'

# Read the contents of the input file
with open(path, 'r') as file:
    contents = file.read()

# Split the contents into rows
rows = contents.split('\n')

# Initialize an empty string to hold the output
output = ''

# Loop over the rows
for i in range(len(rows)-1):
    # Add the current row to the output
    output += rows[i] + '\n'
    
    # Split the current row into columns
    cols = rows[i].split()

    
    # Step size we want is 
    step = 0.001
    # Loop over the range 1 to 100
    for j in range(1, 100):
        # Calculate the new value for the first column
        new_val = float(cols[0]) + step * j
        
        # Format the new row and add it to the output
        new_row = f'{new_val:.6f} {cols[1]} {cols[2]} {cols[3]}'
        # new_row = f'{new_val:.6f} {cols[1]} {0} {cols[3]}'

        output += new_row + '\n'

# Add the last row to the output
output += rows[-1]

with open(path, 'w') as file:
    file.write(output)
