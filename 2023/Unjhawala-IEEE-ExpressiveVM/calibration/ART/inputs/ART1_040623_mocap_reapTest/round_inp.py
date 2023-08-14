import sys
sub_folder = "r" + sys.argv[1]
file_ = sub_folder + "_r" +  sys.argv[2]

path = './' + sub_folder + '/' + file_ + '.txt'

# Open the input file for reading
with open(path, 'r') as f:
    # Read all the lines in the file and store them in a list
    lines = f.readlines()

# Open the input file for writing (this will overwrite the original file)
with open(path, 'w') as f:
    # Loop over each line in the list
    for line in lines:
        # Split the line into columns using whitespace as the delimiter
        columns = line.split()
        
        # Round the first column to one decimal place
        rounded_number = round(float(columns[0]), 1)
        
        # Write the rounded number and the rest of the columns to the file
        f.write(f'{rounded_number:.1f} {" ".join(columns[1:])}\n')
