
# List of ramp files we want to run
ramp_files=("test0" "test1" "test2" "test3" "test4")
# List of full throttle files
full_files=("test1" "test2" "test3" "test5" "test8") 

# Loop through these files and launch our python script with the right arguments
for (( i = 1; i <= $#ramp_files; i++ )) do
    python3 ART_acc_calib.py 1000 smc $ramp_files[i] $full_files[i]
done

