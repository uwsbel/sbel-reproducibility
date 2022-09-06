import csv
import matplotlib.pyplot as plt
import platform

# check operating system, file loc different
if platform.system() == 'Darwin':
    fileloc = "/Users/luning/Manuscripts/Conference/2022/IROS_Half_Implicit/images/"
if platform.system() == 'Linux':
    fileloc = "/home/luning/Manuscripts/Journal/2022/HalfImplicit/images/"

filename = "N_pendulum_scaling_updated.png"
# flag for saving pic directly in image folder
saveFig = True


Fontsize = 80
LineWidth = 10
MarkerSize = 40
plt.rc('font', size=Fontsize)
plt.rc('legend', fontsize=Fontsize*1.3)
plt.rc('figure', titlesize=Fontsize*1.3)
plt.rc('lines', linewidth=LineWidth)
plt.rc('lines', markersize=MarkerSize)
plt.rc('axes', linewidth=LineWidth*0.5)

file = open("data/rA_timing_cmp_sparse.csv")
csvreader = csv.reader(file)
forms = next(csvreader)[1:]
tol = next(csvreader)[1:]

tol_list = []
for tolerance in tol:
    tol_list.append(float(tolerance))

rows = []
nb_list = []
for row in csvreader:
    nb_list.append(float(row[0]))
    val = []
    for num in row:
        val.append(float(num))
    rows.append(val[1:])
    
file.close()

fig, ax = plt.subplots(figsize=(30, 20))


line_styles = ["solid", "solid"]
markers = ['o', '^']
color = ['red', 'blue']
for ii in range(0, len(forms)):
    form = forms[ii]
    tolerance = tol_list[ii]
    time = []
    for jj in range(0, len(nb_list)):
        time.append(rows[jj][ii])
    
    linestyle = line_styles[ii%4]    
    print(form)

    my_label = ''
    if "half" in form:
        color = "blue"
        my_label = "HI"
    else:
        color = "red"
        my_label = "FI"
        
    
    ax.plot(nb_list, time, linestyle = linestyle, color=color, marker=markers[ii%4], label=my_label)

ax.set(xlabel='number of bars', ylabel='CPU time (sec)')
ax.legend(loc='upper left')
ax.grid(linestyle='--', linewidth=0.5*LineWidth)

ax.set_xlim([0, 34])
ax.set_ylim([0, 100])

if saveFig == True:
    plt.savefig(fileloc + filename)