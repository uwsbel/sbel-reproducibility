import csv
import matplotlib.pyplot as plt
import platform

# check operating system, file loc different
if platform.system() == 'Darwin':
    fileloc = "/Users/luning/Manuscripts/Conference/2022/IROS_Half_Implicit/images/"

filename = "N_pendulum_scaling.png"
# flag for saving pic directly in image folder
saveFig = True


Fontsize = 60
LineWidth = 10
MarkerSize = 40
plt.rc('font', size=Fontsize)
plt.rc('legend', fontsize=Fontsize*0.9)
plt.rc('figure', titlesize=Fontsize*0.9)
plt.rc('lines', linewidth=LineWidth)
plt.rc('lines', markersize=MarkerSize)
plt.rc('axes', linewidth=LineWidth*0.5)

file = open("data/scaling_iros.csv")
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

fig, ax = plt.subplots(figsize=(30, 15))


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

    if "half" in form:
        color = "blue"
    else:
        color = "red"
    ax.plot(nb_list, time, linestyle = linestyle, color=color, marker=markers[ii%4], label="{} implicit".format(form))

ax.set(xlabel='number of bars', ylabel='CPU time (sec)')
ax.legend(loc='upper left')
ax.grid(linestyle='--', linewidth=0.5*LineWidth)

ax.set_xlim([0, 34])
ax.set_ylim([0, 400])

if saveFig == True:
    plt.savefig(fileloc + filename)