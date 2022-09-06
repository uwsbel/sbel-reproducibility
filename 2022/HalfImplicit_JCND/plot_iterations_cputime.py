import numpy as np

import pickle
import os

import csv
import matplotlib.pyplot as plt


# finding number of iterations and cpu times given model name and form
def findValues(model_name, form_name):
    cpu_times = []
    avg_itrs = []
    step_sizes = []
    for item in items:
        if item["model"] == model_name and item["form"] == form_name:
            cpu_times.append(float(item["cpu_time"]))
            avg_itrs.append(float(item["numItr"]))
            step_sizes.append((item["step size"]))
    return cpu_times, avg_itrs, step_sizes

# plot bar comparison plot
# x coodrinate stepsize
# y coordinate, number of iterations or cpu time
# parameter: step_sizes, val_half, val_fully, model name, figure name 
def barplot(step_sizes, val_half, val_fully, model_name, y_label_name):    
    # convert step size to a list of string
    string_step_sizes = [str(step_size) for step_size in step_sizes]
    
    numTests = len(step_sizes)
    X = np.arange(numTests) # x location of each group (each group has rA and rA half implicit)
    fig = plt.figure()
    fig.set_size_inches(15,15)
    Fontsize = 80
    LineWidth = 10
    MarkerSize = 40

    plt.rc('font', size=Fontsize)
    plt.rc('legend', fontsize=Fontsize*1.2)
    plt.rc('figure', titlesize=Fontsize)
    plt.rc('lines', linewidth=LineWidth)
    plt.rc('lines', markersize=MarkerSize)
    plt.rc('axes', linewidth=LineWidth*0.5)

    
    
    ax = fig.add_axes([0,0,1,1])
    dim = 0.2
    ax.bar(X + 0.00, val_half, color = 'b', width=dim)
    ax.bar(X + dim, val_fully, color = 'r', width=dim)
    

    # ax.legend(labels=['half implicit','fully implicit'])
    ax.legend(labels=['HI','FI'])

    plt.xticks(X+dim/2, string_step_sizes)
    plt.yticks(fontsize=Fontsize*0.85)
    plt.xticks(fontsize=Fontsize*0.85)

    plt.xlabel('step size')
    plt.ylabel(y_label_name)
    plt.title('{}'.format(" ".join(model_name.split("_"))))
    plt.grid(linestyle='--')
    
    


file = open("data/timing.csv", encoding='utf-8-sig')
csvreader = csv.reader(file)
keywords = next(csvreader)

items = []

for row in csvreader:
    item = {}
    for i, keyword in enumerate(keywords):
        item[keyword] = row[i]
    items.append(item)

file.close()

report_dir = "/home/luning/Manuscripts/Journal/2022/HalfImplicit/images/"

# for model_name in ['slider_crank', 'four_link']:
for model_name in ['slider_crank', 'four_link', 'double_pendulum']:
    form_name = 'half'
    cpu_time_half, avg_itr_half, step_sizes = findValues(model_name, form_name)
    
    form_name = 'fully'
    cpu_time_fully, avg_itr_fully, _ = findValues(model_name, form_name)
    
    barplot(step_sizes, cpu_time_half, cpu_time_fully, model_name, 'CPU time (sec)')
    filename = '{}_timing_updated'.format(model_name)
    plt.savefig(report_dir + filename, bbox_inches = 'tight')
    
    barplot(step_sizes, avg_itr_half, avg_itr_fully, model_name, 'Average number of iterations')
    filename = '{}_numItrs_updated'.format(model_name)
    plt.savefig(report_dir + filename, bbox_inches = 'tight')
    