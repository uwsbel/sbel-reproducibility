import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
import matplotlib.colors as colors
from torch.utils.data import TensorDataset, DataLoader
from Model.model import *
from Model.utils import *
from Model.force_fun import *
from Model.Data_generator import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gd_sc=np.load("saved_data/Slider_Crank/Ground_truth for Slider_Crank with training_size=300_num_steps_test=400_dt=0.01.npy")
pnode_sc=np.load("saved_data/Slider_Crank/PNODE for Slider_Crank with training_size=300_num_steps_test=400_dt=0.01.npy")
#pnode_sc=np.load("saved_data/Slider_Crank/PNODEsingle_step for Slider_Crank with training_size=300_num_steps_test=400_dt=0.01.npy")
pnode_sc_con=np.load("saved_data/Slider_Crank/PNODE_con for Slider_Crank with training_size=300_num_steps_test=400_dt=0.01.npy")
#pnode_sc_scon=np.load("saved_data/Slider_Crank/PNODE_con_soft2single_step for Slider_Crank with training_size=300_num_steps_test=400_dt=0.01.npy")
lstm_sc=np.load("saved_data/Slider_Crank/LSTM for Slider_Crank with training_size=300_num_steps_test=400_dt=0.01.npy")
fcnn_sc=np.load("saved_data/Slider_Crank/FCNN for Slider_Crank with training_size=300_num_steps_test=400_dt=0.01.npy")


pnode_sc=pnode_sc[0:400]
pnode_sc_no_con=pnode_sc_con[0:400]
#pnode_sc_scon=pnode_sc_scon[0:400]
lstm_sc=lstm_sc[0:400]
print(gd_sc.shape)
print(pnode_sc.shape)
print(pnode_sc_no_con.shape)

t_test=np.linspace(0,0.01*400,400)
#visualize the x,y trajectory like the double pendulum
#first figure is the x-t,v-t plot for the two bodies
#we totaly have 4 model with x-t,v-t plot
#using the subplot function just like the double pendulum above
# we want to add the mse error in the legend
#just like the triple mass spring damper, we plot the x-t, v-t for the double pendulum
# make a plot with multiple subplots
# scientific notation!!
# scientific style!!
subplot_labels = ['(a): MNODE', '(b): MNODE', '(c): MNODE_con', '(d): MNODE_con', '(e): LSTM', '(f): LSTM','(g): FCNN','(h):FCNN','(i)','(j)']


label_fontsize = 15
legend_fontsize = 15
# Vertical position of the label below the x-axis
vertical_position = -0.22  # Adjust this as needed
fig,axs=plt.subplots(4,2,figsize=(10,15))
axs[0, 0].plot(t_test, gd_sc[:,0,0], 'b-', label="Ground truth")
axs[0, 0].plot(t_test, gd_sc[:,1,0], 'b-')
axs[0, 0].plot(t_test[:300], pnode_sc[:300,0,0], 'r--', label="Model training")
axs[0, 0].plot(t_test[:300], pnode_sc[:300,1,0], 'r--')
axs[0, 0].plot(t_test[300:], pnode_sc[300:,0,0], 'r:', label="Model testing")
axs[0, 0].plot(t_test[300:], pnode_sc[300:,1,0], 'r:')
axs[0, 0].grid()
axs[0, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[0, 0].set_ylabel(r"$z$",fontsize=label_fontsize)
axs[0, 0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs[0, 0].transAxes)

axs[0, 1].plot(t_test, gd_sc[:,0,1], 'b-', label="Ground truth")
axs[0, 1].plot(t_test, gd_sc[:,1,1], 'b-')
axs[0, 1].plot(t_test[:300], pnode_sc[:300,0,1], 'r--', label="Model training")
axs[0, 1].plot(t_test[:300], pnode_sc[:300,1,1], 'r--')
axs[0, 1].plot(t_test[300:], pnode_sc[300:,0,1], 'r:', label="Model testing")
axs[0, 1].plot(t_test[300:], pnode_sc[300:,1,1], 'r:')
axs[0, 1].grid()
axs[0, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[0, 1].set_ylabel(r"$\dot z$",fontsize=label_fontsize)
axs[0, 1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs[0, 1].transAxes)

axs[1, 0].plot(t_test, gd_sc[:,0,0], 'b-', label="Ground truth")
axs[1, 0].plot(t_test, gd_sc[:,1,0], 'b-')
axs[1, 0].plot(t_test[:300], pnode_sc_no_con[:300,0,0], 'r--', label="Model training")
axs[1, 0].plot(t_test[:300], pnode_sc_no_con[:300,1,0], 'r--')
axs[1, 0].plot(t_test[300:], pnode_sc_no_con[300:,0,0], 'r:', label="Model testing")
axs[1, 0].plot(t_test[300:], pnode_sc_no_con[300:,1,0], 'r:')
axs[1, 0].grid()
axs[1, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[1, 0].set_ylabel(r"$z$",fontsize=label_fontsize)
axs[1, 0].text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=label_fontsize, transform=axs[1, 0].transAxes)

axs[1, 1].plot(t_test, gd_sc[:,0,1], 'b-', label="Ground truth")
axs[1, 1].plot(t_test, gd_sc[:,1,1], 'b-')
axs[1, 1].plot(t_test[:300], pnode_sc_no_con[:300,0,1], 'r--', label="Model training")
axs[1, 1].plot(t_test[:300], pnode_sc_no_con[:300,1,1], 'r--')
axs[1, 1].plot(t_test[300:], pnode_sc_no_con[300:,0,1], 'r:', label="Model testing")
axs[1, 1].plot(t_test[300:], pnode_sc_no_con[300:,1,1], 'r:')
axs[1, 1].grid()
axs[1, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[1, 1].set_ylabel(r"$\dot z$",fontsize=label_fontsize)
axs[1, 1].text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=label_fontsize, transform=axs[1, 1].transAxes)

axs[2, 0].plot(t_test, gd_sc[:,0,0], 'b-', label="Ground truth")
axs[2, 0].plot(t_test, gd_sc[:,1,0], 'b-')
axs[2, 0].plot(t_test[:300], lstm_sc[:300,0,0], 'r--', label="Model training")
axs[2, 0].plot(t_test[:300], lstm_sc[:300,1,0], 'r--')
axs[2, 0].plot(t_test[300:], lstm_sc[300:,0,0], 'r:', label="Model testing")
axs[2, 0].plot(t_test[300:], lstm_sc[300:,1,0], 'r:')
axs[2, 0].grid()
axs[2, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[2, 0].set_ylabel(r"$z$",fontsize=label_fontsize)
axs[2, 0].text(0.5, vertical_position, subplot_labels[4], ha='center', va='top', fontsize=label_fontsize, transform=axs[2, 0].transAxes)

axs[2, 1].plot(t_test, gd_sc[:,0,1], 'b-', label="Ground truth")
axs[2, 1].plot(t_test, gd_sc[:,1,1], 'b-')
axs[2, 1].plot(t_test[:300], lstm_sc[:300,0,1], 'r--', label="Model training")
axs[2, 1].plot(t_test[:300], lstm_sc[:300,1,1], 'r--')
axs[2, 1].plot(t_test[300:], lstm_sc[300:,0,1], 'r:', label="Model testing")
axs[2, 1].plot(t_test[300:], lstm_sc[300:,1,1], 'r:')
axs[2, 1].grid()
axs[2, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[2, 1].set_ylabel(r"$\dot z$",fontsize=label_fontsize)
axs[2, 1].text(0.5, vertical_position, subplot_labels[5], ha='center', va='top', fontsize=label_fontsize, transform=axs[2, 1].transAxes)

axs[3, 0].plot(t_test, gd_sc[:,0,0], 'b-', label="Ground truth")
axs[3, 0].plot(t_test, gd_sc[:,1,0], 'b-')
axs[3, 0].plot(t_test[:300], fcnn_sc[:300,0,0], 'r--', label="Model training")
axs[3, 0].plot(t_test[:300], fcnn_sc[:300,1,0], 'r--')
axs[3, 0].plot(t_test[300:], fcnn_sc[300:,0,0], 'r:', label="Model testing")
axs[3, 0].plot(t_test[300:], fcnn_sc[300:,1,0], 'r:')
axs[3, 0].grid()
axs[3, 0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[3, 0].set_ylabel(r"$z$",fontsize=label_fontsize)
axs[3, 0].text(0.5, vertical_position, subplot_labels[6], ha='center', va='top', fontsize=label_fontsize, transform=axs[3, 0].transAxes)

axs[3, 1].plot(t_test, gd_sc[:,0,1], 'b-', label="Ground truth")
axs[3, 1].plot(t_test, gd_sc[:,1,1], 'b-')
axs[3, 1].plot(t_test[:300], fcnn_sc[:300,0,1], 'r--', label="Model training")
axs[3, 1].plot(t_test[:300], fcnn_sc[:300,1,1], 'r--')
axs[3, 1].plot(t_test[300:], fcnn_sc[300:,0,1], 'r:', label="Model testing")
axs[3, 1].plot(t_test[300:], fcnn_sc[300:,1,1], 'r:')
axs[3, 1].grid()
axs[3, 1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs[3, 1].set_ylabel(r"$\dot z$",fontsize=label_fontsize)
axs[3, 1].text(0.5, vertical_position, subplot_labels[7], ha='center', va='top', fontsize=label_fontsize, transform=axs[3, 1].transAxes)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=legend_fontsize)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/Slider_Crank/SC_xvt.png")

#we want to plot the phase space trajectory
#which gives us the theta-omega, x-dx plot
#we totaly have 5 model with theta-omega, x-dx plot

fig2,axs2=plt.subplots(4,2,figsize=(10,15))
axs2[0, 0].plot(gd_sc[:,0,0],gd_sc[:,0,1],label="Ground truth",color="blue",linestyle="-")
axs2[0, 0].plot(pnode_sc[:300,0,0],pnode_sc[:300,0,1],label="Model training",color="red",linestyle="--")
axs2[0, 0].plot(pnode_sc[300:,0,0],pnode_sc[300:,0,1],label="Model testing",color="red",linestyle=":")
axs2[0, 0].grid()
axs2[0, 0].set_xlabel(r"$\theta$",fontsize=label_fontsize)
axs2[0, 0].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs2[0, 0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs2[0, 0].transAxes)

axs2[0, 1].plot(gd_sc[:,1,0],gd_sc[:,1,1],label="Ground truth",color="blue",linestyle="-")
axs2[0, 1].plot(pnode_sc[:300,1,0],pnode_sc[:300,1,1],label="MNODE training",color="red",linestyle="--")
axs2[0, 1].plot(pnode_sc[300:,1,0],pnode_sc[300:,1,1],label="MNODE testing",color="red",linestyle=":")
axs2[0, 1].grid()
axs2[0, 1].set_xlabel(r"$x$",fontsize=label_fontsize)
axs2[0, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs2[0, 1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs2[0, 1].transAxes)

axs2[1, 0].plot(gd_sc[:,0,0],gd_sc[:,0,1],label="Ground truth",color="blue",linestyle="-")
axs2[1, 0].plot(pnode_sc_no_con[:300,0,0],pnode_sc_no_con[:300,0,1],label="MNODE_con training",color="red",linestyle="--")
axs2[1, 0].plot(pnode_sc_no_con[300:,0,0],pnode_sc_no_con[300:,0,1],label="MNODE_con testing",color="red",linestyle=":")
axs2[1, 0].grid()
axs2[1, 0].set_xlabel(r"$\theta$",fontsize=label_fontsize)
axs2[1, 0].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs2[1, 0].text(0.5, vertical_position, subplot_labels[2], ha='center', va='top', fontsize=label_fontsize, transform=axs2[1, 0].transAxes)

axs2[1, 1].plot(gd_sc[:,1,0],gd_sc[:,1,1],label="Ground truth",color="blue",linestyle="-")
axs2[1, 1].plot(pnode_sc_no_con[:300,1,0],pnode_sc_no_con[:300,1,1],label="MNODE_con training",color="red",linestyle="--")
axs2[1, 1].plot(pnode_sc_no_con[300:,1,0],pnode_sc_no_con[300:,1,1],label="MNODE_con testing",color="red",linestyle=":")
axs2[1, 1].grid()
axs2[1, 1].set_xlabel(r"$x$",fontsize=label_fontsize)
axs2[1, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs2[1, 1].text(0.5, vertical_position, subplot_labels[3], ha='center', va='top', fontsize=label_fontsize, transform=axs2[1, 1].transAxes)

axs2[2, 0].plot(gd_sc[:,0,0],gd_sc[:,0,1],label="Ground truth",color="blue",linestyle="-")
axs2[2, 0].plot(lstm_sc[:300,0,0],lstm_sc[:300,0,1],label="LSTM training",color="red",linestyle="--")
axs2[2, 0].plot(lstm_sc[300:,0,0],lstm_sc[300:,0,1],label="LSTM testing",color="red",linestyle=":")
axs2[2, 0].grid()
axs2[2, 0].set_xlabel(r"$\theta$",fontsize=label_fontsize)
axs2[2, 0].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs2[2, 0].text(0.5, vertical_position, subplot_labels[4], ha='center', va='top', fontsize=label_fontsize, transform=axs2[2, 0].transAxes)

axs2[2, 1].plot(gd_sc[:,1,0],gd_sc[:,1,1],label="Ground truth",color="blue",linestyle="-")
axs2[2, 1].plot(lstm_sc[:300,1,0],lstm_sc[:300,1,1],label="LSTM training",color="red",linestyle="--")
axs2[2, 1].plot(lstm_sc[300:,1,0],lstm_sc[300:,1,1],label="LSTM testing",color="red",linestyle=":")
axs2[2, 1].grid()
axs2[2, 1].set_xlabel(r"$x$",fontsize=label_fontsize)
axs2[2, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs2[2, 1].text(0.5, vertical_position, subplot_labels[5], ha='center', va='top', fontsize=label_fontsize, transform=axs2[2, 1].transAxes)

axs2[3, 0].plot(gd_sc[:,0,0],gd_sc[:,0,1],label="Ground truth",color="blue",linestyle="-")
axs2[3, 0].plot(fcnn_sc[:300,0,0],fcnn_sc[:300,0,1],label="FCNN training",color="red",linestyle="--")
axs2[3, 0].plot(fcnn_sc[300:,0,0],fcnn_sc[300:,0,1],label="FCNN testing",color="red",linestyle=":")
axs2[3, 0].grid()
axs2[3, 0].set_xlabel(r"$\theta$",fontsize=label_fontsize)
axs2[3, 0].set_ylabel(r"$\omega$",fontsize=label_fontsize)
axs2[3, 0].text(0.5, vertical_position, subplot_labels[6], ha='center', va='top', fontsize=label_fontsize, transform=axs2[3, 0].transAxes)

axs2[3, 1].plot(gd_sc[:,1,0],gd_sc[:,1,1],label="Ground truth",color="blue",linestyle="-")
axs2[3, 1].plot(fcnn_sc[:300,1,0],fcnn_sc[:300,1,1],label="FCNN training",color="red",linestyle="--")
axs2[3, 1].plot(fcnn_sc[300:,1,0],fcnn_sc[300:,1,1],label="FCNN testing",color="red",linestyle=":")
axs2[3, 1].grid()
axs2[3, 1].set_xlabel(r"$x$",fontsize=label_fontsize)
axs2[3, 1].set_ylabel(r"$v$",fontsize=label_fontsize)
axs2[3, 1].text(0.5, vertical_position, subplot_labels[7], ha='center', va='top', fontsize=label_fontsize, transform=axs2[3, 1].transAxes)
handles, labels = axs2[0, 0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=legend_fontsize)
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/Slider_Crank/SC_phasespace.png")

con_loss_gd=np.zeros(400)
con_loss_pnode=np.zeros(400)
con_loss_pnode_con=np.zeros(400)
con_loss_lstm=np.zeros(400)
con_loss_fcnn=np.zeros(400)
r=1.0
l=4
#calculate the constraint loss for the whole trajectory following the equation
#x_values[i] = r * np.cos(theta_values[i]) + np.sqrt(l ** 2 - r ** 2 * np.sin(theta_values[i]) ** 2)
for i in range(0,400):
    con_loss_gd[i]=np.abs((r*np.cos(gd_sc[i,0,0])+np.sqrt(l**2-r**2*np.sin(gd_sc[i,0,0])**2)-gd_sc[i,1,0]))
    con_loss_pnode[i]=np.abs((r*np.cos(pnode_sc[i,0,0])+np.sqrt(l**2-r**2*np.sin(pnode_sc[i,0,0])**2)-pnode_sc[i,1,0]))
    con_loss_pnode_con[i]=np.abs((r*np.cos(pnode_sc_no_con[i,0,0])+np.sqrt(l**2-r**2*np.sin(pnode_sc_no_con[i,0,0])**2)-pnode_sc_no_con[i,1,0]))
    con_loss_lstm[i]=np.abs((r*np.cos(lstm_sc[i,0,0])+np.sqrt(l**2-r**2*np.sin(lstm_sc[i,0,0])**2)-lstm_sc[i,1,0]))
    con_loss_fcnn[i]=np.abs((r*np.cos(fcnn_sc[i,0,0])+np.sqrt(l**2-r**2*np.sin(fcnn_sc[i,0,0])**2)-fcnn_sc[i,1,0]))
#split the constraint loss into the training and testing part
con_loss_gd_train=con_loss_gd[0:300]
con_loss_gd_test=con_loss_gd[300:]
con_loss_pnode_train=con_loss_pnode[0:300]
con_loss_pnode_test=con_loss_pnode[300:]
con_loss_pnode_con_train=con_loss_pnode_con[0:300]
con_loss_pnode_con_test=con_loss_pnode_con[300:]
con_loss_lstm_train=con_loss_lstm[0:300]
con_loss_lstm_test=con_loss_lstm[300:]
con_loss_fcnn_train=con_loss_fcnn[0:300]
con_loss_fcnn_test=con_loss_fcnn[300:]
#plot the constraint loss for the whole trajectory

#we also compute the constraint loss for #dxdt_values[i] = omega_values[i] * r * np.sin(theta_values[i])

con_loss_gd2=np.zeros(400)
con_loss_pnode2=np.zeros(400)
con_loss_pnode_con2=np.zeros(400)
con_loss_pnode_scon2=np.zeros(400)
con_loss_lstm2=np.zeros(400)
con_loss_fcnn2=np.zeros(400)
r=1.0
#calculate the constraint loss for the whole trajectory following the equation
#dxdt_values[i] = omega_values[i] * r * np.sin(theta_values[i])
for i in range(0,400):
    con_loss_gd2[i]=np.abs((-gd_sc[i,1,1]-gd_sc[i,0,1]*r*np.sin(gd_sc[i,0,0])))
    con_loss_pnode2[i]=np.abs((-pnode_sc[i,1,1]-pnode_sc[i,0,1]*r*np.sin(pnode_sc[i,0,0])))
    con_loss_pnode_con2[i]=np.abs((-pnode_sc_no_con[i,1,1]-pnode_sc_no_con[i,0,1]*r*np.sin(pnode_sc_no_con[i,0,0])))
    con_loss_lstm2[i]=np.abs((-lstm_sc[i,1,1]-lstm_sc[i,0,1]*r*np.sin(lstm_sc[i,0,0])))
    con_loss_fcnn2[i]=np.abs((-fcnn_sc[i,1,1]-fcnn_sc[i,0,1]*r*np.sin(fcnn_sc[i,0,0])))

fig3,axs3=plt.subplots(1,2,figsize=(10,6))
vertical_position = -0.12  # Adjust this as needed
axs3[0].plot(t_test,con_loss_gd,label="Ground truth",color="blue",linestyle="-")
axs3[0].plot(t_test,con_loss_pnode,label="MNODE",color="red",linestyle="--")
axs3[0].plot(t_test,con_loss_pnode_con,label="MNODE_con training",color="green",linestyle="--")
axs3[0].plot(t_test,con_loss_lstm,label="LSTM",color="orange",linestyle="--")
axs3[0].plot(t_test,con_loss_fcnn,label="FCNN",color="black",linestyle="--")
axs3[0].axvline(x=3.0,linestyle="-",color="black",label="Training and testing split",linewidth=2)
axs3[0].grid()
axs3[0].set_yscale("log")
axs3[0].set_xlabel(r"$t$",fontsize=label_fontsize)
axs3[0].set_ylabel(r"$Constraint loss$",fontsize=label_fontsize)
axs3[0].text(0.5, vertical_position, subplot_labels[0], ha='center', va='top', fontsize=label_fontsize, transform=axs3[0].transAxes)

axs3[1].plot(t_test,con_loss_gd2,label="Ground truth",color="blue",linestyle="-")
axs3[1].plot(t_test,con_loss_pnode2,label="MNODE",color="red",linestyle="--")
axs3[1].plot(t_test,con_loss_pnode_con2,label="MNODE_con training",color="green",linestyle="--")
axs3[1].plot(t_test,con_loss_lstm2,label="LSTM",color="orange",linestyle="--")
axs3[1].plot(t_test,con_loss_fcnn2,label="FCNN",color="black",linestyle="--")
axs3[1].axvline(x=3.0,linestyle="-",color="black",label="Training and testing split",linewidth=2)
axs3[1].grid()
axs3[1].set_xlabel(r"$t$",fontsize=label_fontsize)
axs3[1].set_ylabel(r"$Constraint loss$",fontsize=label_fontsize)
axs3[1].text(0.5, vertical_position, subplot_labels[1], ha='center', va='top', fontsize=label_fontsize, transform=axs3[1].transAxes)
axs3[1].set_yscale("log")
handles, labels = axs3[0].get_legend_handles_labels()
fig3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=legend_fontsize)
fig3.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig("figures/Slider_Crank/SC_con_loss_log.png")
plt.show()
#calculate the MSE between the ground truth and the model prediction


mse_pnode=np.mean((gd_sc-pnode_sc)**2)
mse_pnode_con=np.mean((gd_sc-pnode_sc_no_con)**2)
mse_lstm=np.mean((gd_sc-lstm_sc)**2)
mse_fcnn=np.mean((gd_sc-fcnn_sc)**2)
#print the mse for all the models
print("The MSE for the PNODE model is",mse_pnode)
print("The MSE for the PNODE_con model is",mse_pnode_con)
print("The MSE for the LSTM model is",mse_lstm)
print("The MSE for the FCNN model is",mse_fcnn)

