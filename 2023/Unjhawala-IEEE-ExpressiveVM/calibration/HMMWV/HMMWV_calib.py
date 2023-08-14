import sys
sys.path.append('../../../../../model-repo/simple-vehicles/lang-c/interfaces')
import matplotlib.pyplot as mpl
import scipy as sp
import pytensor.tensor as tt
import pymc as pm
import arviz as az
import pandas as pd
import os
from datetime import datetime
import sys
import numpy as np
import pickle
import time
import random
# import our reduced order models
import rom


rpm2rad = np.pi / 30

# since PyMC requires pickling of the step method, our likelihood function cannot have any references 
# to C objects, we thus have this model function that generates all the references we need, does all the computations
# required for the likelihood, and then return the required model response. As we leave the function call,
# all the C references are popped anyways, so our likelihood can be pickled!
def model(theta,fileName,endTime):
    
    # lets get a vector of entries going 
    driverData = rom.vector_entry()

    # lets fill this up from our data file
    rom.driverInput(driverData,fileName)

    # lets get our vector of doubles which will hold the controls at each time
    controls = rom.vector_double(4,0)

    veh1_param = rom.VehicleParam()
    rom.setVehParamsJSON(veh1_param,fileName_veh)
    tire_param = rom.TMeasyParam()
    rom.setTireParamsJSON(tire_param,fileName_tire)

    # Initialize our vehicle state in each iteration 
    veh1_st = rom.VehicleState()
    rom.vehInit(veh1_st,veh1_param)

    tirelf_st = rom.TMeasyState()
    tirerf_st = rom.TMeasyState()
    tirelr_st = rom.TMeasyState()
    tirerr_st = rom.TMeasyState()
    rom.tireInit(tire_param)


    veh1_param._step = 0.001
    tire_param._step = 0.001
    step = veh1_param._step

    ########################################## Set the new parameters #########
    # print(theta)
    tire_param._dfy0Pn = tire_param._dfy0Pn * theta[0]
    tire_param._dfy0P2n = tire_param._dfy0P2n * theta[0]

    tire_param._fymPn = tire_param._fymPn * theta[1]
    tire_param._fymP2n = tire_param._fymP2n * theta[1]

    veh1_param._maxSteer = veh1_param._maxSteer * theta[1]
    
    tire_param._dfx0Pn = tire_param._dfx0Pn * theta[2]
    tire_param._dfx0P2n = tire_param._dfx0P2n * theta[2]

    tire_param._fxmPn = tire_param._fxmPn * theta[3]
    tire_param._fxmP2n = tire_param._fxmP2n * theta[3]

    # Map factors
    tor_size = veh1_param._powertrainMap.size()
    for p in range(tor_size):
        veh1_param._powertrainMap[p]._y = veh1_param._powertrainMap[p]._y  * theta[5]

    loss_size = veh1_param._lossesMap.size()
    for k in range(loss_size):
        veh1_param._lossesMap[k]._y = veh1_param._lossesMap[k]._y  * theta[3]


    ##########################################################################
    # solve the model
    result = []
    t = 0
    timeStepNo = 0
    while(t<endTime):
        # get the controls for the time step
        rom.getControls(controls, driverData, t)

        #transfrom vehicle velocities to tire velocites
        rom.vehToTireTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls)

        # advanvce our 4 tires with this transformed velocity
        rom.tireAdv(tirelf_st, tire_param, veh1_st, veh1_param, controls)
        rom.tireAdv(tirerf_st, tire_param, veh1_st, veh1_param, controls)

        # rear wheel does not steer, so need to give it modified controls
        mod_controls = [controls[0],0,controls[2],controls[3]]

        rom.tireAdv(tirelr_st, tire_param, veh1_st, veh1_param, mod_controls)
        rom.tireAdv(tirerr_st, tire_param, veh1_st, veh1_param, mod_controls)

        rom.evalPowertrain(veh1_st, tirelf_st, tirerf_st, tirelr_st, tirerr_st, veh1_param, tire_param, controls)

        # transfrom tire forces into vehicle coordinate frame
        rom.tireToVehTransform(tirelf_st,tirerf_st,tirelr_st,tirerr_st,veh1_st,veh1_param,controls)

        # copy useful stuff needed for the vehicle to advance
        fx = [tirelf_st._fx,tirerf_st._fx,tirelr_st._fx,tirerr_st._fx]
        fy = [tirelf_st._fy,tirerf_st._fy,tirelr_st._fy,tirerr_st._fy]
        huf = tirelf_st._rStat
        hur = tirerr_st._rStat

        # look at the vehicle go
        rom.vehAdv(veh1_st,veh1_param,fx,fy,huf,hur)


        #append timestep results
        if(timeStepNo % 10 == 0):
            result.append([t, veh1_st._x, veh1_st._y, veh1_st._u, veh1_st._v, veh1_st._phi,
                            veh1_st._psi, veh1_st._wx, veh1_st._wz,tirelf_st._omega,
                            tirerf_st._omega, tirelr_st._omega, tirerr_st._omega])

        t += step
        timeStepNo += 1
    return result



#This is just a gaussian log likelihood
def loglike(theta,data):
    sigmas = np.array(theta[-(data[0].shape[0]):]).reshape(-1,1)

    # print(theta)
    likes = [None]*len(fileName_con)

    # For each of the input controls, run the model and get the likelihood
    for i,fileName in enumerate(fileName_con):
        n = data[i].shape[1]
        mod = model(theta,fileName,endTimes[i])
        mod = np.array(mod)[:data[i].shape[1]].T # truncate the model output to how many ever points we have in the data
        data_ = data[i]
        likelihood = -np.sum(((n*np.log(2*np.pi * sigmas**2)/2) + np.sum((mod[[3,6],:] - data_)**2/(2.*sigmas**2)))/np.linalg.norm(data_,axis = 1))
        # print(mod[[1,2],:])
        likes[i] = likelihood


    # just take the log like as the sum of our individual likes
    return sum(likes)/len(likes)

#The gradient of the log likelihood using finite differences - Needed for gradient based methods
def grad_loglike(theta,data):
	def ll(theta,data):
		return loglike(theta,data)
	
	#We are not doing finite difference approximation and we define delx as the finite precision 
	eps = np.sqrt(np.finfo(float).eps)
	delx = eps * np.sqrt(np.abs(theta))

	return sp.optimize.approx_fprime(theta,ll,delx,data)



# define a aesara Op for our likelihood function
class LogLike(tt.Op):
	itypes = [tt.dvector] # expects a vector of parameter values when called
	otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

	def __init__(self, loglike,data):

		# add inputs as class attributes
		self.likelihood = loglike
		self.data = data
		self.loglike_grad = LoglikeGrad(self.data)

	def perform(self, node, inputs, outputs):
		# the method that is used when calling the Op
		theta, = inputs  # this will contain my variables

		# call the log-likelihood function
		logp = self.likelihood(theta,self.data)

		outputs[0][0] = np.array(logp) # output the log-likelihood
	def grad(self,inputs,grad_outputs):
		theta, = inputs
		grads = self.loglike_grad(theta)
		return [grad_outputs[0] * grads]
		
		
#Similarly wrapper class for loglike gradient
class LoglikeGrad(tt.Op):
	itypes = [tt.dvector]
	otypes = [tt.dvector]

	def __init__(self,data):
		self.der_likelihood = grad_loglike
		self.data = data

	def perform(self, node, inputs, outputs):
		(theta,) = inputs
		grads = self.der_likelihood(theta,self.data)
		outputs[0][0] = grads


def main():

    # Specify number of draws as a command line argument
    if(len(sys.argv) < 2):
        print("Please provide the number of draws and the stepping method")
    ndraws = int(sys.argv[1])
    #Always burn half - can chnage this at some point
    nburn = int(ndraws/2)


    #######################################################################################
    # Input data is now a combination of alot of data. Our algoirthm randomly picks one data
    #######################################################################################    

    #For saving all necesarry files
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    savedir = str('{}'.format(date))
    print(savedir)

    
    #Initiating the loglikelihood object which is a aesara operation (op)
    like = LogLike(loglike,data)
    with pm.Model() as model:
        # Powertrain map points
        f_dfy = pm.Uniform("f_dfy",lower=0.5,upper=1.75,initval = 1)
        f_fym = pm.Uniform("f_fym",lower=0.5,upper=1.5,initval = 1)
        f_maxSteer = pm.Uniform("f_maxSteer", lower = 0.7, upper = 1.5, initval = 1)
        
        # Longitudinal parameters
        f_dfx = pm.Uniform("f_dfx",lower=0.1,upper=2,initval = 1)
        f_fxm = pm.Uniform("f_fxm",lower=0.25,upper=1.25,initval = 1)
        f_tor = pm.Uniform("f_tor",lower=0.6,upper=1.2,initval = 1) # Factor to multiply torque map
        f_loss = pm.Uniform("f_loss",lower=0.6,upper=1.2,initval = 1) # Factor to multiply losses map

        
        sigmaLOV = pm.HalfNormal("sigmaLOV",sigma = 0.2,initval=0.2)
        sigmaYaw = pm.HalfNormal("sigmaYaw",sigma = 0.02,initval=0.02)
        theta_ = [f_dfy, f_fym, f_maxSteer, f_dfx, f_fxm, f_tor, f_loss, sigmaLOV,sigmaYaw]

        theta = tt.as_tensor_variable(theta_)

        pm.Potential("like",like(theta))


        #Now we sample!
        #We use metropolis as the algorithm with parameters to be sampled supplied through vars
        if(sys.argv[2] == "nuts"):
            # step = pm.NUTS()
            # pm.sampling.init_nuts()
            idata = pm.sample(ndraws ,tune=nburn,discard_tuned_samples=True,return_inferencedata=True,target_accept = 0.9, cores=8)
        elif(sys.argv[2] == "met"):
            step = pm.Metropolis()
            idata = pm.sample(ndraws,step=step, tune=nburn,discard_tuned_samples=True,return_inferencedata=True,cores=8)
        elif(sys.argv[2] == "smc"):
            idata = pm.sample_smc(draws = ndraws,cores=4,return_inferencedata=True,progressbar = True)
        else:
            print("Please provide nuts or met as the stepping method")

        # idata.extend(pm.sample_prior_predictive())
        # idata.extend(pm.sample_posterior_predictive(idata))
        idata.to_netcdf('./results/' + savedir + ".nc")


        for i in range(0,len(theta_)):
            print(f"{theta_[i]}")

        try:
            print(az.summary(idata).to_string())
        except KeyError:
            idata.to_netcdf('./results/' + savedir + ".nc")




if __name__ == "__main__":
    
    #initilaize vehicle from json files and all the shenanigans

    file1 = sys.argv[3]
    file2 = sys.argv[4]
    file3 = sys.argv[5]
    file4 = sys.argv[6]
    file5 = sys.argv[7]
    # get the file name of the vehicle controls

    # Get the input file 
    fileName1 = "./inputs/" + file1 + ".txt"
    fileName2 = "./inputs/" + file2 + ".txt"
    fileName3 = "./inputs/" + file3 + ".txt"
    fileName4 = "./inputs/" + file4 + ".txt"
    fileName5 = "./inputs/" + file5 + ".txt"

    fileName_con = [fileName1, fileName2, fileName3, fileName4, fileName5]

    print(fileName_con)
    # open the data files
    with open("./data/" + file1 + "_shafts.npy", 'rb') as f:
        file1_data = np.load(f)
    with open("./data/" + file2 + "_shafts.npy", 'rb') as f:
        file2_data = np.load(f)
    with open("./data/" + file3 + "_shafts.npy", 'rb') as f:
        file3_data = np.load(f)
    with open("./data/" + file4 + "_shafts.npy", 'rb') as f:
        file4_data = np.load(f)
    with open("./data/" + file5 + "_shafts.npy", 'rb') as f:
        file5_data = np.load(f)

    # data is not a list, in likelihood we will only pick one of these data's randomly
    data = [file1_data, file2_data, file3_data, file4_data, file5_data]
    
    # json parameters file names - not used

    fileName_veh = "./jsons/HMMWV.json"
    fileName_tire = "./jsons/TMeasy.json"
    
    # end time of the simulation
    endTimes = [20.009, 20.009, 20.009, 20.009, 20.009]

    main()
