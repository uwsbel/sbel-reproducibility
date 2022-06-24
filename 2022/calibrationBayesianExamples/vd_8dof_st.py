import scipy as sp
import aesara
import aesara.tensor as tt
import pymc as pm
import arviz as az
import pandas as pd
import os
from datetime import datetime
import sys
import numpy as np
import transcript
from vd_class import vd_8dof
from point import Point

#This is just a gaussian log likelihood
def loglike(theta,data):

    sigmas = np.array(theta[-(data.shape[0]):]).reshape(-1,1)
    n = data.shape[1]
	#Need to update the parameters using the update params method
    vehicle.update_params(Cf=theta[0],Cr=theta[1],krof=theta[2],kror=theta[3],brof=theta[4],bror=theta[5],m=2097.85,muf=127.866,mur=129.98,a= 1.6889,b =1.6889,h = 0.713,cf = 1.82,cr = 1.82,Jx = 1289,Jz = 4519,Jxz = 3.265,
    r0=0.47,ktf=326332,ktr=326332,hrcf=0.379,hrcr=0.327,Jw=11,
    Cxf = 10300,Cxr = 10300,rr=0.0175)


    mod = vehicle.solve_half_impl(t_span = [t_eval[0],t_eval[-1]],t_eval = t_eval,tbar = 1e-2)
    mod = np.transpose(mod)

    vehicle.reset_state(init_state=st)

    return -np.sum(((n*np.log(2*np.pi * sigmas**2)/2) + np.sum((mod[[3,5,6,7],:] - data)**2/(2.*sigmas**2)))/np.linalg.norm(data,axis = 1))

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
    with open('data/vd_chrono_ramp_1.npy', 'rb') as f:
        data = np.load(f)


    #For saving all necesarry files
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    savedir = str('{}'.format(date))

    #Initiating the loglikelihood object which is a aesera operation (op)
    like = LogLike(loglike,data)
    with pm.Model() as model:

        Cf = pm.Uniform("Cf",lower=20000,upper=80000,initval = 40000)
        Cr = pm.Uniform("Cr",lower=20000,upper=80000,initval = 40000)
        krof = pm.Uniform("krof",lower=1000,upper=50000,initval = 20000)
        kror = pm.Uniform("kror",lower=1000,upper=50000,initval = 20000)
        bro = pm.Uniform("bro",lower=200,upper=60000,initval=8000)
        bror = pm.Deterministic('bror',bro/2)
        brof = pm.Deterministic('brof',bro/2)

        # For nuts, we scale the sigmas as they are too small - vanishing gradients
        if(sys.argv[2] == "nuts"):
            sigmaRA_ = pm.HalfNormal("sigmaRA_",sigma = 5,initval=3)
            sigmaRR_ = pm.HalfNormal("sigmaRR_",sigma = 5,initval=1.5)
            sigmaLV_ = pm.HalfNormal("sigmaLV_",sigma = 5,initval=3)
            sigmaYR_ = pm.HalfNormal("sigmaYR_",sigma = 5,initval=1.5)

            sigmaRA =  pm.Deterministic('sigmaRA',sigmaRA_*10e-3)
            sigmaRR =  pm.Deterministic('sigmaRR',sigmaRR_*10e-3)
            sigmaLV =  pm.Deterministic('sigmaLV',sigmaLV_*10e-2)
            sigmaYR =  pm.Deterministic('sigmaYR',sigmaYR_*10e-2)

        else:
            sigmaRA = pm.HalfNormal("sigmaRA",sigma = 0.005,initval=0.003)
            sigmaRR = pm.HalfNormal("sigmaRR",sigma = 0.005,initval=0.0015)
            sigmaLV = pm.HalfNormal("sigmaLV",sigma = 0.05,initval=0.03)
            sigmaYR = pm.HalfNormal("sigmaYR",sigma = 0.05,initval=0.015)


        ## All of these will be a tensor 
        theta_ = [Cf,Cr,krof,kror,brof,bror,sigmaLV,sigmaRA,sigmaRR,sigmaYR]
        theta = tt.as_tensor_variable(theta_)

        pm.Potential("like",like(theta))



        transcript.start('./results/' + savedir + '_dump.log')
        if(sys.argv[2] == "nuts"):
            idata = pm.sample(ndraws ,tune=nburn,discard_tuned_samples=True,return_inferencedata=True,target_accept = 0.9, cores=8)
        elif(sys.argv[2] == "met"):
            step = pm.Metropolis()
            idata = pm.sample(ndraws,step=step, tune=nburn,discard_tuned_samples=True,return_inferencedata=True,cores=8)
        elif(sys.argv[2] == "smc"):
            idata = pm.sample_smc(draws = ndraws,parallel=True,cores=8,return_inferencedata=True,progressbar = True)
        else:
            print("Please provide nuts or met as the stepping method")

        idata.to_netcdf('./results/' + savedir + ".nc")
        transcript.start('./results/' + savedir + '.log')


        for i in range(0,len(theta_)):
            print(f"{theta_[i]}")

        try:
            print(az.summary(idata).to_string())
        except KeyError:
            idata.to_netcdf('./results/' + savedir + ".nc")

        
        transcript.stop('./results/' + savedir + '.log')



if __name__ == "__main__":
    # A ramp steer
    pt1 = Point(0,0)
    pt2 = Point(3.7,0.2)
    ramp_st_3 = pt1.get_eq(pt2)

    # Used for obtaining the state from the chrono vehicle
    n1  = 700
    n2 = 1070


    # The time duration of the simulation-
    st_time = 0.
    end_time = 3.7


    # The times for which we want the simulation
    t_eval  = np.arange(st_time,end_time,0.01)

    def zero_throt(t):
        return 0 * t

    def brake_tor(t):
        return 0 * t

    state = pd.read_csv("data/simp_ramp.csv",sep=',',header='infer')

    st = {'x' : state['x'][n1]-state['x'][n1],'y':state['y'][n1]-state['y'][n1],'u':state['vx'][n1],
    'v':state['vy'][n1],'psi':state['yaw'][n1],'phi':state['roll'][n1],'wx':state['roll_rate'][n1],
    'wz':state['yaw_rate'][n1],'wlf' : state['wlf'][n1],'wlr' : state['wlr'][n1],'wrf' : state['wrf'][n1],'wrr' : state['wrr'][n1]}

    vehicle = vd_8dof(states = st)

    # Set the steering and the throttle functions we just created above
    vehicle.set_steering(ramp_st_3)
    vehicle.set_throttle(zero_throt,gr=0.3*0.2)
    vehicle.set_braking(brake_tor)
    vehicle.debug = 0
    main()
