import numpy as np
import matplotlib.pyplot as plt

delim = '\t'

#The lines before the steer
accel_lines = '0\t0\t0\t0\n0.5\t0\t0\t0\n1.5\t0\t1\t0\n5.5\t0\t1\t0\n6\t0\t0\t0\n'

#The time at which the steering starts
t_start = 7.


delta = 0.01
#The time at which the steering ends
t_end = 10.7 + delta


t = np.arange(t_start,t_end,delta)

#Points for the sin steer
def sin_steer(t,t_start):
    return 4*np.pi/180*np.sin((1/2)*2.*np.pi*(t-t_start))


st = sin_steer(t,t_start)

print(t.shape)

# print(st)
plt.plot(t,st)
plt.show()
with open('sin_steer.txt', 'w') as f:
    f.write(accel_lines)
    #Write all the steering points
    for index,time in enumerate(t):
        f.write(f'{round(time,2)}\t{st[index]}\t0\t0')
        f.write('\n')

