
import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:


x = 0.1; #initial actual state
x_N = 1 #Noise covariance in the system (i.e. process noise in the state update, here, we'll use a gaussian.)
x_R = 1 #Noise covariance in the measurement (i.e. the Quail creates complex illusions in its trail!)
T = 75 #duration the chase (i.e. number of iterations).
N = 10
v=2
x_P_update=np.zeros(shape=(10,1));
z_update=np.zeros(shape=(10,1));
P_w=np.zeros(shape=(10,1));

x_p=np.zeros(shape=(10,1))
print(x_p[3])


# In[4]:


for i in range(N):
    x_p[i]=x+np.sqrt(v)+np.random.randn()
z_out = [x**2 / 20 + np.sqrt(x_R) * np.random.randn()]
print(z_out);
x_out = [x];  #the actual output vector for measurement values.
x_est = x; # time by time output of the particle filters estimate
x_est_out = [x_est]; 
    #print(x_out)# the vector of particle filter estimates.
print(x_est_out)


# In[3]:



for t in range (1,T):
    x = 0.5*x + 25*x/(1 + x**2) + 8*np.cos(1.2*(t-1)) +  np.sqrt(x_N)*np.random.randn(); ##state update
    z = x**2/20 + np.sqrt(x_R)*np.random.randn();
    for i in range (1,N):     ## N = number of particles
        x_P_update[i] = 0.5*x_p[i] + 25*x_p[i]/(1 + x_p[i]**2) + 8*math.cos(1.2*(t-1)) + np.sqrt(x_N)*np.random.randn();## position update for each particle
        z_update[i] = x_P_update[i]**2/20;
        P_w[i] = (1/np.sqrt(2*np.pi*x_R)) * np.exp(-(z - z_update[i])**2/(2*x_R)); ##weights assigned for each particle according to z_update and z,particles with higher relevance assigned larger weights 
    
    P_w[0]= [2.21646841e-76]
    P_w = np.divide(P_w,np.sum(P_w)) ##normalisation of the weights

   
    mar=np.zeros(shape=(1,10));
    for i in range (1 , N):
        mar=((np.random.rand() <= np.cumsum(P_w))) ## code for resampling optimisation  
        for io in range (1 , 10):
            if mar[io]==True:
                x_p[i]=x_P_update[io]
                break
    x_est = np.mean(x_p); 
    x_out.append(x)
    z_out.append(z)
    x_est_out.append(x_est)

rmse=np.sqrt(np.mean((x_out-x_est)**2))
print(rmse)## root mean squared error between the estimated and actual positions or states 



x1 = np.arange(0,T);
plt.plot(x1, x_est_out, label = "(x_est_out) filtered observation") ## plotting out the tracking
plt.plot(x1, x_out, label = "(x_out)  observation")

plt.legend();
plt.title('Observation and Filtered Observation');
plt.show();


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




   


# In[217]:




