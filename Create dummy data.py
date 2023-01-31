#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
#create  the variables
mean = [-3, 0]
cov = [[1.0, 0.8], [0.8, 1.0]]
seed = np.random.seed(0)
rand_num = np.random.multivariate_normal(mean, cov, 500)
print(rand_num)
rand_num.shape


# In[2]:


rand_num.shape


# In[ ]:


Problem 2


# In[3]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.title("Visualization by scatter plot")
plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(rand_num[:,0], rand_num[:,1])


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('x1')
plt.ylabel('frequency')
plt.title('Histogram of x1')
plt.hist(rand_num[:,0])
plt.xlim(left=-5)
plt.show()


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('x2')
plt.ylabel('frequency')
plt.title('Histogram of x2')
plt.hist(rand_num[:,1])
plt.xlim(right=2)
plt.show()


# In[ ]:


Problem 4


# In[8]:


mean = [0, -3]
cov = [[1.0, 0.8], [0.8, 1.0]]
seed = np.random.seed(0)
rand_num1 = np.random.multivariate_normal(mean, cov, 500)
print(rand_num1)


# In[9]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.figure(figsize=(20,20))
plt.title("Two in one scatter plot")
plt.xlabel("x1")
plt.ylabel("x2")

plt.scatter(rand_num[:,0], rand_num[:,1],s=20, label='0', marker='o')

plt.scatter(rand_num1[:,0], rand_num1[:,1], s=20,label='1', marker='o')
plt.legend()
plt.show()


# In[ ]:


Problem 5


# In[10]:


data_combine = np.concatenate((rand_num, rand_num1),axis=0)
print(data_combine)


# In[11]:


data_combine.shape


# In[16]:


## Using vstack
combined_data = np.vstack((rand_num, rand_num1))
combined_data.shape


# In[ ]:


Problem 6


# In[13]:


#labelproblem1 = {}
#labelproblem1[0] = rand_num

#labelproblem2 = {}
#labelproblem2[1] = rand_num1

#labelproblem1
#labelproblem2


# In[ ]:





# In[17]:


#Creating a (1000, 3) ndarray
zeros = np.zeros(500)
ones = np.ones(500)
new_column = np.concatenate((zeros, ones))
new_ndarr = np.append(combined_data, new_column[:,None], 1)
print(new_ndarr)


# In[18]:


new_ndarr.shape


# In[ ]:





# In[ ]:





# In[ ]:




