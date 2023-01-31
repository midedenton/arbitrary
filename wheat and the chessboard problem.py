#!/usr/bin/env python
# coding: utf-8

# In[1]:


n_squares = 4
small_board_list = [1]
for _ in range(n_squares - 1):
    small_board_list.append(2*small_board_list[-1])
print("Arrange wheat on a plate of 4 squares (list)：{}".format(small_board_list))


# In[2]:


import numpy as np
small_board_ndarray = np.array(small_board_list)
print("Arrange wheat on a plate of 4 squares (ndarray)：{}".format(small_board_ndarray.reshape(2, 2)))



# In[3]:


small_board_ndarray.reshape(2, 2)


# In[ ]:


Displaying the wheat across the chess board


# In[4]:


n_squares = 64
small_board_list = [1]
for _ in range(n_squares - 1):
    small_board_list.append(2*small_board_list[-1])
print("Display the wheat across the 64 squares (list)：{}".format(small_board_list))


# In[5]:


small_board_ndarr = np.array(small_board_list).astype(np.uint64)
small_board_ndarr = small_board_ndarr.reshape(8, 8)
print("Arrange wheat on 4 squares (ndarray)：{}".format(small_board_ndarr))


# In[ ]:


[Problem 1] Number of wheat on a 2 x 2 square chess board


# In[10]:


import numpy as np
n_squares = 4
small_board_list = [1]
for _ in range(n_squares - 1):
    small_board_list.append(2*small_board_list[-1])
#print("Arrange wheat on a plate of 4 squares (list)：{}".format(small_board_list))

# modify the shape of the ndarray created with the sample code
small_board_ndarray = np.array(small_board_list).reshape(2,2)
print("The number of wheat on a 2/2 squares (ndarray)：\n{}".format(small_board_ndarray))


# In[11]:


#create a two dimensional array using the zero function
ndarray = np.zeros((2, 2))
ndarray


# In[12]:


# placing the values
#method one
ndarray[0][0] = 1
ndarray[0][1] = 2
ndarray[1][0] = 4
ndarray[1][1] = 8
print(ndarray)


# In[13]:


#defining a function
def num_of_wheat(n, m):
    chess_square = n*m
    chessBoardlist = [1]
    for square in range (chess_square-1):
        chessBoardlist.append(2*chessBoardlist[square])
        
    return np.array(chessBoardlist).reshape(n,m).astype(np.uint64)
print("Number of wheat on a 8/8 chess board is: ")
print(num_of_wheat(8,8))


# In[ ]:


Total number of wheat


# In[14]:


chessboard_ndarray = num_of_wheat(8, 8)
total_wheat = chessboard_ndarray.sum()
print("The total number of wheat: ", total_wheat)


# In[15]:


#using the function in problem 2 to find the average
average_wheat = num_of_wheat(8,8).mean(axis=0)
print(average_wheat)


# In[18]:


#GRAPH


# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("column")
plt.ylabel("number")
plt.title("number in each column")
plt.bar(np.arange(1,9), average_wheat)
plt.show()


# In[20]:


n_squares = 64
small_board_list = [1]
for _ in range(n_squares - 1):
    small_board_list.append(2*small_board_list[-1])
print(small_board_list)


# In[21]:


# plotting a Heat map
plt.xlabel("column")
plt.ylabel("row")
plt.title("heatmap")
plt.pcolor(np.array(small_board_list).reshape(8, 8))
plt.show()


# In[ ]:


PROBLEM 5


# In[22]:


#creating a data frame
whole_chessboard = num_of_wheat(8, 8)
#splicing the data
first_half = whole_chessboard[0:4]
second_half = whole_chessboard[4:8]
#printing the result
print("...........First half of the chess board..........")
print(first_half)
print('n/')
print("...........Second half of the chess board..........")
print(second_half)


# In[32]:


sum_firsHalf = first_half.sum()
sum_secondHalf = second_half.sum()
print("first half: ", sum_firsHalf, "\nsecond half: ",sum_secondHalf)
num_times = sum_secondHalf / sum_firsHalf
print("\nThe second half is {} times longer than the first half".format(num_times))


# In[ ]:


PROBLLEM 6


# In[24]:


def num_of_wheat_append(n, m):
    chess_square = n*m
    chessBoardlist = np.array([1]).astype(np.uint64)
    for seed in range (chess_square-1):
        chessBoardlist = np.append(chessBoardlist, 2*chessBoardlist[-1])
        
    return chessBoardlist
print("Number of wheat on a 8/8 chess board is: ")
print(num_of_wheat_append(8,8))


# In[23]:


num_squares = 64
indices_of_squares = np.arange(n_squares).astype(np.uint64)
board_broadcast = 2**indices_of_squares
print("Number of wheat in the last trout：{}".format(board_broadcast)) 


# In[26]:


#Problem 7 Comparing calculation times


# In[27]:


get_ipython().run_cell_magic('timeit', '', 'num_of_wheat\n')


# In[30]:


get_ipython().run_cell_magic('timeit', '', 'num_of_wheat_append\n\n')


# In[31]:


get_ipython().run_cell_magic('timeit', '', 'board_broadcast\n')


# In[ ]:


From the output above, broadcast is faster than the append and function list methods.

