#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Create folders to store figures
import pathlib
pathlib.Path('./Figure').mkdir()

for k in range(10):
    pathname = "./Figure/Figure{}".format(k+1)
    if k!= 9:
        pathlib.Path(pathname).mkdir()
    else:
        pathlib.Path("./Figure/Supportive_fig").mkdir()


# In[ ]:




