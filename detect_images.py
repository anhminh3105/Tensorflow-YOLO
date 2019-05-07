
# coding: utf-8

# In[ ]:


import cv2
import time
from yolow import Yolow
from imager import *


# In[ ]:


imer = Imager()


# In[ ]:


yl = Yolow()


# In[ ]:


imer.imset_from_path('data/images/')
input_list = imer.preproces()


# In[ ]:


start = time.time()
pred_list = yl.predict(input_list)
print('prediction takes {}s'.format(time.time() - start))


# In[ ]:


ims = imer.visualise_preds(pred_list)


# In[ ]:


# cv2.imshow('Test Image', ims[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[ ]:


imer.imsave(ims)

