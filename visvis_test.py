#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 19:45:03 2021

@author: thom
"""

import numpy as np
import visvis as vv
app = vv.use()

f = vv.clf()
a = vv.cla()

angle = np.linspace(0, 6*np.pi, 1000)
x = np.sin(angle)
y = np.cos(angle)
z = angle / 6.0
vv.plot(x, y, z, lw=10)

angle += np.pi*2/3.0
x = np.sin(angle)
y = np.cos(angle)
z = angle / 6.0 - 0.5
vv.plot(x, y, z, lc ="r", lw=10)

vv.screenshot('figure1.jpg', sf=3, bg='w') # sf: scale factor

app.Run()