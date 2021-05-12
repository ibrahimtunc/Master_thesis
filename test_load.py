# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:51:10 2021

@author: Ibrahim Alperen Tunc
"""

import wef_helper_functions as hlp

directory = r'D:\ALPEREN\Tübingen NB\Semester 4\Thesis\git\codes\data'
flm = hlp.file_management(directory, globals(), dir())
flm.load_file('test')