import numpy as np
from matplotlib import pyplot as plt
from itertools import 

nsite = 14
nalpha = nsite//2
nbeta = nsite - nalpha

basis_map = {}
for icomb, comb in enumerate(combs(range(14), 7)):
    string = ['1']*nsite
    for i in comb: string[nsite-1-i] = '0'
    self.map[''.join(string)] = icomb
