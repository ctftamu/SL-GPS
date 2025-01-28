# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 19:54:38 2022

@author: agnta
"""
from pyMARS_main.pymars.drgep import create_drgep_matrix_new
from pyMARS_main.pymars.drgep import get_importance_coeffs

def reduce_drgep_short(solution, Ts, Ps, Xs, target_species, threshold, species_safe):
    matrices = []
    for i in range(len(Ts)):
        matrices.append(create_drgep_matrix_new((Ts[i], Ps[i], Xs[i]), solution))
    coeffs = get_importance_coeffs(solution.species_names, target_species, matrices)
    species_kept = [sp for sp in solution.species_names
                       if coeffs[sp] > threshold 
                       or sp in species_safe
                       ]
    return species_kept
    
        
