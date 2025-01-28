# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:15:13 2022

@author: agnta
"""
from pyMARS_main.pymars.drgep import run_drgep
from pyMARS_main.pymars.sampling import InputIgnition
import cantera as ct





def drgep_reduce(mech_file, temp, pressure, phi, fuel, oxid, error_limit, spec_safe=[]):

    init_conds = [InputIgnition(kind='constant pressure', pressure=pressure, temperature=temp, end_time=10.0, fuel=fuel, oxidizer=oxid, equivalence_ratio=phi)]
    red_mech = run_drgep(model_file=mech_file, ignition_conditions=init_conds, psr_conditions=None, flame_conditions=None, error_limit=error_limit, species_safe=spec_safe, species_targets=[fuel], path='drgep_mechs')
    return ct.Solution(red_mech.filename)


# init_conds = [InputIgnition(kind='constant pressure', pressure=1.1, temperature=1310, end_time=20.0, fuel={'CH4':1.0}, oxidizer={'O2': 1.0, 'N2':3.76}, equivalence_ratio=0.65)]

# red_mech = run_drgep(model_file='GRI30_cantera/gri30.cti', ignition_conditions=init_conds, psr_conditions=None, flame_conditions=None, error_limit=0.1, species_safe=[], species_targets=['CO2', 'H2O'])