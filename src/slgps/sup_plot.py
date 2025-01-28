# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 18:45:41 2021

@author: agnta
"""
import matplotlib.pyplot as plt
import pickle
import os
with open('results_data/t1310_e0.65_h1.1_a0.001', 'rb') as results_file:
    sup_results = pickle.load(results_file)

# with open('results_data/t1310_e1.35_h1.1_a0.001', 'rb') as results_file:
#     sup_results = pickle.load(results_file)
    
# with open('results_data/t1310_e0.65_h1.1_a0.001', 'rb') as results_file:
#     sup_results = pickle.load(results_file)
    
# with open('results_data/t1505_e1.05_h1.05_a0.05', 'rb') as results_file:
#     sup_results = pickle.load(results_file)



det_times = sup_results['det_times']
det_temps = sup_results['det_temps']
det_CH4 = sup_results['det_CH4']
det_CH2O = sup_results['det_CH2O']
det_CO2 = sup_results['det_CO2']
det_CO = sup_results['det_CO']
det_OH = sup_results['det_OH']
det_CH = sup_results['det_CH']
det_O = sup_results['det_O']
det_H = sup_results['det_H']
det_CH3 = sup_results['det_CH3']
det_H2O = sup_results['det_H2O']
det_HRR = sup_results['det_HRR']
const_times = sup_results['const_times']
const_temps = sup_results['const_temps']
const_CH4 = sup_results['const_CH4']
const_CH2O = sup_results['const_CH2O']
const_CO2 = sup_results['const_CO2']
const_CO = sup_results['const_CO']
const_OH = sup_results['const_OH']
const_CH = sup_results['const_CH']
const_O = sup_results['const_O']
const_H = sup_results['const_H']
const_CH3 = sup_results['const_CH3']
const_H2O = sup_results['const_H2O'] 
const_HRR = sup_results['const_HRR']
ad_times = sup_results['ad_times']
ad_temps = sup_results['ad_temps']
ad_CH4 = sup_results['ad_CH4']
ad_CH2O = sup_results['ad_CH2O']
ad_CO2 = sup_results['ad_CO2']
ad_CO = sup_results['ad_CO']
ad_OH = sup_results['ad_OH']
ad_CH = sup_results['ad_CH']
ad_O = sup_results['ad_O']
ad_H = sup_results['ad_H']
ad_CH3 = sup_results['ad_CH3']
ad_H2O = sup_results['ad_H2O']
ad_HRR = sup_results['ad_HRR']
ad_pred_times = sup_results['ad_pred_times']
ad_pred_temps = sup_results['ad_pred_temps']
ad_pred_CH4 = sup_results['ad_pred_CH4']
ad_pred_CH2O = sup_results['ad_pred_CH2O']
ad_pred_CO2 = sup_results['ad_pred_CO2']
ad_pred_CO = sup_results['ad_pred_CO']
ad_pred_OH = sup_results['ad_pred_OH']
ad_pred_CH = sup_results['ad_pred_CH']
ad_pred_O = sup_results['ad_pred_O']
ad_pred_H = sup_results['ad_pred_H']
ad_pred_CH3 = sup_results['ad_pred_CH3']
ad_pred_H2O = sup_results['ad_pred_H2O']
ad_pred_HRR = sup_results['ad_pred_HRR']
det_start = sup_results['det_start']
det_end = sup_results['det_end']
const_start = sup_results['const_start']
const_end = sup_results['const_end']
ad_start = sup_results['ad_start']
ad_end = sup_results['ad_end']
ad_pred_start = sup_results['ad_pred_start']
ad_pred_end = sup_results['ad_pred_end']
t_steps = sup_results['t_steps']
t_pred_steps = sup_results['t_pred_steps']
const_n_spec = sup_results['const_n_spec']
const_n_rxn = sup_results['const_n_rxn']
n_ad_species = sup_results['n_ad_species']
n_ad_reactions = sup_results['n_ad_reactions']
n_ad_pred_species = sup_results['n_ad_pred_species']
n_ad_pred_reactions = sup_results['n_ad_pred_reactions']
det_specs = sup_results['det_specs']
const_specs = sup_results['const_specs']
always_specs = sup_results['always_specs']
var_specs = sup_results['var_specs']
pred_specs = sup_results['pred_specs']

new_det_times = det_times.copy()
for i in range(len(det_times)):
    new_det_times[i]*=1000
new_const_times = const_times.copy()
for i in range(len(const_times)):
    new_const_times[i]*=1000
new_ad_times = ad_times.copy()
for i in range(len(ad_times)):
    new_ad_times[i]*=1000
new_ad_pred_times = ad_pred_times.copy()
for i in range(len(ad_pred_times)):
    new_ad_pred_times[i]*=1000
new_t_steps = t_steps.copy()
for i in range(len(t_steps)):
    new_t_steps[i]*=1000
new_t_pred_steps = t_pred_steps.copy()
for i in range(len(t_pred_steps)):
    new_t_pred_steps[i]*=1000
    

legend_size=14
plt.rcParams['font.size'] = '30'
plt.rcParams["figure.figsize"] = (20,20)

# plt.plot(det_times, auto_ign_det['temperature'], 'r-')
f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, sharex=False)
f.tight_layout(pad=3.0)

ax1.plot(new_det_times[det_start:det_end], det_temps[det_start:det_end], 'r-')
ax1.plot(new_const_times[const_start:const_end], const_temps[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax1.plot(new_ad_times[ad_start:ad_end], ad_temps[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax1.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_temps[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax1.set(xlabel='Time (ms)', ylabel = 'Temperature (K)', title='Temperature vs Time')
# ax1.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax1.locator_params(axis='x', nbins=5)

#plt.savefig(os.path.join('chem_plots', 'basic14.png'))

ax2.plot(new_det_times[det_start:det_end], det_CH4[det_start:det_end], 'r-')
ax2.plot(new_const_times[const_start:const_end], const_CH4[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax2.plot(new_ad_times[ad_start:ad_end], ad_CH4[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax2.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CH4[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax2.set(xlabel='Time (ms)', ylabel=r'$CH_4$ Mole Fraction', title=r'$CH_4$ Mole Fraction vs Time')
# ax2.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax2.locator_params(axis='x', nbins=5)

#plt.savefig(os.path.join('chem_plots', 'basic13.png'))


ax11.plot(new_det_times[det_start:det_end], det_CH2O[det_start:det_end], 'r-')
ax11.plot(new_const_times[const_start:const_end], const_CH2O[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax11.plot(new_ad_times[ad_start:ad_end], ad_CH2O[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax11.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CH2O[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax11.set(xlabel='Time (ms)', ylabel=r'$CH_2O$ Mole Fraction', title=r'$CH_2O$ Mole Fraction vs Time')
# ax11.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax11.locator_params(axis='x', nbins=5)

#plt.savefig(os.path.join('chem_plots', 'basic12.png'))
# ax3.plot(t_steps, n_ad_species, 'bo-')
# ax3.plot(t_steps, [const_n_spec]*len(t_steps), 'go--')
# ax3.plot(t_pred_steps, n_ad_pred_species, 'mo-')
# ax3.set(xlabel='Time (s)', ylabel='Number of Species', title='Number of Species Used vs. Time')
# ax3.legend(['Adaptive', 'Constant Skeletal', 'Supervised Learning'], prop={'size': legend_size})

# ax4.plot(t_steps, n_ad_reactions, 'bo-')
# ax4.plot(t_steps, [const_n_rxn]*len(t_steps), 'go--')
# ax4.plot(t_pred_steps, n_ad_pred_reactions, 'mo-')
# ax4.set(xlabel='Time (s)', ylabel='Number of Reactions', title='Number of Reactions Used vs. Time')
# ax4.legend(['Adaptive', 'Constant Skeletal', 'Supervised Learning'], prop={'size': legend_size})


ax3.plot(new_det_times[det_start:det_end], det_CO2[det_start:det_end], 'r-')
ax3.plot(new_const_times[const_start:const_end], const_CO2[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax3.plot(new_ad_times[ad_start:ad_end], ad_CO2[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax3.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CO2[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax3.set(xlabel='Time (ms)', ylabel=r'$CO_2$ Mole Fraction', title=r'$CO_2$ Mole Fraction vs Time')
# ax3.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax3.locator_params(axis='x', nbins=5)

#plt.savefig(os.path.join('chem_plots', 'basic1.png'))

ax4.plot(new_det_times[det_start:det_end], det_CO[det_start:det_end], 'r-')
ax4.plot(new_const_times[const_start:const_end], const_CO[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax4.plot(new_ad_times[ad_start:ad_end], ad_CO[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax4.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CO[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax4.set(xlabel='Time (ms)', ylabel=r'$CO$ Mole Fraction', title=r'$CO$ Mole Fraction vs Time')
# ax4.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax4.locator_params(axis='x', nbins=5)

#plt.savefig(os.path.join('chem_plots', 'basic2.png'))

ax5.plot(new_det_times[det_start:det_end], det_OH[det_start:det_end], 'r-')
ax5.plot(new_const_times[const_start:const_end], const_OH[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax5.plot(new_ad_times[ad_start:ad_end], ad_OH[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax5.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_OH[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax5.set(xlabel='Time (ms)', ylabel=r'$OH$ Mole Fraction', title=r'$OH$ Mole Fraction vs Time')
# ax5.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax5.locator_params(axis='x', nbins=5)

#plt.savefig(os.path.join('chem_plots', 'basic3.png'))

ax6.plot(new_det_times[det_start:det_end], det_O[det_start:det_end], 'r-')
ax6.plot(new_const_times[const_start:const_end], const_O[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax6.plot(new_ad_times[ad_start:ad_end], ad_O[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax6.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_O[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax6.set(xlabel='Time (ms)', ylabel=r'$O$ Mole Fraction', title=r'$O$ Mole Fraction vs Time')
# ax6.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax6.locator_params(axis='x', nbins=5)

#plt.savefig(os.path.join('chem_plots', 'basic4.png'))

ax7.plot(new_det_times[det_start:det_end], det_H[det_start:det_end], 'r-')
ax7.plot(new_const_times[const_start:const_end], const_H[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax7.plot(new_ad_times[ad_start:ad_end], ad_H[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax7.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_H[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax7.set(xlabel='Time (ms)', ylabel=r'$H$ Mole Fraction', title=r'$H$ Mole Fraction vs Time')
# ax7.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax7.locator_params(axis='x', nbins=5)

#plt.savefig(os.path.join('chem_plots', 'basic5.png'))

ax8.plot(new_det_times[det_start:det_end], det_H2O[det_start:det_end], 'r-')
ax8.plot(new_const_times[const_start:const_end], const_H2O[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax8.plot(new_ad_times[ad_start:ad_end], ad_H2O[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax8.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_H2O[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax8.set(xlabel='Time (ms)', ylabel=r'$H_2O$ Mole Fraction', title=r'$H_2O$ Mole Fraction vs Time')
# ax8.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax8.locator_params(axis='x', nbins=5)

#plt.savefig(os.path.join('chem_plots', 'basic6.png'))

ax9.plot(new_det_times[det_start:det_end], det_CH3[det_start:det_end], 'r-')
ax9.plot(new_const_times[const_start:const_end], const_CH3[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax9.plot(new_ad_times[ad_start:ad_end], ad_CH3[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax9.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CH3[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax9.set(xlabel='Time (ms)', ylabel=r'$CH_3$ Mole Fraction', title=r'$CH_3$ Mole Fraction vs Time')
# ax9.legend(['Detailed',  'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax9.locator_params(axis='x', nbins=5)

#plt.savefig(os.path.join('chem_plots', 'basic7.png'))

ax10.plot(new_det_times[det_start:det_end], det_CH[det_start:det_end], 'r-')
if 'CH' in const_specs:
    ax10.plot(new_const_times[const_start:const_end], const_CH[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax10.plot(new_ad_times[ad_start:ad_end], ad_CH[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax10.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_CH[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax10.set(xlabel='Time (ms)', ylabel=r'$CH$ Mole Fraction', title=r'$CH$ Mole Fraction vs Time')
# if 'CH' in const_specs:
#     ax10.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
# else:
#     ax10.legend(['Detailed', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size})
ax10.locator_params(axis='x', nbins=5)
ax10.yaxis.get_offset_text().set_position((-0.15,6))

#plt.savefig(os.path.join('chem_plots', 'basic8.png'))

ax12.plot(new_det_times[det_start:det_end], det_HRR[det_start:det_end], 'r-')
ax12.plot(new_const_times[const_start:const_end], const_HRR[const_start:const_end], 's', markerfacecolor="None",
         markeredgecolor='green', ms=8)
ax12.plot(new_ad_times[ad_start:ad_end], ad_HRR[ad_start:ad_end], 'o', markerfacecolor="None",
         markeredgecolor='blue', ms=8)
ax12.plot(new_ad_pred_times[ad_pred_start:ad_pred_end], ad_pred_HRR[ad_pred_start:ad_pred_end], '^', markerfacecolor="None",
         markeredgecolor='purple', ms=8)
ax12.set(xlabel='Time (ms)', ylabel=r'Heat Release Rate ($J/m^3*s$)', title='Heat Release Rate vs Time')
ax12.set_yscale('log')
ax12.legend(['Detailed', 'Constant Skeletal', 'Adaptive Skeletal', 'SL Skeletal'], prop={'size': legend_size}, bbox_to_anchor=(-2.0, -0.2), markerscale=3)
ax12.locator_params(axis='x', nbins=5)

plt.savefig(os.path.join('chem_plots', 'specieAccuracy.png'))
f.show()


plt.rcParams['font.size'] = '18'
plt.rcParams["figure.figsize"] = (10,8)
g, (ax3, ax4) = plt.subplots(2, 1)

ax3.plot(new_t_steps, n_ad_species, 'bo-')
ax3.plot(new_t_steps, [const_n_spec]*len(t_steps), 'go--')
ax3.plot(new_t_pred_steps, n_ad_pred_species, 'mo-')
ax3.set(xlabel='Time (ms)', ylabel='Number of Species', title='Number of Species Used vs. Time')
ax3.legend(['Adaptive', 'Constant Skeletal', 'Supervised Learning'], prop={'size': 12})

#plt.savefig(os.path.join('chem_plots', 'basic10.png'))

ax4.plot(new_t_steps, n_ad_reactions, 'bo-')
ax4.plot(new_t_steps, [const_n_rxn]*len(t_steps), 'go--')
ax4.plot(new_t_pred_steps, n_ad_pred_reactions, 'mo-')
ax4.set(xlabel='Time (ms)', ylabel='Number of Reactions', title='Number of Reactions Used vs. Time')
ax4.legend(['Adaptive', 'Constant Skeletal', 'Supervised Learning'], prop={'size': 12})
g.tight_layout()
g.show()

plt.savefig(os.path.join('chem_plots', 'reacSpecieCount.png'))
