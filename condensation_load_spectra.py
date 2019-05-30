# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:42:37 2019

@author: Nathan G. Brugnone
"""
#%%

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import cycler

## Parameters & Settings
case = 4 # 0: Barbell; 1: Tree; 2: Gaussian Mix; 3/4: Hyperuniform Circle/Ellipse
vname = 'nips' # spectial tag for video file
colormap = cm.viridis # cmap for viz
plot_type = 'val' # 'vec': right singular vec embed; 'val': singula val; '3d': 3D
save = True # Save figure?
save_type = '.pdf'
#double_eps = False # True: Viz; False: Spectra
psi_min = 2 # min right singular vector to plot; 2, 3, ..., 11
psi_max = psi_min + 1 # max right singular vector to plot
psi3d_min = 2 # 2, 3, ..., 11
psi3d_mid = psi3d_min + 1
psi3d_max = psi3d_min + 2
#sig = 2 # 2, 3, ..., 11 ## sigma_{i}
vname = 'nips'

for sig in np.arange(2,16):
    
    if case == 0:
        fname = 'barbell'
        plot_title = 'Barbell'
    elif case == 1:
        fname = 'tree'
        plot_title = 'Tree'
    elif case == 2:
        fname = 'gauss'
        plot_title = 'Gauss'
    elif case == 3:
        fname = 'hyperuni_circle'
        plot_title = 'Hyperuniform Circle'
    elif case == 4:
        fname = 'hyperuni_ellipse'
        plot_title = 'Hyperuniform Ellipse'
        
    fname += vname
        
    # Get # of Iterations
    iter_name = 'dm/iterations_'+fname+'.npy'
    iterations = np.load(iter_name)
    
    # Get Diffusion Maps Spectra
    eDM_name = 'dm/E_'+fname+'.npy'; eDM_sig = np.load(eDM_name)[sig - 2]
    
    # Initialize Specra Lists
    ei = []; et = []; eDM = []
    
    # Get Epsilon (shape = (2, #iterations), 0th axis #eps doublings, 1st axis eps)
    eps_name = 'dm/epsilon_list_'+fname+'.npy'; eps_adjust = np.load(eps_name)
    
    # Get Number of Points in Dataset & Color
    datasize_name = 'dm/V_'+fname+'.npy'; N = np.load(datasize_name).shape[0]
    C_name = 'dm/C_'+fname+'.npy'; C = np.load(C_name)
    
    #%%
    for i in np.arange(1, 1+iterations):
        '''Singular Values (DM for Changing Data, TCDM) & Eigenvalues (DM)'''
        pi_name = 'p_i/Ei_'+str(i)+'_'+fname+'.npy'
        pt_name = 'p_t/Et_'+str(i)+'_'+fname+'.npy'
        
        ei.append([i, np.load(pi_name)[sig - 2]]) # Operator P_i
        et.append([i, np.load(pt_name)[sig - 2]]) # Composed Operator P^((t))
        eDM.append([i, eDM_sig**i]) # Diffusion Maps Operator P^{t}
    #    eDM.append([int(i - (2**eps_adjust[0,i-1])), eDM_sig**i]) # Diffusion Maps Operator P^{t}
    
    
    if plot_type == 'val':
#        n = 2
#        color = plt.cm.viridis(np.linspace(0, 1, n))
#        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
        
#        fig, (ax1, ax2) = plt.subplots(nrows=2)
        
#        plt.title(r"$\sigma_{"+str(sig)+"}$ of $P_{\epsilon}^{t}$, $P_{\epsilon,i}$, and $P_{\epsilon}^{(t)}$ for "+plot_title)
    #    plt.plot(np.asarray(eDM).T[0], np.asarray(eDM).T[1], marker='o', label=r'$P_{\epsilon}^{t}$')
#        plt.subplot(211)
        plt.plot(np.asarray(ei).T[0], np.asarray(ei).T[1], marker='o', label=r'$P_{\epsilon,i}$', color = 'c')
        plt.ylabel(r"$\sigma_{k}$")
#        plt.subplot(212)
#        plt.plot(np.asarray(et).T[0], np.asarray(et).T[1], marker='o', label=r'$P_{\epsilon}^{(t)}$', color = 'y')
        plt.xlabel("Iteration")
#        plt.ylabel(r"$\sigma_{k}$")
#        plt.ylabel(r"$\sigma_{"+str(sig)+"}$")
#        plt.legend()
        plt.show()
        
        save_dir = 'figs/spectral/nips/' 
        save_name = fname+'_sigma'+str(sig)+'_N-'+str(N)
    
    
    elif plot_type == 'vec':
        # Set Singular Vectors
        psiDM_name = 'p_i/Vi_1_'+fname+'.npy'
        psiDM = np.load(psiDM_name)
        
        # Generate Figure
        plt.figsize=(10, 10)
    
        plt.title(r'$\psi_{'+str(psi_min)+'}$ & $\psi_{'+str(psi_max)+'}$ for '+plot_title+' (N = '+str(N)+')')
        plt.scatter(psiDM[:,psi_min-2], psiDM[:,psi_max-2], c=C, cmap = colormap,
                    vmin=np.amin(C), vmax=np.amax(C), label=r'$P_{\epsilon}^{t}$')
        plt.xlabel(r'$\psi_{'+str(psi_min)+'}$')
        plt.ylabel(r'$\psi_{'+str(psi_max)+'}$')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                        labelbottom=False, right=False, left=False, labelleft=False)
        plt.show()
        
        save_dir = 'figs/embed/'
        save_name = plot_title+'_psi'+str(psi_min)+'_psi'+str(psi_max)+'_N-'+str(N)
    
    
    elif plot_type == '3d':
        
        psiDM_name = 'p_i/Vi_1_'+fname+'.npy'
        psiDM = np.load(psiDM_name)
            
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        plt.title(r'$\psi_{'+str(psi3d_min)+'}$, $\psi_{'+str(psi3d_mid)+
                           '}$, & $\psi_{'+str(psi3d_max)+'}$ Embedding of '
            +plot_title+' (N = '+str(N)+')')
        ax.scatter(psiDM[:, psi3d_min - 2], psiDM[:, psi3d_mid - 2], 
                   psiDM[:, psi3d_max - 2], c=C, cmap=colormap)
        #ax.axes('off')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                            labelbottom=False, right=False, left=False, labelleft=False)
        plt.show()
    
    # Save Figure
    if save == True:
        save_name = save_dir+save_name+save_type
        plt.savefig(save_name, bbox_inches='tight', transparent=True, dpi=300)