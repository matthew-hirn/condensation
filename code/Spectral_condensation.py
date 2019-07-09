# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:42:37 2019

@author: Nathan G. Brugnone
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import cycler

def spectral_decay(case = 4,
                   vname = 'example_0',
                   plot_type = 'val',
                   save = False):
    
    ## Parameters & Settings
    # 0: Barbell
    # 1: Tree
    # 2: Gaussian Mix
    # 3: Hyperuniform Circle
    # 4: Hyperuniform Ellipse
    ## Parameters & Settings
    # case = 4 # 0: Barbell; 1: Tree; 2: Gaussian Mix; 3/4: Hyperuniform Circle/Ellipse
    # vname = 'example_0' # spectial tag for video file
    colormap = cm.viridis # cmap for viz
    # plot_type = 'val' # 'vec': right singular vec embed; 'val': singula val; '3d': 3D
    # save = False # Save figure?
    save_type = '.pdf'
    psi_min = 2 # min right singular vector to plot; 2, 3, ..., 11
    psi_max = psi_min + 1 # max right singular vector to plot
    psi3d_min = 2 # 2, 3, ..., 11
    psi3d_mid = psi3d_min + 1
    psi3d_max = psi3d_min + 2
    
    for sig in np.arange(2,12):
        
        if case == 0:
            fname = 'barbell'
            cdir = 'barbell/'
            plot_title = 'Barbell'
        elif case == 1:
            fname = 'tree'
            cdir = 'tree/'
            plot_title = 'Tree'
        elif case == 2:
            fname = 'gauss'
            cdir = 'gauss/'
            plot_title = 'Gauss'
        elif case == 3:
            fname = 'hyperuni_circle'
            cdir = 'h_circle/'
            plot_title = 'Hyperuniform Circle'
        elif case == 4:
            fname = 'hyperuni_ellipse'
            cdir = 'ellipse/'
            plot_title = 'Hyperuniform Ellipse'
            
        sname = fname + '_' + vname # save name tag    
        fname += vname # load name tag
            
        # Get # of Iterations
        iter_name = 'dm/'+cdir+'iterations_'+fname+'.npy'
        iterations = np.load(iter_name)
        
        # Get Diffusion Maps Spectra
        eDM_name = 'dm/'+cdir+'E_'+fname+'.npy'; eDM_sig = np.load(eDM_name)[sig - 2]
        
        # Initialize Specra Lists
        ei = []; et = []; eDM = []
        
        # Get Epsilon (shape = (2, #iterations), 0th axis #eps doublings, 1st axis eps)
        eps_name = 'dm/'+cdir+'epsilon_list_'+fname+'.npy'; eps_adjust = np.load(eps_name)
        
        # Get Number of Points in Dataset & Color
        datasize_name = 'dm/'+cdir+'V_'+fname+'.npy'; N = np.load(datasize_name).shape[0]
        C_name = 'dm/'+cdir+'C_'+fname+'.npy'; C = np.load(C_name)
        
        #%%
        for i in np.arange(1, 1+iterations):
            '''Singular Values (DM for Changing Data, TCDM) & Eigenvalues (DM)'''
            pi_name = 'p_i/'+cdir+'Ei_'+str(i)+'_'+fname+'.npy'
            pt_name = 'p_t/'+cdir+'Et_'+str(i)+'_'+fname+'.npy'
            
            ei.append([i, np.load(pi_name)[sig - 2]]) # Operator P_i
            et.append([i, np.load(pt_name)[sig - 2]]) # Composed Operator P^((t))
            eDM.append([i, eDM_sig**i]) # Diffusion Maps Operator P^{t}    
        
        if plot_type == 'val':
    
            plt.subplot(311)
            plt.plot(np.asarray(ei).T[0], np.asarray(ei).T[1], marker='o', label=r'$P_{\epsilon,i}$', color = 'c')
            plt.subplot(312)
            plt.plot(np.asarray(et).T[0], np.asarray(et).T[1], marker='o', label=r'$P_{\epsilon}^{(t)}$', color = 'purple')
            plt.ylabel(r"$\sigma_{k}$")
            plt.subplot(313)
            plt.plot(np.asarray(eDM).T[0], np.asarray(eDM).T[1], marker='o', label=r'$P_{\epsilon}^{t}$', color = 'g')
            plt.xlabel("Iteration")
#            plt.show()
            
            save_dir = 'figs/spectral/'+cdir 
            save_name = sname+'_sigma'+str(sig)+'_N-'+str(N)
        
        
        elif plot_type == 'vec':
                # Set Singular Vectors
                psiDM_name = 'p_i/'+cdir+'Vi_1_'+fname+'.npy'
                psiDM = np.load(psiDM_name)
                
                # Generate Figure
                plt.title(r'$\psi_{'+str(psi_min)+'}$ & $\psi_{'+str(psi_max)+'}$ for '+plot_title+' (N = '+str(N)+')')
                plt.scatter(psiDM[:,psi_min-2], psiDM[:,psi_max-2], c=C, cmap = colormap,
                            vmin=np.amin(C), vmax=np.amax(C), label=r'$P_{\epsilon}^{t}$')
                plt.xlabel(r'$\psi_{'+str(psi_min)+'}$')
                plt.ylabel(r'$\psi_{'+str(psi_max)+'}$')
                plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                                labelbottom=False, right=False, left=False, labelleft=False)
                plt.show()
                
                save_dir = 'figs/embed/'+cdir
                save_name = sname+'_psi'+str(psi_min)+'_psi'+str(psi_max)+'_N-'+str(N)
            
            
        elif plot_type == '3d':
            # Set Singular Vectors and Plot
            from mpl_toolkits.mplot3d import Axes3D
            psiDM_name = 'p_i/'+cdir+'Vi_1_'+fname+'.npy'
            psiDM = np.load(psiDM_name)
                
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            plt.title(r'$\psi_{'+str(psi3d_min)+'}$, $\psi_{'+str(psi3d_mid)+
                               '}$, & $\psi_{'+str(psi3d_max)+'}$ Embedding of '
                +plot_title+' (N = '+str(N)+')')
            ax.scatter(psiDM[:, psi3d_min - 2], psiDM[:, psi3d_mid - 2], 
                       psiDM[:, psi3d_max - 2], c=C, cmap=colormap)
            
            ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                                labelbottom=False, right=False, left=False, labelleft=False)
            plt.show()
            
            save_dir = 'figs/embed/'+cdir
            save_name = sname+'_psi'+str(psi3d_min)+'_psi'+str(
                    psi3d_mid)+'_psi'+str(psi3d_max)+'_N-'+str(N)
            
    
    
    # Save Figure
    if save == True:
        save_name = save_dir+save_name+save_type
        plt.savefig(save_name, bbox_inches='tight', transparent=True, dpi=300)
        
    return plt.show()
    
    
if "__name__" == "__main__":
    
    spectral_decay()
    
    # Save Figure
    if save == True:
        save_name = save_dir+save_name+save_type
        plt.savefig(save_name, bbox_inches='tight', transparent=True, dpi=300)