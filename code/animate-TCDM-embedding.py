# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 17:37:54 2019

@author: nateb
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
#from mpl_toolkits.mplot3d import Axes3D

## Parameters & Settings
case = 4 # 0: Barbell
# 1: Tree
# 2: Gaussian Mix
# 3: Hyperuniform Circle
# 4: Hyperuniform Ellipse
vname = 'example_0' # tag for video file (vname from condensation_01.py)
colormap = cm.viridis # cmap for viz (viridis default for accessibility)
plot_type = '3d' # '2d' or '3d'
save = False # Save figure?
save_type = '.pdf'
psi_min = 0 # min right singular vector for 2-D embedding; 2, 3, ...va, 11
psi_max = 1 # max right singular vector for 2-D embedding
psi3d_min = 0 # 2, 3, ..., 11 (1st singular vector for 3-D embedding)
psi3d_mid = 1#psi3d_min + 1 # (2nd singular vector for 3-D embedding)
psi3d_max = 2#psi3d_min + 2 # (3rd singular vector for 3-D embedding)

    
#if case == 0:
#    fname = 'barbell'
#    cdir = 'barbell/'
#    plot_title = 'Barbell'
#elif case == 1:
#    fname = 'tree'
#    cdir = 'tree/'
#    plot_title = 'Tree'
#elif case == 2:
#    fname = 'gauss'
#    cdir = 'gauss/'
#    plot_title = 'Gauss'
#elif case == 3:
#    fname = 'hyperuni_circle'
#    cdir = 'h_circle/'
#    plot_title = 'Hyperuniform Circle'
#elif case == 4:
#    fname = 'hyperuni_ellipse'
#    cdir = 'ellipse/'
#    plot_title = 'Hyperuniform Ellipse'
    
if case == 0:
    plot_title = 'Uniformly\,\,Sampled\,\, Barbell'
    fname = 'barbell'
    cdir = 'barbell/'

elif case == 1:
    plot_title = 'Uniformly\,\, Sampled\,\, Tree'
    fname = 'tree'
    cdir = 'tree/'

elif case == 2:
    plot_title = 'Gaussian\,\, Blobs'
    fname = 'gauss'
    cdir = 'gauss/'

elif case == 3:
    plot_title ='Hyperuniformly\,\, Sampled\,\, Circle'
    fname = 'hyperuni_circle'
    cdir = 'h_circle/'

elif case == 4:
    plot_title = 'Hyperuniformly\,\, Sampled\,\, Ellpse'
    fname = 'hyperuni_ellipse'
    cdir = 'ellipse/'

elif case == 5:
    plot_title = 'Uniformly\,\,Sampled\,\, Circle'
    fname = 'uni_circle'
    cdir = 'uni_circle/'

elif case == 6:
    plot_title = 'Noisey\,\, Tree'
    fname = 'perturb_tree'
    cdir = 'ptree/'

elif case == 7:
    plot_title = 'Two\,\, Spirals'
    fname = 'spirals'
    cdir = 'spirals/'

sname = fname + '_' + vname # save name tag    
fname += vname # load name tag

# Get # of Iterations
iter_name = 'dm/'+cdir+'iterations_'+fname+'.npy'
iterations = np.load(iter_name)

# Initialize Specra Lists
Vt = []

# Get Epsilon (shape =(2, #iterations), 0th axis #eps doublings, 1st axis eps)
eps_name = 'dm/'+cdir+'epsilon_list_'+fname+'.npy'; eps_adjust = np.load(eps_name)

# Get Number of Points in Dataset & Color
datasize_name = 'dm/'+cdir+'V_'+fname+'.npy'; N = np.load(datasize_name).shape[0]
C_name = 'dm/'+cdir+'C_'+fname+'.npy'; C = np.load(C_name)


for i in np.arange(1, 1+iterations):
    '''Singular Values (DM for Changing Data, TCDM) & Eigenvalues (DM)'''
#    pi_name = 'p_i/'+cdir+'Vi_'+str(i)+'_'+fname+'.npy'
    pt_name = 'p_t/'+cdir+'Vt_'+str(i)+'_'+fname+'.npy'
    
#    Vt.append(np.load(pi_name)) # Operator P_i
    Vt.append(np.load(pt_name)) # Composed Operator P^(t)

Vt = np.asarray(Vt)
print(Vt.shape)
#%%

if plot_type == '2d':
    ## Generate Animations & Still Frames
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                   labelbottom=False, right=False, left=False, labelleft=False)
    # Init Scatter Plot
    scat = ax.scatter(Vt[0,:,psi_min].T, Vt[0,:,psi_max].T, c = C, s = 10*np.ones(Vt.shape[0]), 
                      cmap = colormap, vmin=np.amin(C), vmax=np.amax(C))
    
    def animate(i):
        '''Animate Density-Dependent Condensation + Gradient Only'''
    #    em = str(int(eps_num*2**epsilon_list[0, i])) # epsilon & multiplier
    #    eps_tex = '\\frac{'+str(em)+'\pi}{'+str(eps_denom)+'N}}'
    #    ax.set_title(r'$\bf{'+ptitle+'}$\nCondensation: '+str(i)+
    #                 ', $\epsilon='+eps_tex+'$') 
        
        scat.set_offsets(np.c_[Vt[i,:,psi_min].T, Vt[0,:,psi_max].T])
    #    scat.set_sizes(S[i,:].T)
    
    anim = FuncAnimation(fig, animate, interval=50, frames=iterations)
 
    
elif plot_type == '3d':
        # Set Singular Vectors and Plot
        from mpl_toolkits.mplot3d import Axes3D
#        psiDM_name = 'p_i/'+cdir+'Vi_1_'+fname+'.npy'
#        psiDM = np.load(psiDM_name)
#            
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        plt.title(r'$\psi_{'+str(psi3d_min)+'}$, $\psi_{'+str(psi3d_mid)+
#                           '}$, & $\psi_{'+str(psi3d_max)+'}$ Embedding of '
#            +plot_title+' (N = '+str(N)+')')
#        
#        ax.scatter(psiDM[:, psi3d_min], psiDM[:, psi3d_mid], 
#                   psiDM[:, psi3d_max], c=C, cmap=colormap)
                
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                            labelbottom=False, right=False, left=False, labelleft=False)
        ax.set_title(r'$\psi_{'+str(
                psi3d_min)+'}$, $\psi_{'+str(psi3d_mid)+'}$, & $\psi_{'+str(
                        psi3d_max)+'}$ Embedding of '+plot_title+' (N = '+str(N)+')')
        
        scat = ax.scatter(Vt[0,:,psi3d_min].T, Vt[0,:,psi3d_mid].T, 
                            Vt[0,:,psi3d_max].T, c=C,s=10*np.ones(Vt.shape[0]),
                            cmap = colormap, vmin=np.amin(C), vmax=np.amax(C))
        
        def animate(i):
                     
#            self.scat.set_offsets(data[:,:2])
#            #self.scat.set_3d_properties(data)
#            self.scat.set_3d_properties(data[:,2:],'z')
            
            scat._offsets3d = (Vt[i,:,psi3d_min].T, Vt[i,:,psi3d_mid].T,
                                    Vt[i,:,psi3d_max].T)
#            title.set_text('3D Test, time={}'.format(num))
        
        anim = FuncAnimation(fig, animate, frames=iterations, interval=20)#, blit=False)
#        plt.show()
        
        save_dir = 'figs/embed/'+cdir
        save_name = sname+'_psi'+str(psi3d_min)+'_psi'+str(
                psi3d_mid)+'_psi'+str(psi3d_max)+'_N-'+str(N)
        

# Set up Formatting for Movie Files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Nathan Brugnone'), bitrate=1800)

if save == True:
    anim.save(fname+'_'+str(N)+'pts_'+str(vname)+'.mp4', writer=writer)

plt.draw()
plt.show()
