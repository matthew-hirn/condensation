# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:01:30 2019

@author: Nathan G. Brugnone
"""

import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from scipy import special
from scipy import optimize
from scipy import spatial

# Fixing random state for reproducibility
np.random.seed(2387)

# Initialize Parameters
N = 2**7 # number of samples (2**7, 2**8 used in paper)
eps_num = 200; eps_denom = 1 # numerator/demoninator of following epsilon 
epsilon_small = eps_num*np.pi/(eps_denom*N) # Set epsilon
alpha = 1 # Effect of density in Q**(-alpha)KQ**(-alpha) (0: KEEP; 1: CANCEL)
case = 6 # 0: Barbell 
# 1: Tree
# 2: Gaussian Mix 
# 3: Hyperuniform Circle
# 4: Hyperuniform Ellipse
# 5: Uniform Cicle
# 6: Perturbed Tree (must set standard dev of perturbation via std_dev)
# 7: Two Spirals
perturb_data = False # If True, set std_dev
std_dev = 0.05 # standard deviation of Gaussian perturbation of data
vname = 'example_0' # \filename tag for video and spectral files
pdir = 'movie/' # vid save parent directory; default 'movie/'
colormap = cm.viridis # for viz (viridis default for accessibility)
double_eps_by = 2 # epsilon increase multiplier
double_eps = True # True: Viz; False: Saving DM Spectra
save_spectra = True # True: SVD of P**t, P_i, P**(t) = P_t P_t-1 ... P_1
save_movie = False # save .mp4 of animation? 
save_stills = False # save ."fig_type" stills of animation (w/o plot labels)? 
fig_type = '.pdf' # filetype for animation stills 
frame_list = [0,2,3,5,6,10,15] # frames to save
lratio = 0.99 # ratio of the way through vid of final save frame (1 -> keep last)
spec_keep = 11 # save 1:spec_keep largest singular vecs/vals; \sigma_0 discarded

if case == 0:
    ptitle = 'Uniformly\,\,Sampled\,\, Barbell'
    fname = 'barbell'
    cdir = 'barbell'
    pdir += 'barbell/'
elif case == 1:
    ptitle = 'Uniformly\,\, Sampled\,\, Tree'
    fname = 'tree'
    cdir = 'tree'
    pdir += 'tree/'
elif case == 2:
    ptitle = 'Gaussian\,\, Blobs'
    fname = 'gauss'
    cdir = 'gauss'
    pdir += 'gauss/'
elif case == 3:
    ptitle ='Hyperuniformly\,\, Sampled\,\, Circle'
    fname = 'hyperuni_circle'
    cdir = 'h_circle/'
    pdir += 'h_circle/'
elif case == 4:
    ptitle = 'Hyperuniformly\,\, Sampled\,\, Ellpse'
    fname = 'hyperuni_ellipse'
    cdir = 'ellipse'
    pdir += 'ellipse/'
elif case == 5:
    ptitle = 'Uniformly\,\,Sampled\,\, Circle'
    fname = 'uni_circle'
    cdir = 'uni_circle'
    pdir += 'uni_circle/'
elif case == 6:
    ptitle = 'Noisey\,\, Tree'
    fname = 'perturb_tree'
    cdir = 'ptree'
    pdir += 'ptree/'
elif case == 7:
    ptitle = 'Two\,\, Spirals'
    fname = 'spirals'
    cdir = 'spirals'
    pdir += 'spirals/'

if save_stills == True:
    fdir = pdir+'stills/'

# Synthetic Datasets
def barbell(N, beta=1):
    '''Generate Uniformly-Sampled 2-D Barbell'''
    X = [[],[]] # init data list [[x],[y]] 
    C = [] # init color list for plotting
    k = 1
    while k <= N:
        x = (2 + beta/2)*np.random.uniform()
        y = (2 + beta/2)*np.random.uniform()
        
        if (x - 0.5)**2 + (y - 0.5)**2 <= 0.25:
            X[0].append(x)
            X[1].append(y)
            C.append(0)
            k += 1
            
        elif abs(x - 1 - beta/4) < beta/4 and abs(y - 0.5) < 0.125:
            X[0].append(x)
            X[1].append(y)
            C.append(1)
            k += 1
            
        elif (x - 1.5 - beta/2)**2 + (y - 0.5)**2 <= 0.25:
            X[0].append(x)
            X[1].append(y)
            C.append(2)
            k += 1
            
    return np.asarray(X), np.asarray(C)


def tree(N, radius=1, levels=3):
    '''Generate Uniformly-Sampled 2-D Tree with 2**(levels) Branches'''
    X = [[],[]] # init data list [[x],[y]] 
    C = [] # init color list for plotting
    
    s = 0; root = [s, s] # root node position
    omega = np.pi/4 # half of anlge between branches
    xtop = np.cos(omega); ytop = np.sin(omega)
    xbot = np.cos(-omega); ybot = np.sin(-omega)
    
    for l in range(levels): # nuber of fork nodes
        for n in range(2**l): # quantify branch doubling 
            for i in range(int(N/(levels*2*(2**l)))): # uniform sample
                ## Top branch of current node
                top = np.random.uniform() # top branch sample
                X[0].append(root[0] + radius*top*xtop) # x
                X[1].append(root[1] + radius*top*ytop) # y
                
                ## Bottom branch of current node
                bottom = np.random.uniform() # bottom branch sample
                X[0].append(root[0] + radius*bottom*xbot) # x
                X[1].append(root[1] + radius*bottom*ybot) # y
                C.extend([l,l])
            
            root[1] -= 2*s # decrease y coordinate of root node
            
        root[0] += radius*xtop # increase x to end of current line
        root[1] = radius*ytop # move y to end of currrent line (reset y)
        radius =  radius / 2 # decrease radius
        s = np.sqrt(2)*radius # compute new branch length
        root[1] += n*2*s # set next y coordinate
            
    return np.asarray(X), np.asarray(C)


def gaussian_mix(N, num_clusters=3, sigma_min=.1, sigma_max=.3):
    '''Generate (3) Gaussian Clusters'''
    X = [[],[]] # init data list [[x],[y]] 
    C = [] # init color list for plotting

    for cluster in range(int(num_clusters)):
        cov = np.random.uniform(sigma_min, sigma_max)*np.diag(np.ones(2))
        mu = [[0.0, 0.5],[1.0, 1.0],[1.0, 0.0]]
       
        for _ in range(int(N/num_clusters)):
            sx, sy = np.random.multivariate_normal(mu[cluster], cov)
            X[0].append(sx); X[1].append(sy) # x; y
            C.append(cluster)
        
    return np.asarray(X), np.asarray(C)


def hyperuniform_circle(N):
    '''Generate Hyperuniformly-Sampled 2-D Circle'''
    X = [[],[]] # init data list [[x],[y]] 
    C = np.linspace(0, 1, N) # init color list for plotting
    
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    for t in theta:
        X[0].append(np.cos(t)) # x
        X[1].append(np.sin(t)) # y
        
    return np.asarray(X), np.asarray(C)


def hyperuniform_ellipse(N, a=1, b=2):
    '''Generate Hyperuniformly-Sampled 2-D Ellipse'''
    assert(a < b) # a must be length of minor semi-axis; b major semi-axis
    
    X = [[],[]] # init data list [[x],[y]] 
    C = np.linspace(0, 1, N) # init color list for plotting
    
    angles = 2*np.pi*np.arange(N)/N
    
    if a != b:
        '''Given N points, combine scipy elliptic integral + optimize to find 
        N equidistant points along ellilpse manifold, then convert to angles'''
        e = np.sqrt(1.0 - a**2 / b**2)
        tot_size = special.ellipeinc(2.0*np.pi, e)
        arc_size = tot_size/N
        arcs = np.arange(N)*arc_size
        res = optimize.root(
                lambda x: (special.ellipeinc(x, e) - arcs), angles)
        angles = res.x
        
        arcs = special.ellipeinc(angles, e)

    for t in angles:
        X[0].append(a*np.cos(t)) # x
        X[1].append(b*np.sin(t)) # y
        
    return np.asarray(X), np.asarray(C)

def uniform_circle(N):
    '''Generate Hyperuniformly-Sampled 2-D Circle'''
    X = [[],[]] # init data list [[x],[y]] 
    C = np.linspace(0, 1, N) # init color list for plotting
    
    theta = np.random.uniform(0, 2*np.pi, N)
    theta.sort()
    
    for t in theta:
        X[0].append(np.cos(t)) # x
        X[1].append(np.sin(t)) # y
        
    return np.asarray(X), np.asarray(C)

def twospirals(n_points, noise=.2):
    """Generate Two Nested Spirals"""
    n = np.sqrt(np.random.rand(n_points,1)) * 220 * (2*np.pi)/360
    dx = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    dy = np.sin(n)*n + np.random.rand(n_points,1) * noise
    
    X = np.vstack((np.hstack((dx,dy)),np.hstack((-dx,-dy)))).T # data 
    C = np.hstack((np.zeros(n_points),np.ones(n_points))) # colors for plot
    
    return X, C 

if case == 0:
    X, C = barbell(N, 3)

elif case == 1:
    X, C = tree(N, 1, 3)

elif case == 2:
    X, C = gaussian_mix(N, num_clusters=3, sigma_min=0.03, sigma_max=0.035)
    
elif case == 3:
    X, C = hyperuniform_circle(N)
    
elif case == 4:
    X, C = hyperuniform_ellipse(N, a=0.5, b=1)
    
elif case == 5:
    X, C = uniform_circle(N)

elif case == 6:
    X, C = tree(N, 1, 6)
    perturb_data = True

elif case == 7:
    X, C = twospirals(int(N/2), noise=0.2)

X = X.T

if perturb_data == True:
    gn = np.random.normal(0, std_dev, X.shape)
    X += gn

### Inhomogeneous Diffusion Clustering
## Kernel/Low-Pass Filter and Algorithm
# Initialize Storage
epsilon_list = [[],[]] # [[epsilon_adjust],[epsilon_small]]
epsilon_list[1].append(epsilon_small) # store small epsilons (required)
eps_adjust = 0 # init number of times epsilon doubled
epsilon_list[0].append(eps_adjust) # store DM timestep adjustments
X_list = []; X_list.append(X.copy()) # store each X_i
S = [] # store densities for point sizes in viz
U = []; U.append(np.zeros(X.shape)) # arrows for gradient viz

# Initialize Density Arrays
D_i_s = np.zeros((X.shape[0],1))
q_i_s = np.zeros_like(D_i_s)
q_i1_s = np.zeros_like(q_i_s)
density_diff = math.inf

# Initialize Counter
i = 0
i_previous = -10

while i - i_previous > 1:    

    # Update i_previous
    i_previous = i
    print("i_previous: ", i_previous)
    
    # While densities are moving, use the same epsilon (if doubling epsilon)
    while density_diff > 1*10**(-4):
        
        # Update counter
        i += 1
        
        # Markov Chain
        K = spatial.distance.pdist(X, metric='euclidean')
        K = spatial.distance.squareform(K)
        K_s = np.exp(-(K**2) / epsilon_small)
        q_i_s = 1./np.sum(K_s, axis = 0) # D_inverse (i.e. 1/q(x_i))
        K1_s = np.diag(q_i_s**alpha) @ K_s @ np.diag(q_i_s**alpha)
        D_i_s = 1./np.sum(K1_s, axis = 0) # D_tilde_inverse
        P_s = np.diag(D_i_s) @ K1_s # _s means small epsilon
        
        # Condensation of X
        X_1 = P_s @ X
        
        ## Compute & Store Gradient
        U.append(X_1.copy() - X.copy()) 
        
        if save_spectra == True:
            ## Compute & Save Signular (Eigen) Values & R/L Vecs for each P_i
            Ui, Ei, Vi = np.linalg.svd(P_s)
            np.save('p_i/'+cdir+'/Vi_'+str(i)+'_'+fname+vname, Vi.T[:,1:spec_keep])
            np.save('p_i/'+cdir+'/Ei_'+str(i)+'_'+fname+vname, Ei[1:spec_keep])
            np.save('p_i/'+cdir+'/Ui_'+str(i)+'_'+fname+vname, Ui[1:spec_keep])
            
            # SVD for each P^t = P_t P_t-1 ... P_1
            if i == 1: # first iteration -> last P^t = Identity
                Pt_last = np.diag(np.ones(P_s.shape[0]))
            
            Pt = P_s @ Pt_last 
            
            Ut, Et, Vt = np.linalg.svd(Pt)
            np.save('p_t/'+cdir+'/Vt_'+str(i)+'_'+fname+vname, Vt.T[:,1:spec_keep])
            np.save('p_t/'+cdir+'/Et_'+str(i)+'_'+fname+vname, Et[1:spec_keep])
            np.save('p_t/'+cdir+'/Ut_'+str(i)+'_'+fname+vname, Ut[:,1:spec_keep])
        
            Pt_last = Pt.copy()
        
        # Check densities
        if i > 1:
            density_diff = np.max(np.abs(1./q_i_s - 1./q_i1_s)) # original dist fcn
        else:
            S.append(1./q_i_s)
            
            if save_spectra == True:
                ## Save Diffusion Map (DM) Spectra, Right Eigenvectors, Colors
                np.save('dm/'+cdir+'/V_'+fname+vname, Vi.T[:,1:spec_keep])
                np.save('dm/'+cdir+'/E_'+fname+vname, Ei[1:spec_keep])
                np.save('dm/'+cdir+'/C_'+fname+vname, C)
            
        ## Update arrays
        q_i1_s = q_i_s.copy()
        X = X_1.copy()
        X_list.append(X)
        S.append(1./q_i_s)
        epsilon_list[0].append(eps_adjust)
        epsilon_list[1].append(epsilon_small)
        
    
    # Update epsilon
    if double_eps == True:
        epsilon_small = double_eps_by * epsilon_small
        eps_adjust += 1
    
    # Reset density_diff
    density_diff = math.inf


# Number of Condensations
num_condensations = i

# One last time to get last D
K = spatial.distance.pdist(X, metric='euclidean')
K = spatial.distance.squareform(K)
K_s = np.exp(-(K**2) / epsilon_small)
q_i_s = 1./np.sum(K_s, axis = 0) # D_inverse (i.e. 1/q(x_i))
K1_s = np.diag(q_i_s**alpha) @ K_s @ np.diag(q_i_s**alpha)
D_i_s = 1./np.sum(K1_s, axis = 0) # D_tilde_inverse
P_s = np.diag(D_i_s) @ K1_s
X = P_s @ X

# Final Storage & Format Conversion 
epsilon_list[1].append(epsilon_small)
epsilon_list[0].append(eps_adjust)
X_list.append(X); X_list = np.asarray(X_list)
S.append(1./q_i_s) 
S = (2*X.shape[0])*np.asarray(S)/np.amax(np.asarray(S)) # normalize densities 
U.append(np.zeros(X.shape))
U = np.asarray(U)/np.amax(np.asarray(U)) # normalize vector field
epsilon_list = np.asarray(epsilon_list)

if save_spectra == True:
    ## Compute & Save Spectra
    # Signular Vals & Singular Vecs for each P_i
    i += 1
    _, Ei, Vi = np.linalg.svd(P_s)
    np.save('p_i/'+cdir+'/Vi_'+str(i)+'_'+fname+vname, Vi.T[:,1:spec_keep])
    np.save('p_i/'+cdir+'/Ei_'+str(i)+'_'+fname+vname, Ei[1:spec_keep])
    np.save('p_i/'+cdir+'/iterations_'+fname+vname, i)
        
    Pt = P_s @ Pt_last 
    
    Ut, Et, Vt = np.linalg.svd(Pt)
    np.save('p_t/'+cdir+'/Vt_'+str(i)+'_'+fname+vname, Vt.T[:,1:spec_keep])
    np.save('p_t/'+cdir+'/Et_'+str(i)+'_'+fname+vname, Et[1:spec_keep])
    np.save('p_t/'+cdir+'/Ut_'+str(i)+'_'+fname+vname, Ut[:,1:spec_keep])
    np.save('p_t/'+cdir+'/iterations_'+fname+vname, i)
    np.save('dm/'+cdir+'/iterations_'+fname+vname, i)
    np.save('dm/'+cdir+'/epsilon_list_'+fname+vname, epsilon_list)
    

## Generate Animations & Still Frames
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                        labelbottom=False, right=False, left=False, 
                        labelleft=False)
# Init Scatter Plot
scat = ax.scatter(X_list[0,:,0].T, X_list[0,:,1].T, c = C, s = S[0,:].T, 
                  cmap = colormap, vmin=np.amin(C), vmax=np.amax(C))

# Init Gradient Plot
qax = ax.quiver(X_list[0,:,0].T, X_list[0,:,1].T, 
                U[0,:,0].T, U[0,:,1].T, C, pivot='tail', 
                scale=3)

# 
if save_stills == True:
    frame_list.append(int(X_list.shape[0]*lratio)) # last still to save
    def animate(i):
        '''Animate Animate Density-Dependent Condensation
        + Gradient and Save Stills'''
        if i in frame_list:
            fsn = fdir+fname+'_'+str(N)+'pts_'+str(i)+'iter_'+str(vname)+fig_type    
            fig.savefig(fsn, bbox_inches='tight', transparent=True, dpi=300)

        scat.set_offsets(np.c_[X_list[i,:,0].T, X_list[i,:,1].T])
        scat.set_sizes(S[i,:].T)
        qax.set_UVC(U[i,:,0].T, U[i,:,1].T)
        qax.set_offsets(np.c_[X_list[i,:,0].T, X_list[i,:,1].T])

else:    
    def animate(i):
        '''Animate Density-Dependent Condensation + Gradient Only'''
        em = str(int(eps_num*2**epsilon_list[0, i])) # epsilon & multiplier
        eps_tex = '\\frac{'+str(em)+'\pi}{'+str(eps_denom)+'N}}'
        ax.set_title(r'$\bf{'+ptitle+'}$\nCondensation: '+str(i)+
                     ', $\epsilon='+eps_tex+'$') 
        scat.set_offsets(np.c_[X_list[i,:,0].T, X_list[i,:,1].T])
        scat.set_sizes(S[i,:].T)
        qax.set_UVC(U[i,:,0].T, U[i,:,1].T)
        qax.set_offsets(np.c_[X_list[i,:,0].T, X_list[i,:,1].T])

anim = FuncAnimation(fig, animate, interval=50, frames= X_list.shape[0])
 
# Set up Formatting for Movie Files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Nathan Brugnone'), bitrate=1800)

if save_movie == True:
    anim.save(pdir+fname+'_'+str(N)+'pts_'+str(vname)+'.mp4', writer=writer)

plt.draw()
plt.show()