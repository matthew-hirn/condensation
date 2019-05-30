# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:01:30 2019

@author: Nathan G. Brugnone
"""
#%%

#from numba import jit
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.colors as clrs
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib 
import numpy as np 
import math
import scipy as sp
from scipy import special
from scipy import optimize
from scipy import spatial

## Customize matplotlib
#matplotlib.rcParams.update(
#    {
#        'text.usetex': False,
#        'font.family': 'stixgeneral',
#        'mathtext.fontset': 'stix',
#    }
#)

# Fixing random state for reproducibility
np.random.seed(2387)

# Initialize Parameters
N = 2**8 # number of samples
fignum = ''#'\,\,(Figure\,\, 5)' # Corresponding figure in paper
eps_num = 1; eps_denom = 50
case = 7 # 0: Barbell; 
# 1: Tree; 
# 2: Gaussian Mix; 
# 3/4: Hyperuniform Circle/Ellipse; 
# 5: Uniform Cicle
# 6: Perturbed Tree !! Set p_cov !!
perturb_data = False # If True, set p_cov 
p_cov = 0.05
epsilon_small = eps_num*np.pi/(eps_denom*N) # Set epsilon (1.415, hyperuni core collapse)
epsilon_large = 3000*epsilon_small # Set epsilon
vname = 'supplement_0-5' # spectial tag for video file
pdir = 'movie/nips/' # vid save parent directory; default 'movie/'
colormap = cm.viridis # cmap for viz
double_eps_by = 2
double_eps = True # True: Viz; False: Saving Spectra
save_spectra = False # True: SVD of P**t, P_i, P**(t) = P_t P_t-1 ... P_1
save_movie = True 
save_stills = False
fig_type = '.pdf'
frame_list = [0,2,3,5,6,10,15,20,41,43,45,46,49,50,51,52,53,54,77,79,81,100,101,102] # all but last to keep
lratio = 1 # percent of way through vid of last frame to keep
spec_keep = 21 # number of spectra to keep

if case == 0:
    ptitle = 'Uniformly\,\,Sampled\,\, Barbell'+fignum
    fname = 'barbell'
    pdir += 'barbell/'
elif case == 1:
    ptitle = 'Uniformly\,\, Sampled\,\, Tree'+fignum
    fname = 'tree'
    pdir += 'tree/'
elif case == 2:
    ptitle = 'Gaussian\,\, Blobs'+fignum
    fname = 'gauss'
    pdir += 'gauss/'
elif case == 3:
    ptitle ='Hyperuniformly\,\, Sampled\,\, Circle'+fignum
    fname = 'hyperuni_circle'
    pdir += 'h_circle/'
elif case == 4:
    ptitle = 'Hyperuniformly\,\, Sampled\,\, Ellpse'+fignum
    fname = 'hyperuni_ellipse'
    pdir += 'ellipse/'
elif case == 5:
    ptitle = 'Uniformly\,\,Sampled\,\, Circle'+fignum
    fname = 'uni_circle'
    pdir += 'uni_circle/'
elif case == 6:
    ptitle = 'Noisey\,\, Tree'+fignum
    fname = 'perturb_tree'
    pdir += 'ptree/'
elif case == 7:
    ptitle = 'Two\,\, Spirals'+fignum
    fname = 'spirals'
    pdir += 'spirals/'

if save_stills == True:
    fdir = pdir+'fig/'

# Synthetic Datasets
def barbell(N, beta=1):
    '''Generate uniformly sampled 2D Barbell'''
    X = [[],[]]
    C = []
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
    '''Generate uniformly sampled 2D Tree with 2**levels branches'''
    X = [[],[]]
    C = []
    s = 0; root = [s, s] # root node position
    omega = np.pi/4 # half of anlge between branches
    xtop = np.cos(omega); ytop = np.sin(omega)
    xbot = np.cos(-omega); ybot = np.sin(-omega)
    
    for l in range(levels): # nuber of fork nodes
        for n in range(2**l): # quantify branch doubling 
            for i in range(int(N/(levels*2*(2**l)))): # uniform sample
                ## Top branch of current node
                top = np.random.uniform() # top branch sample FIX FIX!!
                X[0].append(root[0] + radius*top*xtop) # x
                X[1].append(root[1] + radius*top*ytop) # y
                
                ## Bottom branch of current node
                bottom = np.random.uniform() # bottom branch sample FIX!!
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


def gaussian_mix(N, num_clusters=3, xbound=[0.0,1.0], ybound=[0.0,1.0],
                 sigma_min=.1, sigma_max=.3):
    '''Generate 3 Gaussian clusters'''
    X = [[],[]]
    C = []
#    mu_min = np.amax([xbound[0],ybound[0]]) + sigma_max
#    mu_max = np.amin([xbound[1],ybound[1]]) - sigma_max
    for cluster in range(int(num_clusters)):
        cov = np.random.uniform(sigma_min, sigma_max)*np.diag(np.ones(2))
#        mu = np.random.uniform(mu_min, mu_max, 2)
        mu = [[0.0, 0.5],[1.0, 1.0],[1.0, 0.0]]
        for _ in range(int(N/num_clusters)):
            sx, sy = np.random.multivariate_normal(mu[cluster], cov)
            X[0].append(sx); X[1].append(sy)
            C.append(cluster)
        
    return np.asarray(X), np.asarray(C)


def hyperuniform_circle(N):
    '''Generate Hyperuniformly-Sampled 2-D Circle'''
    X = [[],[]] # Data [[x],[y]] 
    C = np.linspace(0, 1, N) # Color
    
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    for t in theta:
        X[0].append(np.cos(t)) # x
        X[1].append(np.sin(t)) # y
        
    return np.asarray(X), np.asarray(C)


def hyperuniform_ellipse(N, a=1, b=2):
    '''Generate Hyperuniformly-Sampled 2-D Ellipse'''
    assert(a < b) # a must be lentgth mino-semiaxis
    
    X = [[],[]] # Data [[x],[y]] 
    C = np.linspace(0, 1, N) # Color
    
    angles = 2*np.pi*np.arange(N)/N
    
    if a != b:
        e = np.sqrt(1.0 - a**2 / b**2)
        tot_size = sp.special.ellipeinc(2.0*np.pi, e)
        arc_size = tot_size/N
        arcs = np.arange(N)*arc_size
        res = sp.optimize.root(
                lambda x: (sp.special.ellipeinc(x, e) - arcs), angles)
        angles = res.x
        
        arcs = sp.special.ellipeinc(angles, e)

    for t in angles:
        X[0].append(a*np.cos(t)) # x
        X[1].append(b*np.sin(t)) # y
        
    return np.asarray(X), np.asarray(C)

def uniform_circle(N):
    '''Generate Hyperuniformly-Sampled 2-D Circle'''
    X = [[],[]] # Data [[x],[y]] 
    C = np.linspace(0, 1, N) # Color
    
    theta = np.random.uniform(0, 2*np.pi, N)
    theta.sort()
    
    for t in theta:
        X[0].append(np.cos(t)) # x
        X[1].append(np.sin(t)) # y
        
    return np.asarray(X), np.asarray(C)

def twospirals(n_points, noise=.2):
    """Two spirals Dataset"""
#    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
#    n = np.sqrt(np.random.rand(n_points,1)) * 300 * (2*np.pi)/360
#    n = np.sqrt(np.random.rand(n_points,1)) * 150 * (2*np.pi)/360
    n = np.sqrt(np.random.rand(n_points,1)) * 220 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    
    return np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))).T, np.hstack((np.zeros(n_points),np.ones(n_points)))

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


print(X.shape)
#%%
#X = X + np.array([[0],[np.abs(np.min(X[1]))]])
X = X.T
#Y = X.copy()

if perturb_data == True:
#    p_mu = np.zeros(X.shape[0])
#    p_cov = p_cov*np.diag(np.ones(X.shape[0]))
    gn = np.random.normal(0, p_cov, X.shape)
    X += gn

#%%

#eps_tmp = spatial.distance.pdist(X, metric='euclidean'
#eps_tmp = np.abs(X - np.mean(X, axis = 0).T)
#epsilon = np.max(eps_tmp)/2**7.5

#epsilon = np.pi/(10*N)
#epsilon = 1
#epsilon = 0.13

#epsilon = np.pi/(N) # Kernel width: np.pi/ N or N*10 N*100
alpha = 1 # Effect of density (0: KEEP; 1: CANCEL)

## Debug
#print(X[:10]) 
#print("X: ",X.shape) 
##np.save('dm/C_'+fname, C)
#
### Testing 1st Eigenvalue
#K = spatial.distance.pdist(X, metric='euclidean')
#K = spatial.distance.squareform(K)
#K = np.exp(-(K**2))
#q_i = 1./np.sum(K, axis = 0)
#K1 = np.diag(q_i**alpha) @ K @ np.diag(q_i**alpha)
#D_i = 1./np.sum(K1, axis = 0) 
#P = np.diag(D_i) @ K1
#
#v = 1./D_i
#v1 = np.asarray(v/np.sum(v))
#print(v1.shape)
#
#np.allclose(v1.T, v1.T@P)
##print(v1.T - v1.T@P)

#%%
# Plot
##plt.scatter(X1[0], X1[1], c = C1, cmap = cm.summer, vmin=np.amin(C1), vmax=np.amax(C1))
#plt.scatter(X.T[0], X.T[1], c = C, cmap = cm.viridis, vmin=np.amin(C), vmax=np.amax(C), marker='.')
##plt.xlim(np.min(X[0]) - 0.1*np.abs(np.min(X[0])), np.max(X[0])*1.1)
##plt.ylim(np.min(X[1]) - 0.1*np.abs(np.min(X[1])), np.max(X[1])*1.1)
#plt.show()
#%%


### Iterative contraction clustering (double while loop that updates epsilon)
## Kernel and Algorithm
# Storage
epsilon_list = [[],[],[]]; epsilon_list[1].append(epsilon_small) # store epsilons
epsilon_list[2].append(epsilon_large)
eps_adjust = 0; epsilon_list[0].append(eps_adjust) # store DM timestep adjustments
X_list = []; X_list.append(X.copy()) # store each X_i
S = [] # store densities for point sizes in viz
U = []; U.append(np.zeros(X.shape))

# Initialize Density Arrays
D_i_s = np.zeros((X.shape[0],1))
q_i_s = np.zeros_like(D_i_s)
q_i1_s = np.zeros_like(q_i_s)
density_diff = math.inf

# Debug
print("q_i: ", q_i_s.shape)
print("S: ", np.asarray(S).shape)

# Initialize Counter
i = 0
i_previous = -10

while i - i_previous > 1:    

    # Update i_previous
    i_previous = i
    print("i_previous: ", i_previous)
    
    # While densities are shifting, use the same epsilon
    while density_diff > 1*10**(-4):
        
        # Update i
        i += 1
        
        # Markov Chain
        K = spatial.distance.pdist(X, metric='euclidean')
        K = spatial.distance.squareform(K)
        K_s = np.exp(-(K**2) / epsilon_small)
        q_i_s = 1./np.sum(K_s, axis = 0) # D_inverse (i.e. 1/q(x_i))
        K1_s = np.diag(q_i_s**alpha) @ K_s @ np.diag(q_i_s**alpha)
        D_i_s = 1./np.sum(K1_s, axis = 0) # D_tilde_inverse
        P_s = np.diag(D_i_s) @ K1_s
        
#        K_l = np.exp(-(K**2) / epsilon_large)
#        q_i_l = 1./np.sum(K_l, axis = 0) # D_inverse (i.e. 1/q(x_i))
#        K1_l = np.diag(q_i_l**alpha) @ K_l @ np.diag(q_i_l**alpha)
#        D_i_l = 1./np.sum(K1_l, axis = 0) # D_tilde_inverse
#        P_l = np.diag(D_i_l) @ K1_l
        
        # Condensation of X
        X_1 = P_s @ X
        
        ## Compute & Store Gradient
        U.append(X_1.copy() - X.copy()) 
        
        if save_spectra == True:
            ## Compute & Save Spectra
            # Signular Vals & Singular Vecs for each P_i
            _, Ei, Vi = np.linalg.svd(P_s)
            np.save('p_i/Vi_'+str(i)+'_'+fname+vname, Vi.T[:,1:spec_keep])
            np.save('p_i/Ei_'+str(i)+'_'+fname+vname, Ei[1:spec_keep])
            
            # SVD for each P-t = P_t P_t-1 ... P_1
            if i == 1:
                Pt_last = np.diag(np.ones(P_s.shape[0]))
            
            Pt = P_s @ Pt_last 
            
            Ut, Et, Vt = np.linalg.svd(Pt)
            np.save('p_t/Vt_'+str(i)+'_'+fname+vname, Vt.T[:,1:spec_keep])
            np.save('p_t/Et_'+str(i)+'_'+fname+vname, Et[1:spec_keep])
            np.save('p_t/Ut_'+str(i)+'_'+fname+vname, Ut[:,1:spec_keep])
        
            Pt_last = Pt.copy()
        
        # Check densities
        if i > 1:
            density_diff = np.max(np.abs(1./q_i_s - 1./q_i1_s)) # original dist fcn
#            density_diff = np.max(np.abs(q_i - q_i1)) # modified dist fcn
        else:
            S.append(1./q_i_s)
            
            if save_spectra == True:
                ## Compute & Save Diffusion Map Spectra
                np.save('dm/V_'+fname+vname, Vi.T[:,1:spec_keep])
                np.save('dm/E_'+fname+vname, Ei[1:spec_keep])
                np.save('dm/C_'+fname+vname, C)
#                np.save('dm/eps_'+fname, np.asarray(epsilon_list))
#                np.save('p_i/eps_'+fname, np.asarray(epsilon_list))
#                np.save('p_t/eps_'+fname, np.asarray(epsilon_list))
            
        ## Save Arrays
        q_i1_s = q_i_s.copy()
        X = X_1.copy()
        X_list.append(X)
        S.append(1./q_i_s)
        epsilon_list[0].append(eps_adjust)
        epsilon_list[1].append(epsilon_small)
        epsilon_list[2].append(epsilon_large)
        
    
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

#K_l = np.exp(-(K**2) / epsilon_large)
#q_i_l = 1./np.sum(K_l, axis = 0) # D_inverse (i.e. 1/q(x_i))
#K1_l = np.diag(q_i_l**alpha) @ K_l @ np.diag(q_i_l**alpha)
#D_i_l = 1./np.sum(K1_l, axis = 0) # D_tilde_inverse
#P_l = np.diag(D_i_l) @ K1_l

# Final Storage & Format Conversion 
epsilon_list[2].append(epsilon_large)
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
    np.save('p_i/Vi_'+str(i)+'_'+fname+vname, Vi.T[:,1:spec_keep])
    np.save('p_i/Ei_'+str(i)+'_'+fname+vname, Ei[1:spec_keep])
    np.save('p_i/iterations_'+fname+vname, i)
        
    Pt = P_s @ Pt_last 
    
    Ut, Et, Vt = np.linalg.svd(Pt)
    np.save('p_t/Vt_'+str(i)+'_'+fname+vname, Vt.T[:,1:spec_keep])
    np.save('p_t/Et_'+str(i)+'_'+fname+vname, Et[1:spec_keep])
    np.save('p_t/Ut_'+str(i)+'_'+fname+vname, Ut[:,1:spec_keep])
    np.save('p_t/iterations_'+fname+vname, i)
    np.save('dm/iterations_'+fname+vname, i)
    np.save('dm/epsilon_list_'+fname+vname, epsilon_list)
    
    
    

# Debug
print("X_list: ", X_list.shape)
print("S: ", S.shape)
print("U: ", U.shape)

#plt.scatter(X_list[4,:,0], X_list[4,:,1], c = C, cmap = cm.summer, 
#            vmin=np.amin(C), vmax=np.amax(C))
#plt.show()
#%%

#import matplotlib.animation as animation
#from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
#figsize=(7, 7)


#minx = np.min(X_list[:,:,0]); maxx = np.max(X_list[:,:,0])
#miny = np.min(X_list[:,:,1]); maxy = np.max(X_list[:,:,1])
#mnx = minx - 0.05*np.abs(maxx-minx); mxx = maxx + 0.05*np.abs(maxx-minx) # X-axis 
#mny = miny - 0.05*np.abs(maxy-miny); mxy = maxy + 0.05*np.abs(maxy-miny) # Y-axis 
#mino = min(mnx, mny); maxo = max(mxx, mxy)
#ax.set(xlim=(mino, maxo), ylim=(mino, maxo))
ax.set_aspect('equal')
ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                        labelbottom=False, right=False, left=False, labelleft=False)

scat = ax.scatter(X_list[0,:,0].T, X_list[0,:,1].T, c = C, s = S[0,:].T, 
                  cmap = colormap, vmin=np.amin(C), vmax=np.amax(C))

qax = ax.quiver(X_list[0,:,0].T, X_list[0,:,1].T, 
                U[0,:,0].T, U[0,:,1].T, C, pivot='tail', 
                scale=3)


if save_stills == True:
    frame_list.append(int(X_list.shape[0]*lratio)) # last image ratio of total iterations
    ## Animate Density-Dependent Condensation + Gradient
    def animate(i):
        if i in frame_list:
            fsn = fdir+fname+'_'+str(N)+'pts_'+str(i)+'iter_'+str(vname)+fig_type    
            fig.savefig(fsn, bbox_inches='tight', transparent=True, dpi=300)
        
    #    ax.set_title('Condensation: '+str(i)+', Epsilon: '+str(epsilon_list[1, i]))
        scat.set_offsets(np.c_[X_list[i,:,0].T, X_list[i,:,1].T])
        scat.set_sizes(S[i,:].T)
        qax.set_UVC(U[i,:,0].T, U[i,:,1].T)
        qax.set_offsets(np.c_[X_list[i,:,0].T, X_list[i,:,1].T])

else:    
    ## Animate Density-Dependent Condensation + Gradient
    def animate(i):
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
#writer = Writer(fps=20, metadata=dict(artist='Nathan Brugnone'), bitrate=1800)
writer = Writer(fps=20, metadata=dict(artist='None'), bitrate=1800)

if save_movie == True:
    anim.save(pdir+fname+'_'+str(N)+'pts_'+str(vname)+'.mp4', writer=writer)

plt.draw()
plt.show()

#%%