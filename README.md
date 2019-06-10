# Condensation
The Condensation project provides the latest code associated with the paper, "_Coarse Grainging of Data via Inhomogeneous Diffusion Condensation_," which details a clustering algorithm that iteratively employs deep cascades of intrinsic low-pass filters in a given data set. The paper is available on the arXiv at [URL Coming Soon].  
>Jupyter Notebook implemention coming soon!    
## Getting Started
### Condensation code and videos
#### Code
One will find `condensation_<version>.py` and `condensation_load_spectra_<version>.py` in the `code` folder. `condensation_<version>.py generates data sets, videos, and video still frames presented in "_Coarse Grainging of Data via Inhomogeneous Diffusion Condensation_."  To use: 
1. Open `condensation_<version>.py` in your favorite IDE
2. Choose parameter levels via `strings` and `floats` just below `module` imports 
3. Run the script
Some of the output may be imported into `condensation_load_spectra_<version>.py`. Specifically, if `save_spectra = True`, then one may use `condensation_load_spectra_<version>.py` to plot the singular values and create 2-D/3-D spectral embeddings, the latter of which are provided to encourage experimentation. Beware, however, that one may need to find a suitable basis in which to rotate the data for intelligble embedding. Subsequent versions will include tools to assist with this process. 
#### Media
The `media` folder contains `.mp4` videos generated using `condensation_<version>.py`. We believe the presentation of these dynamic videos ecourages the development of an intuition about the algorithm and its effects when iteratively applied to a data set.  
## Authors
These scripts were created by Nathan Brugnone and Matthew J. Hirn.
