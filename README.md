# Condensation
The Condensation project provides the latest code associated with the paper, "_Coarse Grainging of Data via Inhomogeneous Diffusion Condensation_," which details a clustering algorithm that iteratively employs deep cascades of intrinsic low-pass filters in a given data set. The paper is available on the arXiv at [URL Coming Soon].  
>Jupyter Notebook implemention **now availeble**!    
## Getting Started
### Condensation code and media
#### Code
One will find `condensation_animations.ipynb`, `Condensation.py`, and `Spectral_condensation.py` in the `code` folder. `condensation_animations.ipynb` generates data sets, videos, and video still frames presented in "_Coarse Grainging of Data via Inhomogeneous Diffusion Condensation_."  To use: 
1. Open the Jupyter Notebook file `condensation_animations.ipynb`
2. Follow the enclosed instructions

More advanced users may wish to explore the code deeper via `Condensation.py` and `Spectral_condensation.py`. For instance, condesnation output files may be imported into `Spectral_condensation.py`. Then, one may plot the singular values and create 2-D and 3-D spectral embeddings, the latter of which are provided to encourage experimentation. 
#### Media
The `media` folder contains `.mp4` videos generated using `condensation_animations.ipynb`. We believe the presentation of these dynamic videos ecourages the development of an intuition about the algorithm and its effects when iteratively applied to a data set.  
## Authors
These scripts were created by Nathan Brugnone and Matthew J. Hirn.