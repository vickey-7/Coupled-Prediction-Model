# Coupled-Prediction-Model
A hybrid CFD–machine learning framework that accelerates pore-scale reactive transport simulations.

Modifications to the solver are based on  GeoChemFoam (https://github.com/GeoChemFoam).
## Requirements
- Python 3.9
- Pytorch 2.5.1(CUDA 12.4)
- Openfoam v2406

Before compilation, ensure that Miniconda is installed and the makefile options for the Python environment paths are located in:
 `/Make/options`
You can modify the following variables to match your Conda environment’s include and library directories:

 `PYTHON_INCLUDE = -I$(HOME)/miniconda3/envs/cpm-env/include/python3.9`
 
 `PYTHON_LIB  = -L$(HOME)/miniconda3/envs/cpm-env/lib -lpython3.9`

For more information, contact at s24020059@s.upc.edu.cn
