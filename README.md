# Efficient clustering 

## TODO

### AFKMC^2 

* Add the possibility to pass an array of random numbers: this is the bottleneck in term of speed. 

### Clustering

* Add a function to compute weights assigned to the centroids. 
* Add tests for the clustering algorithms. 

### Khorn

* Add the storage order as a template parameter, to allow for matrices stored in column-major. 

### Builds

Clean once and for all the external librairies. For now, they are installed locally and added to the C++ include path. Fix:
* GoogleTest: make it download in CMake
* Eigen, Pybind: 