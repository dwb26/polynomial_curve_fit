# polynomial_curve_fit
This is a code created as an exercise based on *Pattern Recognition and Machine Learning* by Bishop (2006). The code is my own.

The driver code pcf_driver.py takes as input degrees of polynomials that are fit to synthetic data from a generative model. The fitting is done with respect to the coefficients of the polynomial. The optimal coefficients are found using scipy.optimize.least_squares. Note that the input can be 1 value or several.

The code outputs 3 subplots. The 1st subplot shows the generative function and the noisy data generated from this function using additive Gaussian noise. The choice of function is hard coded into pcf_class.py. The 2nd subplot shows the fitted curves for each input polynomial degree value. The 3rd is the mean square error on the training set and validation for each polynomial.

To run the driver code, do "ipython pcf_driver.py inputs" in the command line, where inputs is one or multiple polynomial degrees of interest.
