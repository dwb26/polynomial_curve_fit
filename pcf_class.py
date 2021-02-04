from scipy import stats, optimize
import numpy as np

class PCF(object):

	def __init__(self, std_dev=0.25):
		self.std_dev = std_dev

	def f(self, x):
		""" 
		This is the function used to generate the synthetic data.
		"""
		return np.sin(2 * np.pi * x)

	def generate_data(self, x):
		"""
		Generate the noisy data.
		"""
		y = self.f(x) + stats.norm.rvs(scale=self.std_dev, size=len(x))
		return y

	def model(self, w, x):
		"""
		This is the polynomial model we use to fit the data.
		"""
		y = np.zeros(len(x))
		for m in range(len(w)):
			y += w[m] * x ** m
		return y

	def solve(self, degree, x, t):
		"""
		Fit the model to the training data.
		"""
		def res(w, x, t):
			return self.model(w, x) - t
		return optimize.least_squares(res, x0=np.ones(degree + 1), args=(x, t))