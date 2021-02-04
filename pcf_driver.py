import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pcf_class import PCF
from sklearn.metrics import mean_squared_error as mse

# ------------------------------------------------------------------------------------------------------------------- #
# 
# Parameters
#
# ------------------------------------------------------------------------------------------------------------------- #
Ms = list(map(int, input("Enter the orders of the approximating polynomials: ").split()))
N_train = 10							# Size of the training set
N_val = 100								# Size of the validation set
fine_len = 1000							# Length of fine array for plotting
x_train = np.linspace(0, 1, N_train)	# We use N_train equally spaced points in the interval [0, 1] as input
x_val = np.linspace(0, 1, N_val)		# We use N_val equally spaced validation points in the interval [0, 1]
w_hats = []								# List to store the weight solutions
fine_x = np.linspace(0, 1, fine_len)	# x_train domain on a fine mesh for plotting
curves = np.empty((len(Ms), fine_len))	# Fitted curve solutions
train_errors = np.empty(len(Ms))		# Errors between the fitted curves and the training data
val_errors = np.empty(len(Ms))			# Errors between the fitted curves and the validation data
np.random.seed(13)						# Set the seed for reproducibility
for m in Ms:
	if m < 0:
		raise ValueError("The order of the polynomial should be a non-negative integer")


# ------------------------------------------------------------------------------------------------------------------- #
# 
# Parameter estimation
#
# ------------------------------------------------------------------------------------------------------------------- #
# Generate the data from the generative model
pcf = PCF()
t_train = pcf.generate_data(x_train)

# Generate the validation data
t_val = pcf.generate_data(x_val)

# Compute the fitted curve solutions for each input polynomial degree parameter
for m in range(len(Ms)):
	# Fit the model according to the training set
	result = pcf.solve(Ms[m], x_train, t_train)

	# Plug the minimising weight into the model to compute the curve
	curves[m] = pcf.model(result.x, fine_x)

	# Take the mean sqaured errors fpr the training and validation sets
	train_errors[m] = mse(pcf.model(result.x, x_train), t_train)
	val_errors[m] = mse(pcf.model(result.x, x_val), t_val)


# ------------------------------------------------------------------------------------------------------------------- #
# 
# Plotting
#
# ------------------------------------------------------------------------------------------------------------------- #
plot = True
# plot = False
if plot:
	fig = plt.figure(figsize=(10, 8))
	plt.subplots_adjust(hspace=0.5)
	ax1 = plt.subplot(311)
	ax2 = plt.subplot(312)
	ax3 = plt.subplot(313)

	ax1.plot(fine_x, pcf.f(fine_x), "red")
	ax1.scatter(x_train, t_train, color="black")

	for m in range(len(Ms)):
		ax2.plot(fine_x, curves[m], label="M={}".format(Ms[m]))
	ax2.scatter(x_train, t_train, color="black")

	ax3.plot(Ms, train_errors, marker="o", label="Training")
	ax3.plot(Ms, val_errors, marker="o", label="Val.")

	ax1.set_title("Generative sample")
	ax2.set_title("Fitted polynomial curves")
	ax3.set_title("Mean squared errors")
	ax1.set_xlabel("")
	ax1.set_xticks([])
	ax2.set_xlabel("x")
	ax3.set_xlabel("M")
	ax2.legend()
	ax3.legend()
	plt.savefig("M={}".format(Ms))
	plt.show()

