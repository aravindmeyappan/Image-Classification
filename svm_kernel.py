import itertools
import warnings

import cvxopt
import numpy as np

class svm:
    """
    Class implementing a Support Vector Machine:
    First the Constrained optimization problem is converted to an unconstrained optimization using the lagrange multipliers. 
    Instead of minimising the primal function - L_P(w, b, lambda_mat) = 1/2 ||w||^2 - sum_i{lambda_i[(w * x + b) - 1]},
    we try to maximise the dual function - 
        L_D(lambda_mat) = sum_i{lambda_i} - 1/2 sum_i{sum_j{lambda_i lambda_j y_i y_j K(x_i, x_j)}}.
    """
    def __init__(
            self,
            kernel: str = "linear",
            gamma: float | None = None,
            deg: int = 3, # will be used by polynomial kernel, ignored by rest
            r: float = 0.,
            c: float | None = 1.
    ):
        # Lagrangian's multipliers, hyperparameters and support vectors are initially set to None
        self._lambdas = None # lagrange multiplers associated with each support vector
        self._sv_x = None # input features of the support vectors
        self._sv_y = None # labels of the support vectors
        self._w = None # weights of the support vectors
        self._b = None # biases of the support vectors

        # If gamma is None, it will be computed during fit process
        self._gamma = gamma

        # computing the kernel matrix that can be used to 
        self._kernel = kernel
        if kernel == "linear":
            self._kernel_fn = lambda x_i, x_j: np.dot(x_i, x_j)
        elif kernel == "rbf":
            self._kernel_fn = lambda x_i, x_j: np.exp(-self._gamma * np.dot(x_i - x_j, x_i - x_j))
        elif kernel == "poly":
            self._kernel_fn = lambda x_i, x_j: (self._gamma * np.dot(x_i, x_j) + r) ** deg
        elif kernel == "sigmoid":
            self._kernel_fn = lambda x_i, x_j: np.tanh(np.dot(x_i, x_j) + r)

        # Soft margin
        self._c = c

        self._is_fit = False

    def fit(self, x: np.ndarray, y: np.ndarray, verbosity: int = 1) -> None:
        """Fit the SVM on the given training set.

        Parameters
        ----------
        x : ndarray
            Training data with shape (n_samples, n_features).
        y : ndarray
            Ground-truth labels.
        verbosity : int, default=1
            Verbosity level in range [0, 3].
        """
        # If "verbosity" is outside range [0, 3], set it to default (1)
        if verbosity not in {0, 1, 2}:
            verbosity = 1

        n_samples, n_features = x.shape # (m,n)
        # If gamma was not specified in "__init__", it is set according to the "scale" approach
        if not self._gamma:
            self._gamma = 1 / (n_features * x.var())

        # computing the kernel matrix based on the chosen kernel
        k = np.zeros(shape=(n_samples, n_samples))
        for i, j in itertools.product(range(n_samples), range(n_samples)):
            k[i, j] = self._kernel_fn(x[i], x[j])

        ## rewriting the optimization problem according to cvxopt's API
        # p will me a matrix of size num_samples*num_samples such that Hij = yi*yj*phi(xi)T*phi(xj)
        p = cvxopt.matrix(np.outer(y, y) * k)

        # q will be a vector of size num_samples*1 of -1s
        q = cvxopt.matrix(-np.ones(n_samples))
        
        # Compute G and h matrix according to the type of margin used
        # g is a matrix of size 2m*m such that a diagonal matrix of -1s of size m*m is 
        # concatenated vertically with another diagonal matrix of 1s of size m*m
        # h is a vector of size 2m*1 with first m cells being 0s and the last m cells being c
        if self._c: # soft margin case
            g = cvxopt.matrix(np.vstack((
                -np.eye(n_samples),
                np.eye(n_samples)
            )))
            h = cvxopt.matrix(np.hstack((
                np.zeros(n_samples),
                np.ones(n_samples) * self._c
            )))
        else: # hard margin case
            g = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))

        # a is label vector and b is a scalar 0
        a = cvxopt.matrix(y.to_numpy().reshape(1, -1).astype(np.double))
        b = cvxopt.matrix(np.zeros(1))

        # Set CVXOPT options
        cvxopt.solvers.options["show_progress"] = False
        cvxopt.solvers.options["maxiters"] = 200

        # Compute the solution using the quadratic solver
        try:
            sol = cvxopt.solvers.qp(p, q, g, h, a, b)
        except ValueError as e:
            print(f"Impossible to fit, try to change kernel parameters; CVXOPT raised Value Error: {e:s}")
            return
            
        # Extract Lagrange multipliers
        lambdas = np.ravel(sol["x"])
        
        # Find indices of the support vectors, which have non-zero Lagrange multipliers, and save the support vectors
        # as instance attributes. only the support vectors contribute to the decision boundary
        if self._c: # soft margin puts an upperbound on the values that a positive lagrangian multiplier can take
            is_sv = (lambdas >= 1e-5) & (lambdas <= self._c) # we get a list of booleans
        else: # hard margin just wants all positive lagrangian multiplier
            is_sv = lambdas >= 1e-5

        # extract the input features and the labels of the support vectors and their corresponding lagrangian multipliers
        self._sv_x = x[is_sv]
        self._sv_y = y[is_sv]
        self._lambdas = lambdas[is_sv]
        
        # Compute b as 1/N_s sum_i{y_i - sum_sv{lambdas_sv * y_sv * K(x_sv, x_i}}
        sv_index = np.arange(len(lambdas))[is_sv]
        self._b = 0
        for i in range(len(self._lambdas)):
            self._b += self._sv_y.iloc[i]
            self._b -= np.sum(self._lambdas * self._sv_y * k[sv_index[i], is_sv])
        self._b /= len(self._lambdas)
        
        # Compute w only if the kernel is linear 
        # in other kernel methods we directly compute it during prediction using the kernels instead of storing it in w
        if self._kernel == "linear":
            self._w = np.zeros(n_features)
            for i in range(len(self._lambdas)):
                self._w += self._lambdas[i] * self._sv_x[i] * self._sv_y.iloc[i]
        else:
            self._w = None
        self._is_fit = True

        # Print results according to verbosity
        if verbosity in {1, 2}:
            print(f"{len(self._lambdas):d} support vectors found out of {n_samples:d} data points")
            if verbosity == 2:
                for i in range(len(self._lambdas)):
                    print(f"{i + 1:d}) X: {self._sv_x[i]}\ty: {self._sv_y.iloc[i]}\tlambda: {self._lambdas[i]:.2f}")
            print(f"Bias of the hyper-plane: {self._b:.3f}")
            print("Weights of the hyper-plane:", self._w)

    def project(
            self,
            x: np.ndarray,
            i: int | None = None,
            j: int | None = None
    ) -> np.ndarray:
        """Project data on the hyperplane.
        It is an helper function for the prediction. computes the function that decides which class it will be

        Parameters
        ----------
        x : ndarray
            Data points with shape (n_samples, n_features).
        i : int or None, default=None
            First dimension to plot (in the case of non-linear kernels).
        j : int or None, default=None
            Second dimension to plot (in the case of non-linear kernels).

        Returns
        -------
        ndarray
            Projection of the points on the hyperplane.
        """
        # If the model is not fit, raise an exception
        if not self.is_fit:
            raise SVMNotFitError
        # If the kernel is linear and "w" is defined, the value of f(x) is determined by
        #   f(x) = X * w + b
        if self._w is not None:
            return np.dot(x, self._w) + self._b
        else:
            # Otherwise, it is determined by
            #   f(x) = sum_i{sum_sv{lambda_sv y_sv K(x_i, x_sv)}}
            y_predict = np.zeros(len(x))
            for k in range(len(x)):
                for lda, sv_x, sv_y in zip(self._lambdas, self._sv_x, self._sv_y):
                    # Extract the two dimensions from sv_x if "i" and "j" are specified
                    if i or j:
                        sv_x = np.array([sv_x[i], sv_x[j]])

                    y_predict[k] += lda * sv_y * self._kernel_fn(x[k], sv_x)
            return y_predict + self._b

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the class of the given data points.
        Basically just returns the sign of the project function output

        Parameters
        ----------
        x : ndarray
            Data points with shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Predicted labels.
        """
        # To predict the point label, only the sign of f(x) is considered
        return np.sign(self.project(x))

    @property
    def is_fit(self) -> bool:
        return self._is_fit

    @property
    def sv_x(self) -> np.ndarray:
        return self._sv_x

    @property
    def sv_y(self) -> np.ndarray:
        return self._sv_y


class SVMNotFitError(Exception):
    """Exception raised when the "project" or the "predict" method of an SVM object is called without fitting
    the model beforehand.
    """