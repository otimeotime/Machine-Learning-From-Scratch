import numpy as np

"""
Support vector machine using sequential minimal optimization algorithm, here the algorithm optimizes the dual function, which
is derived from the original marginal function. Here I also add more kernels for the model (rbf, linear, quadratic)

To control the kernel, you have to initialize it when create object SVM with parameter 'kernel_type' (linear, quadratic, rbf)

Reference: Platt, J. C. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines.
Microsoft Research Technical Report MSR-TR-98-14.
"""

class SVM:
    def __init__(self, max_ite=1000, kernel_type='linear', C=1.0, epsilon=0.001, tol=1e-3):
        self.kernels = {
            'linear': self.kernel_linear,
            'quadratic': self.kernel_quadratic,
            'rbf': self.kernel_rbf
        }
        self.kernel = self.kernels[kernel_type]
        self.max_ite = max_ite
        self.C = C
        self.epsilon = epsilon # Epsilon for alpha change check
        self.tol = tol # Tolerance for KKT violation check
        
        # Parameters learned during fitting
        self.alpha = None
        self.b = 0.0
        self.X_train = None
        self.Y_train = None
        self.n_feature = 0
        self.n_example = 0
        self.support_vector_indices = None # Store indices of support vectors

        # Optional: Error cache for speed
        self.error_cache = None


    def kernel_linear(self, x1, x2):
        # Ensure inputs are treated as column vectors if necessary
        return np.dot(x1.T, x2)[0, 0]

    def kernel_quadratic(self, x1, x2):
        return (np.dot(x1.T, x2)[0, 0])**2

    def kernel_rbf(self, x1, x2, gamma=0.5): # Added gamma parameter for RBF
        # Ensure inputs are treated as column vectors
        diff = x1 - x2
        return np.exp(-gamma * np.dot(diff.T, diff)[0, 0])
        # Note: Original code used hardcoded 0.2 -> gamma = 1 / (2 * 0.2^2) approx if sigma=0.2
        # Using a more standard gamma formulation here.

    def _compute_f(self, x_k_idx):
        # Computes decision function output for example k using dual form
        f_k = 0
        if self.support_vector_indices is None: # Before fitting or if no SVs yet
             indices_to_use = range(self.n_example)
        else: # Optimization: Use only support vectors later if needed
            indices_to_use = range(len(self.support_vector_indices)) # Using all for simplicity now

        for i in indices_to_use:
             # Use stored alpha, Y_train, X_train
             if self.alpha[i] > 0: # Only non-zero alphas contribute
                 f_k += self.alpha[i] * self.Y_train[i] * self.kernel(
                     self.X_train[:, i].reshape(-1, 1),
                     self.X_train[:, x_k_idx].reshape(-1, 1)
                 )
        f_k += self.b
        return f_k

    def _compute_error(self, k):
         # Compute error E_k = f(x_k) - y_k
        if self.error_cache is not None and self.error_cache[k] is not None:
             return self.error_cache[k]
        
        f_k = self._compute_f(k)
        E_k = f_k - self.Y_train[k]

        if self.error_cache is not None:
             self.error_cache[k] = E_k
        return E_k


    def predict(self, X_test):
        if self.alpha is None or self.X_train is None or self.Y_train is None:
            raise RuntimeError("SVM model has not been trained yet.")

        Y_pred = []
        n_test = X_test.shape[1]

        # Use support vector indices if available for efficiency
        indices_to_use = np.where(self.alpha > 1e-5)[0] # Use a small threshold
        if len(indices_to_use) == 0:
             indices_to_use = range(self.n_example) # Fallback if no SVs found yet


        for i in range(n_test):
            f_pred = 0
            x_t = X_test[:, i].reshape(-1, 1)
            for k in indices_to_use:
                 # Access stored alpha, Y_train, X_train using index k
                 f_pred += self.alpha[k] * self.Y_train[k] * self.kernel(
                     self.X_train[:, k].reshape(-1, 1), x_t
                 )
            f_pred += self.b
            Y_pred.append(np.sign(f_pred))

        return np.array(Y_pred).reshape(-1, 1)

    def fit(self, X: np.array, Y: np.array):
        self.n_feature, self.n_example = X.shape
        self.X_train = X
        self.Y_train = Y
        self.alpha = np.zeros((self.n_example, 1))
        self.b = 0.0
        # Initialize error cache
        self.error_cache = [None] * self.n_example


        num_changed = 0
        examine_all = True # Start by examining all alphas
        ite = 0

        while (num_changed > 0 or examine_all) and ite < self.max_ite:
            num_changed = 0
            if examine_all:
                # Loop over all examples
                indices_to_examine = range(self.n_example)
            else:
                # Loop over examples where alpha is not 0 and not C (potential KKT violators)
                indices_to_examine = np.where((self.alpha > 0) & (self.alpha < self.C))[0]

            for i in indices_to_examine:
                Ei = self._compute_error(i)
                
                # Check if example i violates KKT conditions (within tolerance tol)
                # Platt's second heuristic condition check:
                if (self.Y_train[i] * Ei < -self.tol and self.alpha[i] < self.C) or \
                   (self.Y_train[i] * Ei > self.tol and self.alpha[i] > 0):

                    # Select j != i randomly (simple heuristic)
                    # More complex heuristics exist (e.g., maximizing |Ei - Ej|)
                    j = i
                    while j == i:
                         j = np.random.randint(0, self.n_example)

                    Ej = self._compute_error(j)

                    # Save old alphas
                    ai_old, aj_old = self.alpha[i].copy(), self.alpha[j].copy()

                    # Compute bounds L and H (U and V in your original code)
                    if self.Y_train[i] != self.Y_train[j]:
                        L = max(0, aj_old - ai_old)
                        H = min(self.C, self.C + aj_old - ai_old)
                    else:
                        L = max(0, ai_old + aj_old - self.C)
                        H = min(self.C, ai_old + aj_old)

                    if L == H:
                        continue # Skip to next i

                    # Compute eta = Kii + Kjj - 2Kij
                    xi = self.X_train[:, i].reshape(-1, 1)
                    xj = self.X_train[:, j].reshape(-1, 1)
                    eta = self.kernel(xi, xi) + self.kernel(xj, xj) - 2 * self.kernel(xi, xj)

                    if eta <= 0:
                        # print(f"Warning: eta <= 0 ({eta}) for pair ({i},{j}). Skipping.")
                        continue # Skip if eta is non-positive

                    # Compute new aj
                    aj_new = aj_old + self.Y_train[j] * (Ei - Ej) / eta

                    # Clip aj
                    if aj_new > H:
                        aj_new = H
                    elif aj_new < L:
                        aj_new = L

                    # Check if change in aj is significant
                    if abs(aj_new - aj_old) < self.epsilon: # Use epsilon here
                        continue # Skip update if change is too small

                    # Compute new ai
                    ai_new = ai_old + self.Y_train[i] * self.Y_train[j] * (aj_old - aj_new)

                    # Update alpha vector
                    self.alpha[i] = ai_new
                    self.alpha[j] = aj_new
                    
                    # Update error cache for i and j if using it
                    self.error_cache[i] = Ei # Mark as invalid
                    self.error_cache[j] = Ej # Mark as invalid


                    # Compute new threshold b
                    b1 = self.b - Ei - self.Y_train[i] * (ai_new - ai_old) * self.kernel(xi, xi) - \
                                      self.Y_train[j] * (aj_new - aj_old) * self.kernel(xi, xj)
                    b2 = self.b - Ej - self.Y_train[i] * (ai_new - ai_old) * self.kernel(xi, xj) - \
                                      self.Y_train[j] * (aj_new - aj_old) * self.kernel(xj, xj)

                    if 0 < ai_new < self.C:
                        self.b = b1
                    elif 0 < aj_new < self.C:
                        self.b = b2
                    else:
                        # Both new alphas are at bounds 0 or C
                        self.b = (b1 + b2) / 2.0

                    num_changed += 1

            # --- End of loop through indices_to_examine ---

            ite += 1
            # print(f"Iteration {ite}, Alpha changes: {num_changed}")

            # Decide whether to examine all alphas next time
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True # If no changes in non-bound alphas, check all again
        
        # Store support vector indices after training
        self.support_vector_indices = np.where(self.alpha > 1e-5)[0]
        # print(f"Training finished after {ite} iterations.")
        # print(f"Number of support vectors: {len(self.support_vector_indices)}")


    def get_accuracy(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        correct = (Y_pred == Y_test)
        acc = sum(correct) / Y_test.shape[0] * 100
        return acc[0] # Return scalar value