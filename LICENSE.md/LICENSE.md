@@ -99,7 +99,7 @@ def compute_distances_one_loop(self, X):
      # points, and store the result in dists[i, :].                        #
      #######################################################################

      # L2 distance
      # L2 distance.
      dists[i,:] = np.sqrt(np.sum((X[i,:] - self.X_train)**2, axis = 1))

      #######################################################################
 @@ -135,7 +135,11 @@ def compute_distances_no_loops(self, X):
    Y_squared = np.sum(self.X_train**2,axis=1)
    XY = np.dot(X, self.X_train.T)

    # Expand L2 distance formula to get L2(X,Y) = sqrt((X-Y)^2) = sqrt(X^2 + Y^2 -2XY)
    dists = np.sqrt(X_squared[:,np.newaxis] + Y_squared -2*XY)

    # Also useful https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
 @@ -167,10 +171,14 @@ def predict_labels(self, dists, k=1):
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################

      # Select a test row.
      test_row = dists[i,:]

      # np.argsort returns indices of sorted input.
      sorted_row = np.argsort(test_row)

      # Get the k closest indices.
      closest_y = self.y_train[sorted_row[0:k]]

      #########################################################################
 @@ -181,6 +189,7 @@ def predict_labels(self, dists, k=1):
      # label.                                                                #
      #########################################################################

      # Find the most occuring index in our closest k.
      y_pred[i] = np.argmax(np.bincount(closest_y))

      #########################################################################
