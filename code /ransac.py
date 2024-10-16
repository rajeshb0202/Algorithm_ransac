import numpy as np

# Function to fit a line to a set of inliers using least squares (linear regression)
def fit(inliers):
    """
    Args:
        inliers (np.array): A 2D array of inliers where each point is [x, y].
    
    Returns:
        tuple: The slope and intercept of the fitted line.
    """
    x_inliers = inliers[:, 0]  # Extract the x-values of the inliers
    y_inliers = inliers[:, 1]  # Extract the y-values of the inliers
    
    # Use np.polyfit to fit a line (1st degree polynomial) to the inliers. 
    slope, intercept = np.polyfit(x_inliers, y_inliers, 1)
    
    return slope, intercept



# Function to calculate the distance of a point from a line (model)
def distance(point, model):
    """
    Args:
        point (np.array): A point [x, y] to measure the distance from.
        model (tuple): The line model, represented by (slope, intercept).
    
    Returns:
        float: The perpendicular distance from the point to the line.
    """
    x, y = point  
    slope, intercept = model 
    
    # Calculate the perpendicular distance from the point to the line using the formula:
    distance = np.abs(slope * x - y + intercept) / np.sqrt(slope**2 + 1)
    
    return distance




# Function to calculate the total error between a set of inliers and the fitted model (line)
def error(model, inliers):
    """
    Args:
        model (tuple): The line model, represented by (slope, intercept).
        inliers (np.array): The array of inliers where each point is [x, y].
    
    Returns:
        float: The total error, measured as the sum of absolute errors.
    """
    slope, intercept = model  # Extract the slope and intercept of the model
    y_inliers = inliers[:, 1]  # Extract the y-values of the inliers
    
    # Predict the y-values using the fitted model (y = mx + b)
    estimated_y = slope * inliers[:, 0] + intercept
    
    # Calculate the total error as the sum of absolute differences between actual and predicted y-values
    error = np.sum(np.abs(y_inliers - estimated_y))
    
    return error




# RANSAC algorithm to find the best-fitting line model in the presence of outliers
def ransac(temp_points, iterations, input_data, thres, min_good_points):
    """
    Runs the RANSAC (Random Sample Consensus) algorithm to find the best-fitting line model
    by iteratively sampling points and fitting a model that is robust to outliers.
    
    Args:
        temp_points (int): Number of points to randomly select for fitting the model.
        iterations (int): The number of iterations to run the RANSAC algorithm.
        input_data (np.array): The input data points, where each point is [x, y].
        thres (float): The distance threshold to consider a point as an inlier.
        min_good_points (int): The minimum number of inliers required to accept a model.
    
    Returns:
        tuple: The best-fit model (slope, intercept) and the inliers used to fit that model.
    """
    
    bestFit = None  # To store the best model (slope, intercept) found during iterations
    bestError = np.inf  # Initialize best error to infinity for comparison

    # Loop through the RANSAC iterations
    for i in range(iterations):
        # Randomly select 'temp_points' points from the input data to fit the model
        idx_tempInliers = np.random.choice(np.arange(input_data.shape[0]), temp_points)
        
        # Initialize an empty list to store additional inliers
        tempInlierAdd = []
        
        # Fit a model (line) using the randomly selected points (tempInliers)
        tempModel = fit(input_data[idx_tempInliers])
        
        # Loop through the input data and find additional inliers that are close to the fitted model
        for j, point in enumerate(input_data):
            # Skip points that were already used for the initial model fitting
            if j not in idx_tempInliers:
                # If the distance of the point from the model is less than the threshold, consider it an inlier
                if distance(point, tempModel) < thres:
                    tempInlierAdd.append(point)
        
        # If the number of inliers is greater than the minimum required, refit the model with all inliers
        if len(tempInlierAdd) > min_good_points:
            tempInlierAdd = np.array(tempInlierAdd)  # Convert the list of inliers to a NumPy array
            
            # Fit a new model using all the inliers (both initial and additional)
            newModel = fit(tempInlierAdd)
            
            # Calculate the error of the new model
            newError = error(newModel, tempInlierAdd)

            # If the new model has a lower error, update the best model and error
            if newError < bestError:
                bestError = newError
                bestFit = newModel
    
    # Return the best-fit model and the inliers used to fit that model
    return bestFit, tempInlierAdd
