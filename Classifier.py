from sklearn.neighbors import KNeighborsClassifier
import math

def findShortestDistance(distances, nNeighbors):
    # Sort distances and get the indices of the closest nNeighbors
    sorted_indices = sorted(range(len(distances)), key=lambda x: distances[x])
    return sorted_indices[:nNeighbors]

def KNNClassifier(features, labels, predictionSet, nNeighbors):
    # Initialize an empty list for distances
    distances = []

    # Find all distances between prediction set and training features
    for i in range(len(predictionSet)):
        distance_list = []
        for j in range(len(features)):
            dist = math.sqrt(
                (predictionSet[i][0] - features[j][0]) ** 2 +
                (predictionSet[i][1] - features[j][1]) ** 2 +
                (predictionSet[i][2] - features[j][2]) ** 2
            )
            distance_list.append(dist)
        distances.append(distance_list)

    # Prepare the result list
    result = []

    # Find the K nearest neighbors and classify each prediction
    for i in range(len(distances)):
        # Get the indices of the nearest neighbors
        nearest_neighbors = findShortestDistance(distances[i], nNeighbors)
        
        # Count labels for the nearest neighbors
        healthy = 0
        sickle = 0
        
        for index in nearest_neighbors:
            if labels[index] == 0:
                healthy += 1
            elif labels[index] == 1:
                sickle += 1
        
        # Classify based on majority vote
        if healthy > sickle:
            result.append(0)  # Class 0: healthy
        else:
            result.append(1)  # Class 1: sickle

    return result

