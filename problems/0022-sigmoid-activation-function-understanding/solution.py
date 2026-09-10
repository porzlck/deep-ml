import math

def sigmoid(z: float) -> float:
	#Your code here
	result=1/(1+math.exp(-z))
	return result