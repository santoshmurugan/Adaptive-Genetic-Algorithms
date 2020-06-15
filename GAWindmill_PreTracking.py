from pyeasyga.pyeasyga import GeneticAlgorithm
import numpy as np
import argparse

def main():

	parser = argparse.ArgumentParser()
    parser.add_argument('--crossover_probability', default=0.8, type=float, help='enter a decimal between 0 and 1')
    parser.add_argument('--starting_mutation_probability', default=0.005, type=float, help='enter a decimal between 0 and 1')
    parser.add_argument('--adaptive_mutation_factor', default=1.3, type=float, help='enter a decimal value for how much to multiply mutation rate per generation')
    parser.add_argument('--clipping_threshold', default=0.9, type=float, help='enter a decimal between 0 and 1 at which clipping is performed')
    parser.add_argument('--turbines', default=2, type=int, help='enter an integer number of turbines to start with')
    parser.add_argument('--grid_size', default=36, type=int, help='enter an perfect square integer representing grid size (e.g. 6x6 is 36)')


    args = parser.parse_args()
    given_crossover_probability = args.crossover_probability
    given_mutation_probability = args.starting_mutation_probability
    given_adaptive_mutation_factor = args.adaptive_mutation_factor
    given_clipping_threshold = args.clipping_threshold
    given_turbines = args.turbines
    grid_size=  args.grid_size
    
	input_data = [0]*grid_size # length of this list is the # of grid cells in layout
	easyga = GeneticAlgorithm(input_data, crossover_probability =given_crossover_probability , mutation_probability=given_mutation_probability, adaptive_mutation_factor = given_adaptive_mutation_factor, clipping_threshold = given_clipping_threshold, num_turbines = given_turbines)
	easyga.fitness_function = fitness
	easyga.run()
	print(easyga.best_individual())


def fitness (individual, data):

	freestream_velocity = 15
	individual_2D_list = [[individual[0], individual[1]],[individual[2], individual[3]]]

	individual_np = np.asarray(individual_2D_list)
	velocities = np.zeros(individual_np.shape)
	alpha = .9 # entrainment constant
	x = 150 # distance between perpendicular turbine centers
	r = 40 # radius of turbine

	for rowIndex in range(np.shape(individual_np)[0]):
		for colIndex in range(np.shape(individual_np)[1]):
			if individual_np[rowIndex][colIndex] == 1:
				if rowIndex == 0:
					velocities[rowIndex][colIndex] = freestream_velocity
				elif rowIndex == 1: # only look 1 up
					if individual_np[rowIndex-1][colIndex] == 1: 
						velocities[rowIndex][colIndex] = velocities[rowIndex-1][colIndex] * (1- (2/3)* (r/(alpha*x+r))**2)
					else:
						velocities[rowIndex][colIndex] = freestream_velocity
				elif rowIndex == 2: # look 1 up, then 2 up
					if individual_np[rowIndex-1][colIndex] == 1:
						velocities[rowIndex][colIndex] = velocities[rowIndex-1][colIndex] * (1- (2/3)* (r/(alpha*x+r))**2)
					elif individual_np[rowIndex-2][colIndex] == 1:
						velocities[rowIndex][colIndex] = velocities[rowIndex-1][colIndex] * (1- (2/3)* (r/(alpha*2*x+r))**2)
					else:
						velocities[rowIndex][colIndex] = freestream_velocity

	
	power = .3*np.sum(velocities**3)
	return power

main()