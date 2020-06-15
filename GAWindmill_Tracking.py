from pyeasyga.pyeasyga import ModifiedGeneticAlgorithm
from pyeasyga.pyeasygaorig import GeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--crossover_probability', default=0.3, type=float, help='enter a decimal between 0 and 1')
	parser.add_argument('--starting_mutation_probability', default=0.5, type=float, help='enter a decimal between 0 and 1')
	parser.add_argument('--adaptive_mutation_factor', default=0.2, type=float, help='enter a decimal value for how much to multiply mutation rate per generation')
	parser.add_argument('--clipping_threshold', default=0.9, type=float, help='enter a decimal between 0 and 1 at which clipping is performed')
	parser.add_argument('--turbines', default=10, type=int, help='enter an integer number of turbines to start with')
	parser.add_argument('--grid_size', default=36, type=int, help='enter an perfect square integer representing grid size (e.g. 6x6 is 36)')

	args = parser.parse_args()
	given_crossover_probability = args.crossover_probability
	given_mutation_probability = args.starting_mutation_probability
	given_adaptive_mutation_factor = args.adaptive_mutation_factor
	given_clipping_threshold = args.clipping_threshold
	given_turbines = args.turbines
	grid_size=  args.grid_size

	input_data = [0]*grid_size # length of this list is the # of grid cells in layout
	
	easyga = ModifiedGeneticAlgorithm(input_data, crossover_probability = given_crossover_probability, mutation_probability=given_mutation_probability, adaptive_mutation_factor = given_adaptive_mutation_factor, clipping_threshold = given_clipping_threshold, num_turbines = given_turbines)
	easyga.fitness_function = fitness
		
	m1_fitness_list = easyga.run()
	m1_generations = [i[0] for i in m1_fitness_list]
	m1_fitnesses = [i[1][0] for i in m1_fitness_list]

	# compared to original
	originalga = GeneticAlgorithm(input_data, crossover_probability = given_crossover_probability, mutation_probability=given_mutation_probability)
	originalga.fitness_function = fitness
		
	orig_fitness_list = easyga.run()
	orig_generations = [i[0] for i in orig_fitness_list]
	orig_fitnesses = [i[1][0] for i in orig_fitness_list]


	NUM_RUNS = 100
	for index in tqdm(range(NUM_RUNS)):
		easyga = ModifiedGeneticAlgorithm(input_data, crossover_probability = 0.4, mutation_probability=0.5, adaptive_mutation_factor = .65, clipping_threshold = 0.9, num_turbines = 4)
		easyga.fitness_function = fitness
		
		mnew_fitness_list = easyga.run()
		mnew_generations = [i[0] for i in mnew_fitness_list]
		mnew_fitnesses = [i[1][0] for i in mnew_fitness_list]

		for i in range(len(m1_fitnesses)):
			m1_fitnesses[i] += mnew_fitnesses[i]

		# compared to original
		originalga = GeneticAlgorithm(input_data, crossover_probability = 0.4, mutation_probability=0.5)
		originalga.fitness_function = fitness
		
		orignew_fitness_list = easyga.run()
		orignew_generations = [i[0] for i in orignew_fitness_list]
		orignew_fitnesses = [i[1][0] for i in orignew_fitness_list]

		for i in range(len(orig_fitnesses)):
			orig_fitnesses[i] += orignew_fitnesses[i]

	for i in range(len(orig_fitnesses)):
		m1_fitnesses[i] /= NUM_RUNS
		orig_fitnesses[i] /= NUM_RUNS


	print('Making Plot: ')

	plt.plot(m1_generations, m1_fitnesses, label = 'Modified')
	plt.xlabel("Generations")
	plt.ylabel("Fitness")
	plt.title("Convergence: Fitness of Best Individual Over Time")
	#plt.show()

	plt.plot(orig_generations, orig_fitnesses, label = 'Original')
	plt.legend()
	plt.show()





	#print(easyga.best_individual())


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