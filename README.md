# Adaptive-Genetic-Algorithms
----------------------------------------------------------------------------------------------------------
Usage: 

The following command-line modifiers/flags are available:
--crossover_probability: decimal between 0 and 1, dictates how chromosomes perform crossover. Default is 0.3.
--starting_mutation_probability: decimal between 0 and 1, dictates how individual chromosomes can mutate. Default is 0.5.
--adaptive_mutation_factor: decimal for how much to multiply mutation rate per generation. Default is 0.2.
--clipping_threshold: decimal between 0 and 1 at which clipping is performed. Default is 0.9
--turbines: integer number of turbines to start with. Default is 10.
--grid_size: perfect square integer representing grid size (e.g. 6x6 is 36). Default is 36.

For further explanation, please see the description below.

----------------------------------------------------------------------------------------------------------
Description: 

This repository contains code for an improved genetic algorithms framework, using biologically inspired modifications. (The original genetic algorithms framework, PyEasyGA, can be found at https://github.com/remiomosowon/pyeasyga. Any code originally from there belongs to the owner of that repository).

We improve the performance of the genetic algorithm in two ways. First, we use an adaptive mutation rate, which is inversely proportional to the number of generations already created by the genetic algorithm. This way, later generations will have a lower mutation rate, potentially accelerating model convergence.  

Second, we use an adaptive crossover rate which has two modifications from the normal, called parent-weighted inheritance and clipping. 

Parent-Weighted Inheritance: Contrary to uniform crossover, in which the child has a 50% chance of inheriting a particular bit from either parent, in parent-weighted inheritance, we instead say that the probability of inheriting a bit from parent A is equal to the value of the objective function of parent A, divided by the sum of the objective function values of both parents. This way, the child is more likely to inherit from the “better” parent. 

Clipping: Now, consider the case where one parent is significantly better than the other parent. In this case, the child would likely look extremely similar to the better parent, but unfortunately this ends up hindering genetic diversity. To keep parent-weighted inheritance from skewing too far to one parent, we additionally introduce clipping, which states that the probability of inheriting from one particular parent cannot exceed a threshold (e.g. 90%).

Taken together, the adaptive mutation and adaptive crossover rates (parent-weighted inheritance and clipping) improve the performance of our genetic algorithms.

----------------------------------------------------------------------------------------------------------

We provide a proof of concept, in which we apply our modified genetic algorithms to the task of Windmill Farm Layout Optimization. Over 100 runs, with a 6x6 windfarm (i.e. 36 potential spots for windmills), our improved genetic algorithms consistently outperform the original genetic algorithms framework by 750 fitness points (approximately 7%). (Calculations of fitness are derived from equations in Mosetti et al.'s original windfarm optimization paper- source provided upon request).

(Developed by Santosh Murugan (Stanford, Artificial Intelligence) and Andy Kim (Stanford, Aero/Astro) - any questions can be directed to smurugan AT stanford DOT edu).
