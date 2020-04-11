import random
import numpy as np

# Author Antonio Leblanc
# v_0.2 - 18/09/19

class Population:
    def __init__(self, f, lower_bounds, upper_bounds, x_step, f_if, if_po, if_target, if_max_deviation, pop_size, n_parents, 
                 deterministic_selection_percentage=.7, mutation_rate = 0.05):
        
        self.f = f
        self.n_genes = len(lower_bounds)
        self.ub = upper_bounds
        self.lb = lower_bounds
        self.x_step = x_step
        
        self.f_if = f_if
        self.if_po = if_po
        self.if_target = if_target
        self.if_max_deviation = if_max_deviation
        
        self.individuals = [Individual(f, lower_bounds, upper_bounds, x_step, f_if, if_po, if_target, if_max_deviation) for i in range(pop_size)]
        self.parents = []
        self.kids = []
        
        self.pop_size = pop_size
        self.n_parents = n_parents
        self.n_kids = pop_size - n_parents
        self.n_parents_to_select_deterministically = round(deterministic_selection_percentage*n_parents)
        self.n_parents_to_select_randomly = self.n_parents - self.n_parents_to_select_deterministically
        self.mutation_rate = mutation_rate
    
    def sort_by_fitness(self):
        self.individuals.sort(key = lambda individual: individual.get_fitness())

    def get_best_individual(self):
        return min(self.individuals, key = lambda individual: individual.get_fitness())
    
    def get_oldest_individual(self):
        return max(self.individuals, key = lambda individual: individual.get_age())
    
    def select_parents(self):
        self.sort_by_fitness()
        self.parents = self.individuals[:self.n_parents_to_select_deterministically]
        remaining_individuals = self.individuals[self.n_parents_to_select_deterministically:]
        self.parents.extend(random.sample(remaining_individuals, k = self.n_parents_to_select_randomly))
            
    def reproduce(self):
        self.kids = []
        for i in range(self.n_kids):
            mom = random.choice(self.parents)
            dad = random.choice(self.parents)
            kid = Individual(self.f, self.lb, self.ub, self.x_step, self.f_if, self.if_po, self.if_target, self.if_max_deviation)
            new_genes = [np.mean([mom.genes[gene],dad.genes[gene]]) for gene in range(self.n_genes)]
            kid.set_genes(new_genes)
            self.kids.append(kid)
    
    def mutate(self):
        for individual in self.kids:
            if random.random() < self.mutation_rate:
                individual.mutate()
    
    def next_generation(self):
        self.individuals = []
        self.individuals = self.parents + self.kids
        for parent in self.parents:
            parent.increase_age()
            
    def get_fitness_percentiles(self, percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
        fitness_array = [individual.get_fitness() for individual in self.individuals]
        fitness_percentiles = np.percentile(fitness_array, percentiles)
        percentiles_dict = {}
        for i in range(len(percentiles)):
            percentiles_dict[str(percentiles[i])] = fitness_percentiles[i]
        return percentiles_dict
    
class Individual:
    def __init__(self, f, lower_bounds, upper_bounds, x_step, f_if, if_po, if_target, if_max_deviation):
        self.f = f
        self.n_genes = len(lower_bounds)
        self.ub = upper_bounds
        self.lb = lower_bounds
        self.x_step = x_step
        self.genes = [random.uniform(self.lb[gene] ,self.ub[gene]) for gene in range(self.n_genes)]
        self.round_genes()
        
        self.fitness = None
        self.predicted_if = None
        
        self.f_if = f_if
        self.if_po = if_po
        self.if_target = if_target
        self.if_max_deviation = if_max_deviation
        self.age = 0
        
    def get_fitness(self):
        if not self.fitness:
            self.fitness =  self.f(self.genes)
            if abs(self.get_if() - self.if_target) > self.if_max_deviation:
                self.fitness *= 1.5
        return self.fitness
    
    def get_if(self):
        if not self.predicted_if:
            self.predicted_if = self.f_if(np.append(self.genes,self.if_po))
        return self.predicted_if
    
    def mutate(self):
        random_index = random.randint(0, self.n_genes - 1)
        self.genes[random_index] = random.uniform(self.lb[random_index],self.ub[random_index])
        self.round_gene_i(random_index)
        
        self.fitness = None
        self.predicted_if = None
    
    def set_genes(self,genes):
        self.genes = genes
        self.round_genes()
        
        self.fitness = None
        self.predicted_if = None
        
    def get_age(self):
        return self.age
    
    def increase_age(self):
        self.age += 1
    
    def round_genes(self):
        for i in range(self.n_genes):
            self.round_gene_i(i)
    
    def round_gene_i(self,i):
        self.genes[i] = self.x_step[i] * round(self.genes[i]/self.x_step[i])
        