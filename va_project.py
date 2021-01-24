#imports 
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv 

#makes the reference individual(with no pores) at costum lenght
def best_individual(chromosome_length):
    reference = np.zeros(chromosome_length)
    reference= reference.astype(int)
    reference= reference.tolist()
    return reference

#generates initial random population
def initial_population(individuals, chromosome_length):
    population = np.zeros((individuals, chromosome_length))

    for i in range(individuals):
        ones = random.randint(0, chromosome_length)
        population[i, 0:ones] = 1
        np.random.shuffle(population[i])
        
    population=population.astype(int)
    population=population.tolist()
    
    return population

#calculates the fitness score
def fitness(reference, individual):
    score=0
    
    size_ind=len(individual)
    for x in range(size_ind):
        if individual[x]== reference[x]:
            score+=1
    fitness_scores = (score/number_pores)*100
        
    return fitness_scores

#mutates randomly the population with desired mutation rate
def mutation(population, mutation_probability):
    
     population=np.asarray(population)
     random_mutation_array = np.random.random(
     size=(population.shape))
        
     random_mutation_boolean = \
     random_mutation_array <= mutation_probability

     population[random_mutation_boolean] = \
     np.logical_not(population[random_mutation_boolean])
             
     population=population.tolist()
      
     return population

#converts example files to the correct format
def example_preparation(filename,tipo):
    with open(filename, 'rU') as p:
        my_list = [list(map(tipo,rec)) for rec in csv.reader(p, delimiter=',')]
      
        return my_list

#defines paramethers
number_pores =20
population_size = int(input('What is desired population size?'))
generations = int(input('For how many generations do you wish our GA to run?'))
mutation_rate=float(input('What is the desired mutation rate?'))

#creates reference solution 
reference = best_individual(number_pores)

#creates initial population
population=initial_population(population_size, number_pores)

#replace the initial population with our example
#population= example_preparation('initial_population_example.csv',int)


#creates empty datastructures to be populated later
gen_size=[]
gens={}
scores=[]
mean_score_progress=[]
best_score_progress = []

#Loops trough each generation 
for gen in range(generations):
   
    size=len(population)
    gen_size.append(size)
    
    gen_pop=[]
    for i in range(size):
        ind=population[i]
        
             
        score=fitness(reference, ind)
        scores.append(score)
        gen_pop.append(ind)
        if score>90:
            population.append(ind)
            
      
    population = mutation(population, mutation_rate)
    gens[gen]=gen_pop
    mean_score = np.mean(scores)
    mean_score_progress.append(mean_score)
    best_score = np.max(scores)
    best_score_progress.append(best_score)

#replace the final population with our example
#population= example_preparation('final_population_example.csv',int)

#counts each genotypes clones
output = {} 
for lis in population: 
    output.setdefault(tuple(lis), list()).append(1) 
for a, b in output.items(): 
    output[a] = sum(b) 
    
#replace the mean score with our example
#mean_score_progress= example_preparation('mean_score_example.csv',float)

#replace the best score with our example
#best_score_progress= example_preparation('best_score_example.csv',float)   

#filters the genotypes that generated a number below the average number of clones
df=pd.DataFrame.from_dict(output, orient='index')
df=df.reset_index()
df=df.rename(columns={"index": "individual type", 0: "clones_number"})
mean=df.clones_number.mean()
df = df[df.clones_number >= mean]

#plots the filtered number of clones per genotype
ax = df.plot.bar()

#dt=pd.DataFrame.from_dict(gens, orient='index')
#dt=pd.read_csv('individuals_per_generation_example.csv')
#dt=dt.reset_index()



#plots the mean fitness score per generation
plt.figure(2)
plt.plot(mean_score_progress)
plt.xlabel('Generation')
plt.ylabel('Mean fitness score %')

plt.show()

#plots the best fitness score per generation
plt.figure(3)
plt.plot(best_score_progress)
plt.xlabel('Generation')
plt.ylabel('Best fitness score %')

plt.show()

print(mean)

