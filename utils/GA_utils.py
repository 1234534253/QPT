from deap import base
from deap import creator
from deap import tools
import numpy as np
import random
import math


def off_ind(icls, ind_as_list):
    """
    :param ind_as_list: individual as list
    :return: individual as DEAP object
    """
    return icls(ind_as_list)


def generate_ind_from_vec(icls, vec_norm):
    ind = icls()
    ind.append(vec_norm[0])
    ind.append(vec_norm[1])
    ind.append(vec_norm[2])
    ind.append(vec_norm[3])

    return ind


def near_square(starting_point, d, im_size_x, im_size_y):
    d_start = 1
    rows, columns = starting_point[0] + d_start, starting_point[1] + d_start
    pixels_to_consider = []
    starting_row, starting_col = (
        starting_point[0] - d_start,
        starting_point[1] - d_start,
    )
    if starting_row < 0:
        starting_row = 0
    if starting_col < 0:
        starting_col = 0
    final_row, final_col = starting_point[0] + d_start, starting_point[1] + d_start
    if final_row > im_size_x - 1:
        final_row = im_size_x - 1
    if final_col > im_size_y - 1:
        final_col = im_size_y - 1
    for i in range(starting_row, final_row + 1):
        for j in range(starting_col, final_col + 1):
            if (
                abs(i - starting_point[0]) == d_start
                or abs(j - starting_point[1]) == d_start
            ):
                pixels_to_consider.append((i, j))
    final = []
    if d > 1:
        for p in pixels_to_consider:
            for k in near_square(p, 1, im_size_x, im_size_y):
                final.append(k)
    else:
        final = pixels_to_consider

    f = list(set(final))
    if starting_point in f:
        f.remove(starting_point)

    return f


def pop_from_near_square(
    icls,
    pixel,
    pop_size,
    im_size_x,
    im_size_y,
    best_ind_pixels,
    epsilon=0.2,
    d=2,
):
    pixels = near_square(pixel, d, im_size_x, im_size_y)
    pix_pop = []
    pops = []
    for p in pixels:
        if best_ind_pixels[str(p[0]) + "_" + str(p[1])] != "":
            pops.append(best_ind_pixels[str(p[0]) + "_" + str(p[1])])
    c = 0
    for ind in range(pop_size):
        if len(pops) - 1 != 0:
            pop_ = pops[random.randint(0, len(pops) - 1)]
        else:
            pop_ = pops[0]
        ind = pop_.copy()
        for i in range(len(ind)):
            ind[i] = ind[i] + random.uniform(-epsilon, epsilon)
            if i == 0:
                ind[i] = ind[i] % np.pi
        vec = ind[1:]
        vec_norm = vec / np.linalg.norm(vec)

        ind_new = generate_ind_from_vec(
            icls=icls, vec_norm=[ind[0], vec_norm[0], vec_norm[1], vec_norm[2]]
        )
        pix_pop.append(ind_new)
        c = c + 1
    return pix_pop


def generate_ind(icls):
    """
    Function generating a candidate solution
    ________________________________________
    :return individual as un-normalized list
    """
    ind = icls()
    ind.append(random.uniform(0, np.pi))
    vec = np.random.rand(3)
    vec_norm = vec / np.linalg.norm(vec)
    ind.append(vec_norm[0])
    ind.append(vec_norm[1])
    ind.append(vec_norm[2])
    return ind


def norm_list(list_):
    """
    Function normalizing a list to its square norm
    :param list_: list
    :return: normalize np.array
    """
    vec = np.array(list_)
    return vec / np.linalg.norm(vec, ord=2)


def checkBounds(min_i, max_i):
    """
    Function rescaling the Energy of a solution in the range [min_i, max_i]
    and normalizing the trivector.
    :param min_i: min possible value of Energy
    :param max_i: max possible value of Energy
    :return: Possible solution respecting the constraints
    """

    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                if child[0] < min_i:
                    child[0] = min_i
                if child[0] > max_i:
                    child[0] = child[0] % math.pi
                norm_ind = norm_list(child[1:])
                for i in range(1, 4):
                    child[i] = norm_ind[i - 1]
            return offspring

        return wrapper

    return decorator


def createToolbox():
    """
    Setup of operators used in the Genetic Algorithm
    :return: DEAP toolbox
    """
    # Creating Classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    # Individual
    toolbox.register("individual", generate_ind, creator.Individual)
    # Population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "population_from_near_square", pop_from_near_square, creator.Individual
    )

    # Crossover
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    # Mutation
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.20)
    # Selection
    toolbox.register("select_TS", tools.selTournament)

    return toolbox


def createStats():
    """
    Setup of Statistics for Genetic Algorithm
    :return: Statistic tool of DEAP
    """
    # Statistical Features
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def updatedGA(
    toolbox, pop_size, cxpb, mutpb, ngen, stats, hof, tourn_size, verbose=False
):
    """
    ___________________________________
    Executes the Genetic Algorithm
    ____________________________________
    :param toolbox: DEAP toolbox object
    :param pop_size: Population size
    :param cxpb: Crossover Probability
    :param mutpb: Mutation Probability
    :param ngen: Number of Generations
    :param stats: DEAP stats object
    :param hof: Hall of Fame DEAP object
    :param tourn_size: Tournament size for Tournament selection
    :param verbose: True for printing the stats during the evoultion
    :return: Final Population, Logbook, Hall of Fame
    """
    # Creating the population
    pop = toolbox.population(n=pop_size)

    # Defining the Logbook
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the entire population
    fitness = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = [fit]

    hof.update(pop) if stats else {}

    record = stats.compile(pop) if stats else {}

    logbook.record(gen=0, nevals=len(pop), **record)
    if verbose:
        print(logbook.stream)
    # Starting the evolution
    for g in range(ngen):
        # Elitism
        bests = toolbox.clone(tools.selBest(pop, 1))
        elitist = bests[0]

        # Select the next generation individuals

        offspring = toolbox.select_TS(pop, k=pop_size, tournsize=tourn_size)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant, indpb=0.6)
                del mutant.fitness.values

        # Evaluate the entire population
        fitness = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitness):
            ind.fitness.values = [fit]

        # The population is entirely replaced by the offspring and the elitist
        pop[:] = tools.selBest(offspring, pop_size - 1)
        pop.append(elitist)

        # Update stats and hof
        hof.update(pop) if stats else {}
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=g + 1, **record)
        if verbose:
            print(logbook.stream)

    return pop, logbook, hof


def updateGA_Map(
    toolbox,
    pixel,
    best_ind_pixels,
    im_size_x,
    im_size_y,
    pop_size,
    cxpb,
    mutpb,
    ngen,
    stats,
    tourn_size=None,
    hof=tools.HallOfFame(1),
    starting_point=True,
    verbose=False,
):
    # Creating the population
    if starting_point:
        pop = toolbox.population(n=pop_size)
    else:
        pop = toolbox.population_from_near_square(
            pixel=pixel,
            pop_size=pop_size,
            best_ind_pixels=best_ind_pixels,
            im_size_x=im_size_x,
            im_size_y=im_size_y,
        )

    # Defining the Logbook
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the entire population
    fitness = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = [fit]

    hof.update(pop) if stats else {}

    record = stats.compile(pop) if stats else {}

    logbook.record(gen=0, nevals=len(pop), **record)
    if verbose:
        print(logbook.stream)

    for g in range(ngen):

        # elitism
        bests = toolbox.clone(tools.selBest(pop, 1))
        elitist = bests[0]
        # print(elitist.fitness)

        # Select the next generation individuals

        offspring = toolbox.select_TS(pop, k=pop_size, tournsize=tourn_size)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant, indpb=0.6)
                del mutant.fitness.values

        # Evaluate the entire population
        fitness = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitness):
            ind.fitness.values = [fit]

        # The population is entirely replaced by the offspring
        # pop = offspring
        pop[:] = tools.selBest(offspring, pop_size - 1)
        pop.append(elitist)

        hof.update(pop) if stats else {}

        record = stats.compile(pop) if stats else {}

        # print(record)
        logbook.record(gen=g + 1, **record)
        if verbose:
            print(logbook.stream)

    # print(final_pop_pixels)
    # dict = final_pop_pixels[pixel] = pop
    # best_ind_pixels[pixel] = hof[0]

    return pop, logbook, hof
