import numpy as np
from scipy.special import expit

# How many do we start with?
POPULATION_SIZE = 150

# Probability of Structural Mutation of Adding a New Node (Between Pre-Existing Nodes)
CHANCE_ADD_NODE = 0.03

# Probability of Structural Mutation of Adding a New Link Between Pre-Existing Nodes
CHANCE_ADD_LINK = 0.05

# Max weight perturbation per mutation
PERTURB_WEIGHT_DELTA = 25

# Chance we perturb weights by making them stronger during uniform perturbation
CHANCE_PERTURB_HIGHER = 0.5

# Chance we perturb weights uniformly (rather than reassigning them randomly)
CHANCE_PERTURB_UNIFORMLY = 0.9

# Chance a genome has all of its connection genes undergo perturbations
CHANCE_GENOME_CONNECTION_WEIGHTS_PERTURBED = 0.8

# Target Value
TRUE_VALUE = 0.4824

# Acceptable Margin of Error
GOOD_ENOUGH = 0.0001


def softmax(list_of_values):
    e_x = np.exp(list_of_values)
    return e_x / e_x.sum()

def chance(p):
    return bool(np.random.random() <= p)

def safe_weight():
    w = np.random.random()
    while not(w):
        w += np.random.random()
    return w

def expit_ind(arr):
    new_array = np.zeros_like(arr)
    for i in range(len(arr)):
        element = arr[i]
        if element:
            new_array[i] = expit(arr[i])
        else:
            new_array[i] = element
    return new_array

LAYER_DICT = {}
LAYER_DICT[0] = 0
LAYER_DICT[1] = 0
LAYER_DICT[2] = 0
LAYER_DICT[3] = 1
LAYER_DICT[4] = 1
LAYER_DICT[5] = 1
LAYER_DICT[6] = 1
LAYER_DICT[7] = 2
LAYER_DICT[8] = 2
LAYER_DICT[9] = 2
LAYER_DICT[10] = 2
LAYER_DICT[11] = 3

LAYER_THRESHOLDS = {}
LAYER_THRESHOLDS[0] = 3
LAYER_THRESHOLDS[1] = 7
LAYER_THRESHOLDS[2] = 11

W0_legal_link_index_pairs = [(0,3), (0,4), (0,5), (0,6), (0,7),
                             (1,3), (1,4), (1,5), (1,6), (1,7),
                             (2,3), (2,4), (2,5), (2,6), (2,7)]

W1_legal_link_index_pairs = [(0,7), (0,8), (0,9), (0,10), (0,11),
                             (1,7), (1,8), (1,9), (1,10), (1,11),
                             (2,7), (2,8), (2,9), (2,10), (2,11),
                             (3,7), (3,8), (3,9), (3,10), (3,11),
                             (4,7), (4,8), (4,9), (4,10), (4,11),
                             (5,7), (5,8), (5,9), (5,10), (5,11),
                             (6,7), (6,8), (6,9), (6,10), (6,11)]

W2_legal_link_index_pairs = [(0,11),(1,11),(2,11),(3,11),(4,11),(5,11),(6,11),(7,11),(8,11),(9,11),(10,11)]

W1_legal_node_index_pairs = [(0,7),(0,8),(0,9),(0,10),
                             (1,7),(1,8),(1,9),(1,10),
                             (2,7),(2,8),(2,9),(2,10)]

W2_legal_node_index_pairs = [(0,11),(1,11),(2,11),(3,11),(4,11),(5,11),(6,11)]

ddd = {}
ddd["s0->h00"] = (0, 0, 3)
ddd["s0->h01"] = (0, 0, 4)
ddd["s0->h02"] = (0, 0, 5)
ddd["s0->h03"] = (0, 0, 6)
ddd["s0->h10"] = (1, 0, 7)
ddd["s0->h11"] = (1, 0, 8)
ddd["s0->h12"] = (1, 0, 9)
ddd["s0->h13"] = (1, 0, 10)
ddd["s0->o0"] = (2, 0, 11)
ddd["s1->h00"] = (0, 1, 3)
ddd["s1->h01"] = (0, 1, 4)
ddd["s1->h02"] = (0, 1, 5)
ddd["s1->h03"] = (0, 1, 6)
ddd["s1->h10"] = (1, 1, 7)
ddd["s1->h11"] = (1, 1, 8)
ddd["s1->h12"] = (1, 1, 9)
ddd["s1->h13"] = (1, 1, 10)
ddd["s1->o0"] = (2, 1, 11)
ddd["s2->h00"] = (0, 2, 3)
ddd["s2->h01"] = (0, 2, 4)
ddd["s2->h02"] = (0, 2, 5)
ddd["s2->h03"] = (0, 2, 6)
ddd["s2->h10"] = (1, 2, 7)
ddd["s2->h11"] = (1, 2, 8)
ddd["s2->h12"] = (1, 2, 9)
ddd["s2->h13"] = (1, 2, 10)
ddd["s2->o0"] = (2, 2, 11)
ddd["h00->h10"] = (1, 3, 7)
ddd["h00->h11"] = (1, 3, 8)
ddd["h00->h12"] = (1, 3, 9)
ddd["h00->h13"] = (1, 3, 10)
ddd["h00->o0"] = (2, 3, 11)
ddd["h01->h10"] = (1, 4, 7)
ddd["h01->h11"] = (1, 4, 8)
ddd["h01->h12"] = (1, 4, 9)
ddd["h01->h13"] = (1, 4, 10)
ddd["h01->o0"] = (2, 4, 11)
ddd["h02->h10"] = (1, 5, 7)
ddd["h02->h11"] = (1, 5, 8)
ddd["h02->h12"] = (1, 5, 9)
ddd["h02->h13"] = (1, 5, 10)
ddd["h02->o0"] = (2, 5, 11)
ddd["h03->h10"] = (1, 6, 7)
ddd["h03->h11"] = (1, 6, 8)
ddd["h03->h12"] = (1, 6, 9)
ddd["h03->h13"] = (1, 6, 10)
ddd["h03->o0"] = (2, 6, 11)
ddd["h10->o0"] = (2, 7, 11)
ddd["h11->o0"] = (2, 8, 11)
ddd["h12->o0"] = (2, 9, 11)
ddd["h13->o0"] = (2, 10, 11)

bbb = {}
for key in ddd:
    val = ddd[key]
    bbb[val] = key

bif = {}

PARTICLES = {}
def particles(string):
    if not(string in PARTICLES):
        result = string.split("->")
        PARTICLES[string] = result
        return result
    return PARTICLES[string]

def intermediates(string_particles):
    if string_particles[0].startswith('s'):
        # Case: sa->h1b
        if (string_particles[1][-2] == "1"):
            return ['h00', 'h01', 'h02', 'h03']
        # Case: sa->o0
        return ['h00', 'h01', 'h02', 'h03', 'h10', 'h11', 'h12', 'h13']
    # Case: h0a -> o0
    return ['h10', 'h11', 'h12', 'h13']

def unify(p0, p1):
    return p0 + "->" + p1

# ddd takes an easy to reason about representation of a link and returns a tuple
# in the form (weight_matrix_index, i, j)

# bbb takes a tuple in the form (weight_matrix_index, i, j) and returns an easy to
# reason about representation of a link.

def return_weight_perturbation():
    return np.random.randint(1, (PERTURB_WEIGHT_DELTA + 1)) / 100

def perturb_weight_uniformly(weight):
    perturbation = return_weight_perturbation()
    if chance(CHANCE_PERTURB_HIGHER):
        return max(1.0, weight + perturbation)
    else:
        return min(0, weight - perturbation)

def perturb_weight_random_reallocation():
    return np.random.random()

def mutate_weight(weight):
    if chance(CHANCE_PERTURB_UNIFORMLY):
        return perturb_weight_uniformly(weight)
    else:
        return np.random.random()

def mutate_connection_weights_W0(W0):
    for pair in W0_legal_link_index_pairs:
        if chance(CHANCE_GENOME_CONNECTION_WEIGHTS_PERTURBED):
            W0[pair[0]][pair[1]] = mutate_weight(W0[pair[0]][pair[1]])

def mutate_connection_weights_W1(W1):
    for pair in W1_legal_link_index_pairs:
        if chance(CHANCE_GENOME_CONNECTION_WEIGHTS_PERTURBED):
            W1[pair[0]][pair[1]] = mutate_weight(W1[pair[0]][pair[1]])

def mutate_connection_weights_W2(W2):
    for pair in W2_legal_link_index_pairs:
        if chance(CHANCE_GENOME_CONNECTION_WEIGHTS_PERTURBED):
            W2[pair[0]][pair[1]] = mutate_weight(W2[pair[0]][pair[1]])

def mutate_connection_weights(W0, W1, W2):
    mutate_connection_weights_W0(W0)
    mutate_connection_weights_W1(W1)
    mutate_connection_weights_W2(W2)

def undergo_mutation(W0, W1, W2):
    mutate_connection_weights(W0, W1, W2)
    structural_mutation_add_link(W0, W1)
    structural_mutation_add_node(W0, W1, W2)

def legal_link_mutations(W0, W1):
    candidates = []
    for pair in W0_legal_link_index_pairs:
        if not(W0[pair[0]][pair[1]]):
            candidates.append((pair[0], pair[1], 0))

    for pair in W1_legal_link_index_pairs:
        if not(W1[pair[0]][pair[1]]):
            candidates.append((pair[0], pair[1], 1))

    if candidates:
        np.random.shuffle(candidates)
        link_mutation = candidates[0]
        return link_mutation
    return None

def structural_mutation_add_link(W0, W1, forced=False):
    if ((chance(CHANCE_ADD_LINK)) or forced):
        pair = legal_link_mutations(W0, W1)
        if (pair is not None):
            if pair[-1]:
                W0[pair[0]][pair[1]] = safe_weight()
            else:
                W1[pair[0]][pair[1]] = safe_weight()

def legal_node_query(query, W1, W2):
    if (query[0] == 1):
        matrix_to_use = W1
    else:
        matrix_to_use = W2
    return bool(not(matrix_to_use[query[0]][query[1]]))

def legal_node_mutations(W0, W1, W2):
    candidates = []
    for pair in W1_legal_node_index_pairs:
        if (W1[pair[0]][pair[1]]):
            candidates.append((1, pair[0], pair[1]))

    for pair in W2_legal_node_index_pairs:
        if (W2[pair[0]][pair[1]]):
            candidates.append((2, pair[0], pair[1]))

    if candidates:
        viable_candidates = []
        for candidate in candidates:
            candidate_string = bbb[candidate]
            candidate_particles = particles(candidate_string)
            inters = intermediates(candidate_particles)
            for inter in inters:
                pre_link_string = unify(candidate_particles[0], inter)
                pre_link_query = ddd[pre_link_string]
                post_link_string = unify(inter, candidate_particles[1])
                post_link_query = ddd[post_link_string]
                if ((legal_node_query(pre_link_query, W1, W2)) and (legal_node_query(post_link_query, W1, W2))):
                    viable_candidates.append((candidate, pre_link_query, post_link_query))
        candidates = viable_candidates
    return candidates


def structural_mutation_add_node(W0, W1, W2, forced=False):
    if ((chance(CHANCE_ADD_NODE)) or forced):
        node_mutation = legal_node_mutations(W0, W1, W2)
        if (node_mutation):
            matrices = [W0, W1, W2]
            np.random.shuffle(node_mutation)
            mutation_to_induce = node_mutation[0]
            link_to_remove = mutation_to_induce[0]
            pre_link_to_add = mutation_to_induce[1]
            post_link_to_add = mutation_to_induce[2]

            # Remove the link that's bifurcating
            temp_W = matrices[link_to_remove[0]]
            old_outbound_weight = temp_W[link_to_remove[1]][link_to_remove[2]]
            temp_W[link_to_remove[1]][link_to_remove[2]] = 0.0

            # Add the pre-link
            temp_W = matrices[pre_link_to_add[0]]
            temp_W[pre_link_to_add[1]][pre_link_to_add[2]] = 1.0

            # Add the post-link
            temp_W = matrices[post_link_to_add[0]]
            temp_W[post_link_to_add[1]][post_link_to_add[2]] = old_outbound_weight


def clone_myself(matrixnet, mutate=False):
    mini_me = MatrixNet(W0=matrixnet.W0.copy(),
                        W1=matrixnet.W1.copy(),
                        W2=matrixnet.W2.copy(),
                        x=matrixnet.x.copy(),
                        base_fitness=matrixnet.base_fitness,
                        modified_fitness=matrixnet.modified_fitness)
    if mutate:
        undergo_mutation(mini_me.W0, mini_me.W1, mini_me.W2)
    return mini_me


def pairwise_fitter_idx(n0, n1):
    if (n0.base_fitness > n1.base_fitness):
        return 0
    elif (n0.base_fitness < n1.base_fitness):
        return 1
    else:
        return np.random.randint(0,2)

def compose_networks(net0, net1):
    network_list = [net0, net1]
    fitter_idx = pairwise_fitter_idx(net0, net1)
    fitter_network = network_list.pop(fitter_idx)
    weaker_network = network_list.pop()
    offspring = clone_myself(fitter_network, mutate=False)
    for pair in W0_legal_link_index_pairs:
        weaker_value = weaker_network.W0[pair[0]][pair[1]]
        fitter_value = fitter_network.W1[pair[0]][pair[1]]
        if (not(fitter_value) and weaker_value):
            value_to_assign = weaker_value
        elif (fitter_value and not(weaker_value)):
            value_to_assign = fitter_value
        else:
            if chance(0.5):
                value_to_assign = fitter_value
            else:
                value_to_assign = weaker_value
        offspring.W0[pair[0]][pair[1]] = value_to_assign

    for pair in W1_legal_link_index_pairs:
        weaker_value = weaker_network.W1[pair[0]][pair[1]]
        fitter_value = fitter_network.W1[pair[0]][pair[1]]
        if (not(fitter_value) and weaker_value):
            value_to_assign = weaker_value
        elif (fitter_value and not(weaker_value)):
            value_to_assign = fitter_value
        else:
            if chance(0.5):
                value_to_assign = fitter_value
            else:
                value_to_assign = weaker_value
        offspring.W1[pair[0]][pair[1]] = value_to_assign

    for pair in W2_legal_link_index_pairs:
        weaker_value = weaker_network.W2[pair[0]][pair[1]]
        fitter_value = fitter_network.W2[pair[0]][pair[1]]
        if (not(fitter_value) and weaker_value):
            value_to_assign = weaker_value
        elif (fitter_value and not(weaker_value)):
            value_to_assign = fitter_value
        else:
            if chance(0.5):
                value_to_assign = fitter_value
            else:
                value_to_assign = weaker_value
        offspring.W2[pair[0]][pair[1]] = value_to_assign

    undergo_mutation(offspring.W0, offspring.W1, offspring.W2)
    return offspring


class MatrixNet:
    def __init__(self, **kwargs):
        self.refresh_attribute_values()
        self.base_fitness = 1.0
        self.modified_fitness = 1.0
        self.network_idx = None
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

    def initial_construction(self):
        self.W2[0][11] = safe_weight()
        self.W2[1][11] = safe_weight()
        self.W2[2][11] = safe_weight()
        self.x[0] = safe_weight()
        self.x[1] = safe_weight()
        self.x[2] = safe_weight()


    def n_synapses(self):
        n = 0
        n += len(np.ravel(self.W0[self.W0!=0]))
        n += len(np.ravel(self.W1[self.W1!=0]))
        n += len(np.ravel(self.W2[self.W2!=0]))
        return n

    def refresh_attribute_values(self):
        self.W0 = np.zeros((12,12))
        self.W1 = np.zeros((12,12))
        self.W2 = np.zeros((12,12))
        self.x = np.zeros((12,))
        self.h0 = np.zeros((12,))
        self.h1 = np.zeros((12,))
        self.o = np.zeros((12,))
        self.x_ = None
        self.h0_ = None
        self.h1_ = None

    def receive_input_values(self, input_values):
        assert (len(input_values) == len(self.x))
        for i, value in enumerate(input_values):
            self.x[i] = value

    def compute_output(self):
        # Hit x with non-linearity
        self.x_ = expit_ind(self.x)
        # Compute h0
        self.h0 = np.dot(self.W0.T, self.x_)
        # Hit h0 with a non-linearity
        self.h0_ = expit_ind(self.h0)
        # Compute h1
        self.h1 = np.dot(self.W1.T, self.x_)
        self.h1 = self.h1 + np.dot(self.W1.T, self.h0_)
        # Hit h1 with a non-linearity
        self.h1_ = expit_ind(self.h1)
        # Compute o
        self.o = np.dot(self.W2.T, self.x_)
        self.o = self.o + np.dot(self.W2.T, self.h0_)
        self.o = self.o + np.dot(self.W2.T, self.h1_)
        # Compute the final_value of o
        final_value = np.sum(self.o)
        return expit(final_value)


class Simulation:
    def __init__(self, **kwargs):
        self.unique_network_idx = 0
        self.genomes = []
        self.species_exemplars = []
        self.species_list_of_sublists_of_genomes = []
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])
        self.winner = None

    def accept_network(self, network_to_accept):
        if (network_to_accept.network_idx is None):
            self.unique_network_idx += 1
            network_to_accept.network_idx = self.unique_network_idx
        self.genomes.append(network_to_accept)

    def speciate(self, genome):
        ''' The Genome passed to this function has been deemed to be representative of a new species. '''
        self.species_exemplars.append(genome)
        self.species_list_of_sublists_of_genomes.append([genome])

    def solve_species(self):
        self.species_list_of_sublists_of_genomes.clear()
        self.species_exemplars.clear()
        temp_all_species = list(self.genomes)
        sorted_temp = sorted(temp_all_species, key=lambda g: g.n_synapses())
        n_sorted_temp = len(sorted_temp)
        i0 = int(np.floor(n_sorted_temp / 5))
        i1 = i0 * 2
        i2 = i0 * 3
        i3 = i0 * 4
        s0 = sorted_temp[:i0]
        s1 = sorted_temp[i0:i1]
        s2 = sorted_temp[i1:i2]
        s3 = sorted_temp[i2:i3]
        s4 = sorted_temp[i3:]
        self.species_exemplars.append(np.random.choice(s0))
        self.species_exemplars.append(np.random.choice(s1))
        self.species_exemplars.append(np.random.choice(s2))
        self.species_exemplars.append(np.random.choice(s3))
        self.species_exemplars.append(np.random.choice(s4))
        self.species_list_of_sublists_of_genomes.append(s0)
        self.species_list_of_sublists_of_genomes.append(s1)
        self.species_list_of_sublists_of_genomes.append(s2)
        self.species_list_of_sublists_of_genomes.append(s3)
        self.species_list_of_sublists_of_genomes.append(s4)


    def compute_shared_fitness(self):
        for species in self.species_list_of_sublists_of_genomes:
            total_species_fitness = 0
            n_genome = 0

            for genome in species:
                n_genome += 1
                total_species_fitness += genome.base_fitness

            if n_genome:
                modified_fitness = total_species_fitness / n_genome
                for genome in species:
                    genome.modified_fitness = modified_fitness


    def evaluate_fitness(self, genome):
        fitness = self.train_phenotype(genome)
        genome.base_fitness = fitness
        if not(genome.base_fitness is None):
            if (fitness <= GOOD_ENOUGH):
                self.winner = genome
                self.winner_fitness = fitness


    def find_highest_base_fitness(self, list_of_genomes):
        sorted_genomes = sorted(list_of_genomes, key=lambda g: g.base_fitness, reverse=True)
        best_fitness = sorted_genomes[0].base_fitness
        champions = list(filter(lambda genome: (genome.base_fitness == best_fitness), sorted_genomes))
        return np.random.choice(champions)

    def deduce_champion(self, species):
        best_fitness = species[0].base_fitness
        champions = list(filter(lambda genome: (genome.base_fitness == best_fitness), species))
        np.random.shuffle(champions)
        return champions[0]

    def train_phenotype(self, phenotype):
        attempt = phenotype.compute_output()
        delta = np.abs(TRUE_VALUE - attempt)
        if (delta <= GOOD_ENOUGH):
            self.winner = phenotype
            self.winner_attempt = attempt
            return 5 - delta
        else:
            phenotype.base_fitness = 5 - delta
            return phenotype.base_fitness

    def evaluate_fitnesses(self):
        # This method needs to manifest each genotype, make it perform the training tasks,
        # and return a scalar representing its fitness.
        for genome in self.genomes:
            self.evaluate_fitness(genome)

    def statistics(self):
        vector = [genome.base_fitness for genome in self.genomes]
        vector = [val for val in vector if (val is not None)]
        if vector:
            M = np.mean(vector)
            SD = np.std(vector)
            max_ = np.max(vector)
            min_ = np.min(vector)
            print("M: {} SD: {} Max: {} Min: {}".format(M, SD, max_, min_))
        else:
            print("No genomes had non-null base_fitness attribute_value")


    def sort_species(self, list_of_genomes):
        return sorted(list_of_genomes, key=lambda g: g.base_fitness, reverse=True)


    def mean_fitness(self, species):
        return np.mean([genome.base_fitness for genome in species])

    def generate_generation(self):
        '''\
            We evaluate all genomes in self.genomes by manifesting a phenotype and training them on task.
            We keep the best genome in each species automatically if that species is of large enough size.
            We decide which 25% of the genomes will be continued through strictly with mutation.
            We match up as many of the remaining 75% of genomes as we can, and those we can't get added
            to the strictly through mutation list.
            We induce the mutants to mutate and keep them.
            We induce the pairs to mate and keep their offspring.
            We randomly remove members of the current generation until we've maintain the overall population
            size.
            We make self.genomes the list of genomes in the new generation.
        '''
        if (self.winner is None):
            new_generation = []

            # Manifest phenotypes for each genome and figure out their fitness
            self.evaluate_fitnesses()

            # Printout to see how things are going
            self.statistics()

            # Solve the species
            self.solve_species()

            # Sort species in descending order in terms of fitness
            sorted_species = []
            for species in self.species_list_of_sublists_of_genomes:
                sorted_species.append(self.sort_species(species))
            self.species_list_of_sublists_of_genomes = sorted_species

            # Determine the champions of each species
            champions = []
            for species in self.species_list_of_sublists_of_genomes:
                champion = self.deduce_champion(species)
                champions.append(champion)

            n_champions = len(champions)
            n_to_make = POPULATION_SIZE - n_champions

            n_to_self_replicate = int(np.floor((n_to_make / 4)))
            n_to_breed = n_to_make - n_to_self_replicate

            mean_fitnesses = [self.mean_fitness(species) for species in self.species_list_of_sublists_of_genomes]
            clone_sample_pdist = softmax(mean_fitnesses)
            n_samples = n_to_self_replicate
            sample_with_replacement = True
            species_index_list = list(range(len(self.species_list_of_sublists_of_genomes)))
            self_replicating_species = np.random.choice(a=species_index_list,
                                                        size=n_samples,
                                                        replace=sample_with_replacement,
                                                        p=clone_sample_pdist)
            self_replicators = []
            for species_idx in self_replicating_species:
                species = self.species_list_of_sublists_of_genomes[species_idx]
                self_replicators.append(np.random.choice(species))

            species_that_can_mate = list(filter(lambda species: (len(species) > 1), self.species_list_of_sublists_of_genomes))
            mean_fitnesses = [self.mean_fitness(species) for species in species_that_can_mate]
            mating_sample_pdist = softmax(mean_fitnesses)
            n_samples = n_to_breed
            sample_with_replacement = True
            species_index_list = list(range(len(species_that_can_mate)))
            mating_species = np.random.choice(a=species_index_list,
                                              size=n_samples,
                                              replace=sample_with_replacement,
                                              p=mating_sample_pdist)
            mating_pairs = []
            for species_idx in mating_species:
                species = species_that_can_mate[species_idx]
                species_clone = [genome for genome in species]
                mates = np.random.choice(a=species, size=2, replace=False)
                mating_pairs.append((mates[0], mates[1]))

            # Add the champions to the new generation as mutation-free clones
            for champion in champions:
                champion_clone = clone_myself(champion, mutate=False)
                new_generation.append(champion_clone)

            # Add the clones to the new generation
            for self_replicator in self_replicators:
                clone = clone_myself(self_replicator, mutate=True)
                new_generation.append(clone)

            # Add the offspring of the mating pairs to the new generation
            for pair in mating_pairs:
                offspring = compose_networks(pair[0], pair[1])
                new_generation.append(offspring)

            n_to_delete = len(self.genomes)
            for i in range(n_to_delete):
                if self.genomes:
                    genome_to_delete = self.genomes.pop()
#                    genome_to_delete.self_destruct()
                    del genome_to_delete

            self.genomes = []
            for genome in new_generation:
                self.accept_network(genome)

        else:
            print("We have a winner.")




SIM = Simulation()
for i in range(POPULATION_SIZE):
    new_network = MatrixNet()
    new_network.initial_construction()
    SIM.accept_network(new_network)

import time
start_time = time.time()
for i in range(1000):
    print("Generation {} [{}]".format(i, time.time()-start_time))
    if (SIM.winner is None):
        SIM.generate_generation()
    else:
        break
if (SIM.winner is not None):
    print("We have a winner, with return value: {} (Target: {})".format(SIM.winner_attempt, TRUE_VALUE))
else:
    print("Failed to evolve a network able to approximate target value.")
