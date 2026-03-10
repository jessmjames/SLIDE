import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy.optimize import curve_fit

### DIRECTED EVOLUTION ITERATOR -------------------------------------------------------------------------------------

default_extra_info = {
    "fitness": lambda *args, **kwargs : kwargs["fitnesses"],
    "pop": lambda *args, **kwargs : kwargs["pop"]
}

def run_directed_evolution(rng,
                        i_pop,
                        selection_function,
                        mutation_function, 
                        fitness_function, 
                        num_options = 2,
                        fitness_noise = 1e-4,
                        num_steps=30, 
                        extra_function_dict = default_extra_info):
    
    """
    Function to run a directed evolution simulation.

    Parameters:
    - rng: Jax random number key (e.g jax.random.PRNGKey(0)).
    - i_pop: Initial starting population. An array of dimensions (popsize, N).
    - selection_function: Generated from build_selection_function. Takes in a population and output true/false values of selection.
    - mutation_function: Genreated from build_mutation_function. Takes in a population and applies mutation.
    - fitness_function: Generated from build_NK_landscape_function/build_empirical_function. Takes in population sequences and outputs fitness values.
    - fitness_noise: Noise to be applied to fitness values.
    - num_steps: The number of mutation-selection iterations to be performed.
    - output_dict: Dictionary of information to be outputted.

    Returns:
    - (Customisable) dictionary of populations and fitness values.
    """
    
    ######################################
    ## Function for a single iteration. ##
    ######################################
    
    def single_iteration(rng,
                         population, 
                         selection_function, 
                         mutation_function, 
                         fitness_function, 
                         num_options,
                         fitness_noise=1e-4,
                         extra_function_dict = {}):

        r1, r2, r3 = jr.split(rng, 3)
        
        # Get population fitness values.
        population_fitness = fitness_function(population) # Calculate population fitness values.
        fitness_noise =  population_fitness*jr.normal(r1, population_fitness.shape)*fitness_noise # Add noise to fitness readings.
        population_fitness = population_fitness + fitness_noise

        # Perform selection on the population.
        selected_population = selection_function(r2, population_fitness)
        resamped_pop = population[selected_population]

        # Apply mutation.
        mutated_pop = mutation_function(r3, resamped_pop)
        
        current_info_dict = {"fitnesses" : population_fitness, "pop" : population}
        extra_info = {key: func(**current_info_dict) for key, func in extra_function_dict.items() }
        return mutated_pop, extra_info

    ###############################################################
    ## Multiple iterations with Jax lax scan for GPU efficiency. ##
    ###############################################################

    def multiple_iterations(population, local_rng):

        new_pop, extra_info = single_iteration(local_rng, 
                                   population, 
                                   selection_function,
                                   mutation_function, 
                                   fitness_function,
                                   num_options=num_options,
                                   fitness_noise=fitness_noise,
                                   extra_function_dict = extra_function_dict)
        
        return ((new_pop) , extra_info)
    
    rng_array = jr.split(rng, num_steps)

    return jax.lax.scan(multiple_iterations,(i_pop),rng_array)


### LANDSCAPE GENERATION -------------------------------------------------------------------------------------------------

def build_empirical_landscape_function(landscape):
    """
    Looks up fitness values on pre-defined landscape such as GB1.

    Parameters:
    - landscape: N-dimensional empirical landscape array. (e.g. GB1 shape 20x20x20x20).

    Returns:
    - Function that takes GB1 sequence as input, and returns fitness value.
    """

    def get_fitness(i):
        return landscape[tuple(i)]

    return jax.jit(jax.vmap(get_fitness))


### CODON MAPPING UTILITIES ----------------------------------------------------------------------------------------

# Uses ordering [U, C, A, G] for nucleotide.

GB1_acid_start = jnp.array([3, 17, 0, 3], dtype=jnp.int32)
GB1_codon_start = jnp.array(
    [3, 0, 0, 3, 0, 2, 3, 0, 3, 3, 0, 0], dtype=jnp.int32
)

CODON_MAPPER = jnp.array(
    [
        [[8, 18, 9, 7], [8, 18, 9, 7], [4, 18, -1, -1], [4, 18, -1, 10]],
        [[4, 1, 11, 13], [4, 1, 11, 13], [4, 1, 14, 13], [4, 1, 14, 13]],
        [[5, 19, 15, 18], [5, 19, 15, 18], [5, 19, 12, 13], [6, 19, 12, 13]],
        [[3, 2, 17, 0], [3, 2, 17, 0], [3, 2, 16, 0], [3, 2, 16, 0]],
    ],
    dtype=jnp.int32,
)

INVERSE_CODON_MAPPER = jnp.array(
    [
        [3, 0, 3],  # 0: Trp (W)
        [1, 0, 1],  # 1: Cys (C)
        [3, 0, 1],  # 2: Arg (R) subset
        [3, 0, 0],  # 3: Ser (S) subset
        [0, 2, 0],  # 4: Tyr (Y)
        [2, 0, 0],  # 5: His (H)
        [2, 3, 0],  # 6: Gln (Q)
        [0, 0, 3],  # 7: Phe (F)
        [0, 0, 0],  # 8: Leu (L) subset
        [0, 0, 2],  # 9: Leu (L) subset
        [0, 3, 3],  # 10: Leu (L) subset
        [1, 0, 2],  # 11: Pro (P)
        [2, 2, 2],  # 12: Leu (L) subset
        [1, 0, 3],  # 13: Pro (P) subset
        [1, 2, 2],  # 14: Pro (P) subset
        [2, 0, 2],  # 15: Arg (R) subset
        [3, 2, 2],  # 16: Arg (R) subset
        [3, 0, 2],  # 17: Arg (R) subset
        [0, 0, 1],  # 18: Ser (S) subset
        [2, 0, 1],  # 19: Arg (R) subset
    ],
    dtype=jnp.int32,
)

# Backward-compatible alias for the original typo.
INVERSE_CODON_MAPER = INVERSE_CODON_MAPPER


def inverse_codon(codon):
    """
    Returns the inverse (codon triplet) of an amino-acid index.
    Returns [-1, -1, -1] for invalid indices (e.g., stop codons).
    """
    codon = jnp.asarray(codon)
    max_index = INVERSE_CODON_MAPPER.shape[0] - 1
    valid = (codon >= 0) & (codon <= max_index)
    safe_codon = jnp.clip(codon, 0, max_index)
    return jnp.where(valid[..., None], INVERSE_CODON_MAPPER[safe_codon], -1)


def get_pre_defined_landscape_function_with_codon(landscape):
    """
    Looks up fitness values on a pre-defined landscape defined by amino acids,
    using the codon look-up method. Assumes each dimension has size 20.
    Stop codons (index -1) are mapped to min fitness.
    """
    n = len(landscape.shape)
    min_fitness = jnp.min(landscape)
    buffered_landscape = jnp.pad(landscape, [(0, 1)] * n, constant_values=min_fitness)

    def get_codon(i):
        return CODON_MAPPER[tuple(i)]

    vmapped_get_codons = jax.jit(jax.vmap(get_codon))

    def get_fitties(params):
        parries_reshaped = jnp.reshape(params, (-1, 3))
        codon_set = vmapped_get_codons(parries_reshaped)
        return buffered_landscape[tuple(codon_set)]

    return jax.jit(jax.vmap(get_fitties))


def get_pd_landscape_function_codon_masked(landscape, mask, replacement):
    """
    Like get_pre_defined_landscape_function_with_codon but with a site mask
    applied before fitness lookup.
    """
    n = len(landscape.shape)
    min_fitness = jnp.min(landscape)
    buffered_landscape = jnp.pad(landscape, [(0, 1)] * n, constant_values=min_fitness)

    def get_codon(i):
        return CODON_MAPPER[tuple(i)]

    vmapped_get_codons = jax.jit(jax.vmap(get_codon))

    def get_fitties(params):
        new_params = jnp.where(mask, replacement, params)
        parries_reshaped = jnp.reshape(new_params, (-1, 3))
        codon_set = vmapped_get_codons(parries_reshaped)
        return buffered_landscape[tuple(codon_set)]

    return jax.jit(jax.vmap(get_fitties))


def convert_landscape_function_to_codon(
    landscape_function,
    stop_codon_strategy="min_fitness",
    stop_codon_value=None,
    stop_codon_index=20,
    sample_size=256,
    sample_extra=0.0,
    rng=jr.PRNGKey(0),
):
    """
    Converts a landscape function defined on amino acids to one defined on codons.

    stop_codon_strategy options:
      min_fitness: uses the minimum fitness (estimated by sampling if stop_codon_value not provided).
      fill_fitness: uses stop_codon_value for any sequence containing a stop codon.
      another_amino_acid: treats stop codons as a 21st amino acid (index stop_codon_index).
      sample_min_fitness: samples values, takes the min, adds sample_extra.
    """
    valid_strategies = {"min_fitness", "fill_fitness", "another_amino_acid", "sample_min_fitness"}
    if stop_codon_strategy not in valid_strategies:
        raise ValueError(
            f"stop_codon_strategy must be one of {sorted(valid_strategies)} (got {stop_codon_strategy})."
        )
    if stop_codon_strategy == "fill_fitness" and stop_codon_value is None:
        raise ValueError("stop_codon_value must be provided for fill_fitness.")

    def convert_gene(gene):
        # gene shape: (n_sites * 3,) nucleotides
        reshaped = gene.reshape(-1, 3)
        codon_set = CODON_MAPPER[tuple(reshaped.T)]
        return codon_set

    vmapped_convert_gene = jax.jit(jax.vmap(convert_gene))

    def codon_landscape_function(pop):
        # pop: (popsize, n_sites * 3) nucleotides
        codon_genes = vmapped_convert_gene(pop)
        return landscape_function(codon_genes)

    return codon_landscape_function


def build_NK_landscape_function(rng, N, K, fitness_distribution=jr.normal):
    """
    Looks up fitness values on an NK landscape.

    Parameters:
    - rng: Jax random number key (e.g jax.random.PRNGKey(0)).
    - N: integer value representing number of sites in the gene.
    - K: integer value representing the number of interactions per site.
    - fitness_distribution: Distribution from which individual fitness values are sampled.

    Returns:
    - Function that takes NK gene sequence as input, and returns fitness value.
    """

    r1, r2, r3 = jr.split(rng, 3)

    ##################################
    ## Generate interaction matrix. ##
    ##################################

    base_row = 1* (jnp.arange(N-1) < K) # Array of size N, with K entries = 1.

    def permutate_rows(rng, i):
        perm_row = jr.permutation(rng, base_row)
        return jnp.insert(perm_row, i, 1.0)
    
    permutate_rows = jax.vmap(permutate_rows)
    interaction_matrix = permutate_rows(jr.split(r1, N), jnp.arange(N))

    ##########################################
    ## Build function for sampling from NK. ##
    ##########################################
    
    # Function for generating rng keys, ensuring that they are derived from the same base rng.
    vector_foldin = jax.vmap(lambda base_rng, data: jr.fold_in(base_rng, data))
    
    fitness_distribution = jax.vmap(fitness_distribution)

    def get_fitness(gene):
        individual_site_fitness = vector_foldin(jr.split(r2, N), gene)
        interaction_fitness = (interaction_matrix @ individual_site_fitness) + jr.split(r3, N)
        return jnp.sum(fitness_distribution(interaction_fitness))
    
    return jax.jit(jax.vmap(get_fitness))


### MUTATION FUNCTION ---------------------------------------------------------------------------------------------

def build_mutation_function(mutation_chance, num_options=2):
    """
    Generates a function that mutates a population of gene sequence.

    Parameters:
    - mutation_chance: Probability of a mutation per site.
    - num_options: Number of possibilities each site can be. Defaults to 2 for [0,1] options.

    Returns:
    - Function that applies mutations to a whole population of gene sequences.
    """

    def mutation_function(rng, pop):
        r1, r2 = jr.split(rng, 2)
        pshape = pop.shape
        has_mutation = jr.bernoulli(r1, mutation_chance, pshape)
        mut_delta = jr.randint(r2, pshape, 1, num_options)
        return (pop + (has_mutation*mut_delta)) % num_options
    return mutation_function


def build_custom_mutation_function(mutation_chance, transition_matrix, A=None):
    """
    Generates a mutation function that uses a custom transition matrix
    (e.g. empirical nucleotide substitution rates).

    Parameters:
    - mutation_chance: Probability of a mutation per site.
    - transition_matrix: (A x A) array of transition probabilities between states.
    - A: Number of states (inferred from transition_matrix if not provided).

    Returns:
    - Function that applies mutations to a whole population of sequences.
    """
    if A is None:
        A = transition_matrix.shape[0]

    transition_matrix = jnp.array(transition_matrix)

    def mutation_function(rng, pop):
        r1, r2 = jr.split(rng, 2)
        pshape = pop.shape
        has_mutation = jr.bernoulli(r1, mutation_chance, pshape)

        flat_pop = pop.flatten()
        pop_size = flat_pop.shape[0]

        def mutate_site(rng, aa):
            return jr.choice(rng, jnp.arange(A), p=transition_matrix[aa])

        flat_mutated_pop = jax.vmap(mutate_site)(jr.split(r2, pop_size), flat_pop)
        mutated_pop = flat_mutated_pop.reshape(pshape)

        return jnp.where(has_mutation, mutated_pop, pop)

    return mutation_function


### SELECTION FUNCTION ---------------------------------------------------------------------------------------------

def build_selection_function(selection_function_shape, params):
    """
    Function for determining which cells of a population are selected.

    Parameters:
    - selection_function: A function that takes in an array of fitness values, and outputs an array of probabilities of selection.
                          Options in selection_function_library.
    - params: Dictionary containing the parameters defining the shape of the selection_function.

    Returns:
    - Array of true/false values describing which population members are selected.
    """

    def selection_function(rng, fitnesses, state=0):
        psize = fitnesses.shape[-1]
        selection_prob = selection_function_shape(fitnesses, params)
        selected = jnp.ones((psize,))*jr.bernoulli(rng,
                                                   p=selection_prob, shape=(psize,))

        return jr.choice(rng, jnp.arange(psize), (psize,), p=selected)
    
    return selection_function

### PARAMETER SAMPLING --------------------------------------------------------------------------------------------

def param_sampler(*ranges, rng, num_samples=10):

    samples = jnp.array(
        [jnp.linspace(range[0], range[1], num=num_samples) for range in ranges])
    return jr.permutation(rng, samples, axis=1, independent=True)

def grid_sampler(x, y, num_samples=10):
    x = jnp.linspace(x[0], x[1], num=num_samples)
    y = jnp.linspace(y[0], y[1], num=num_samples)
    X, Y = jnp.meshgrid(x,y)
    return jnp.array([X.flatten(),Y.flatten()])


def base_chance_threshold_fixed_prop(base_chance_range, proportion, num_samples=10):

    def base_chance_threshold_integral(base_chance, proportion):
        # Returns the threshold required to achieve a certain keep proportion for any base_chance value.
        return (1-proportion)/(1-base_chance)

    base_chance_range_relevant = [base_chance_range[0], min(
        proportion, base_chance_range[1])]

    base_chance_samples = jnp.linspace(
        base_chance_range_relevant[0], base_chance_range_relevant[1], num=num_samples)

    threshold_samples = base_chance_threshold_integral(base_chance_samples, proportion)

    return jnp.array([threshold_samples, base_chance_samples])

### CURVE FITTING FUNCTION ---------------------------------------------------------------------------------------------

def model_function(x,*params,mut=0.1):

    """
    This is the what we are fitting to (sum of exponentials).
    It assumes a decay rate of 0.1 mutations per step.

    x = steps
    params = the output of the fitting function (get_single_decay_rate).
    """
    
    num_params = 1
    constant = params[-1]
    params = params[:-1]
    mut_curves = np.exp(-1.0*mut*x[:,None]*np.array(params)[None,:])
    weights = np.linspace(0.1, 0.9, num_params)
    weights = np.ones(num_params)
    weights = weights / weights.sum()
    sum_curves = np.sum(mut_curves * weights[None,:], axis = 1)
    return sum_curves * (1 - constant) + constant

## Measure decay rate function.

def get_single_decay_rate(decay_data, mut = 0.1, num_steps = 25):

    num_params = 1
    decay_data = decay_data/decay_data[0]

    if isinstance(mut, (int, float, complex)) or jnp.ndim(mut) == 0:
        steps = np.linspace(0,num_steps-1,num_steps)
    else:
        steps = mut

    if mut is None:
        mut = np.arange(len(decay_data))  # Default steps

    asymptote_guess = decay_data[-3:].mean() / decay_data[0]
    lower_bound = min(0.0, asymptote_guess - 0.2)
    upper_bound = max(1.1, asymptote_guess + 0.2)

    init_guess = np.concatenate([np.linspace(0.1, 0.9, num_params), [asymptote_guess]])
    lbounds = [0.0]*num_params + [lower_bound]
    ubounds = [2.0]*num_params + [upper_bound]
    # print("orig")
    # print(init_guess)
    # print(lbounds)
    # print(ubounds)
    # print(steps)
    # print(decay_data)

    model = lambda x, *params: model_function(x, *params, mut=mut)

    params, _ = curve_fit(model, steps, decay_data, p0=init_guess, maxfev=9000,
                          ftol=1e-4, xtol=1e-5, bounds=(lbounds, ubounds))
    mean_params = np.mean(params[:-1])
    fitted_constant = params[-1]  # The second returned parameter

    return mean_params, fitted_constant  # Return full params for plotting

def model_function_IK(x,*params,mut=0.1,y0=1):

    """
    This is the what we are fitting to (sum of exponentials).
    It assumes a decay rate of 0.1 mutations per step.

    x = steps
    params = the output of the fitting function (get_single_decay_rate).
    """
    
    num_params = 1
    constant = params[-1]
    params = params[:-1]
    mut_curves = np.exp(-1.0*mut*x[:,None]*np.array(params)[None,:])
    weights = np.linspace(0.1, 0.9, num_params)
    weights = np.ones(num_params)
    weights = weights / weights.sum()
    sum_curves = np.sum(mut_curves * weights[None,:], axis = 1)
    return sum_curves * (y0 - constant) + constant

def get_single_decay_rate_IK(decay_data, mut = 0.1, num_steps = 25):

    num_params = 1
    decay_data = decay_data # /decay_data[0]

    if isinstance(mut, (int, float, complex)) or jnp.ndim(mut) == 0:
        steps = np.linspace(0,num_steps-1,num_steps)
    else:
        steps = mut

    if mut is None:
        mut = np.arange(len(decay_data))  # Default steps

    # New: take last 20% for asymptote
    asymptote_guess = decay_data[-int(np.round(0.2*len(decay_data))):].mean() # / decay_data[0]
    lower_bound = 0.8*asymptote_guess # min(0.0, asymptote_guess - 0.2)
    upper_bound = 1.2*asymptote_guess # max(1.1, asymptote_guess + 0.2)

    # New: use gradient for rho
    nd = 2
    inds_a = np.arange(0,2*nd,1,dtype=int)
    mid_ab = int(np.round(num_steps/2))
    inds_b = np.arange(mid_ab-nd,mid_ab+nd+1,1,dtype=int)
    y_a = decay_data[inds_a].mean()
    y_b = decay_data[inds_b].mean()
    if (y_a > asymptote_guess) and (y_b > asymptote_guess) and (y_a > y_b):
        init_guess_rho = -np.log((y_b-asymptote_guess)/(y_a-asymptote_guess))/mid_ab/mut
        if (init_guess_rho < 0.1) or (init_guess_rho > 2.5):
            init_guess_rho = 0.6
        init_guess = np.concatenate([[init_guess_rho], [asymptote_guess]])
    else:
        init_guess = np.concatenate([np.linspace(0.1, 0.9, num_params), [asymptote_guess]])
    lbounds = [0.1]*num_params + [lower_bound]
    ubounds = [3]*num_params + [upper_bound]

    decay_start = decay_data[0]
    model = lambda x, *params: model_function_IK(x, *params, mut=mut, y0=decay_start)

    params, _ = curve_fit(model, steps, decay_data, p0=init_guess, maxfev=9000,
                          ftol=1e-4, xtol=1e-5, bounds=(lbounds, ubounds))
    mean_params = np.mean(params[:-1])
    fitted_constant = params[-1]  # The second returned parameter

    return mean_params, fitted_constant  # Return full params for plotting

def model_function_IK_v2(x,*params,mut=0.1,fix_amplitude=False,F0=1):

    """
    Models a function y(x)+C*exp(-x*mut*rho)+c with 
    rho = params[0]
    C = params[1]
    c = params[2]
    if fix_amplitude=False, and
    rho = params[0]
    c = params[1]
    and C=F0-c if fix_amplitude=True
    """
    rho = params[0]
    c = params[-1]
    C = params[1] if fix_amplitude==False else F0-c
    return C*np.exp(-1.0*mut*x*rho)+c

def get_single_decay_rate_IK_v2(decay_data, mut = 0.1, num_steps = 25, fix_amplitude=False):
    if fix_amplitude:
        F0 = decay_data[0]
        # decay_data = decay_data[1:]
        # steps = np.arange(1,num_steps)
    else:
        F0 = None
    steps = np.arange(0,num_steps)
    
    relb = 0.9
    # decay_data = decay_data/decay_data[0]
    
    # Asymptote
    asymptote_guess = decay_data[-int(np.round(0.2*len(decay_data))):].mean() # / decay_data[0]
    lb_asymptote = (1-np.sign(asymptote_guess)*relb)*asymptote_guess
    ub_asymptote = (1+np.sign(asymptote_guess)*relb)*asymptote_guess
    
    # Amplitude
    if not fix_amplitude:
        amplitude_guess = decay_data[0]-asymptote_guess
        lb_amplitude = (1-np.sign(amplitude_guess)*relb)*amplitude_guess
        ub_amplitude = (1+np.sign(amplitude_guess)*relb)*amplitude_guess+0.1
    
    # Rate
    rho_guess = 0.5
    lb_rho = 0.1
    ub_rho = 5
    mid_ab = int(np.round(num_steps/2))
    y_a = decay_data[0]
    y_b = decay_data[mid_ab]
    if (y_a > asymptote_guess) and (y_b > asymptote_guess) and (y_a > y_b):
        rho_guess = -np.log((y_b-asymptote_guess)/(y_a-asymptote_guess))/mid_ab/mut
        if (rho_guess < lb_rho) or (rho_guess > ub_rho):
            rho_guess = 0.5*(lb_rho+ub_rho)
    
    if fix_amplitude:
        init_guess = np.array([rho_guess, asymptote_guess])
        lbounds = [lb_rho, lb_asymptote]
        ubounds = [ub_rho, ub_asymptote]
        # init_guess = np.array([0.1, decay_data[-3:].mean()/decay_data[0]])
        # lbounds = [0, 0]
        # ubounds = [2, 1.1]
    else:
        init_guess = np.array([rho_guess, amplitude_guess, asymptote_guess])
        lbounds = [lb_rho, lb_amplitude, lb_asymptote]
        ubounds = [ub_rho, ub_amplitude, ub_asymptote]

    model = lambda x, *params: model_function_IK_v2(x, *params, mut=mut, fix_amplitude=fix_amplitude, F0=F0)
    params, _ = curve_fit(model, steps, decay_data, p0=init_guess, maxfev=9000,
                          ftol=1e-4, xtol=1e-5, bounds=(lbounds, ubounds))

    rho = params[0]
    c = params[-1]
    C = params[1] if fix_amplitude==False else F0-c

    return rho, C, c


