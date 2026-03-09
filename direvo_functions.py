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


