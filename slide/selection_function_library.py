from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

def base_chance_threshold_select(fitnesses: jax.Array, params: Mapping[str, float]) -> jax.Array:
    """Return rank-threshold selection probabilities with a baseline chance.

    Parameters:
    - fitnesses: jax.Array
        Fitness values for one population.
    - params: Mapping[str, float]
        Selection parameters containing ``base_chance`` and ``threshold``.

    Returns:
    - jax.Array
        Selection probabilities clipped to the interval ``[0, 1]``.
    """
    normed_fitness = jnp.argsort(jnp.argsort(fitnesses)) / (fitnesses.shape[0] - 1)
    return jnp.clip(params['base_chance'] + (normed_fitness >= params['threshold']), 0, 1)

def sigmoid_select(fitnesses: jax.Array, params: Mapping[str, float]) -> jax.Array:
    """Return sigmoid rank-based selection probabilities.

    Parameters:
    - fitnesses: jax.Array
        Fitness values for one population.
    - params: Mapping[str, float]
        Selection parameters containing ``base_chance``, ``threshold``, and
        ``steepness``.

    Returns:
    - jax.Array
        Selection probabilities for each population member.
    """
    normed_fitness = jnp.argsort(jnp.argsort(fitnesses)) / (fitnesses.shape[0] - 1)
    return (1 - params['base_chance']) / (1 + jnp.exp(params['steepness'] * (params['threshold'] - normed_fitness))) + params['base_chance']
