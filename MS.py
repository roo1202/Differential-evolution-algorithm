import numpy as np
import random
import math
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
from rana_function import rana_function

class MultiStrategyDE:
    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        population_size: int,
        mutation_factor: float = 0.5,
        crossover_probability: float = 0.5,
        max_iterations: int = 1000,
        pni: int = 100,
        epsilon: float = 1e-6,
        strategy_probs: List[float] = None,
        seed: int = None
    ):
        """
        Inicializa el optimizador de Evolución Diferencial Multiestrategia.

        Args:
            func (Callable): Función objetivo a minimizar.
            bounds (List[Tuple[float, float]]): Límites inferiores y superiores para cada dimensión.
            population_size (int): Tamaño de la población (K).
            mutation_factor (float): Factor de escala (F).
            crossover_probability (float): Probabilidad de cruce (Cr).
            max_iterations (int): Número máximo de iteraciones.
            pni (int): Número de iteraciones para evaluar mejora (PNI).
            epsilon (float): Umbral de mejora mínima requerida (ε).
            strategy_probs (List[float]): Probabilidades para seleccionar cada estrategia.
            seed (int): Semilla para la generación de números aleatorios.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.func = func
        self.bounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.population_size = population_size
        self.F = mutation_factor
        self.CR = crossover_probability
        self.max_iterations = max_iterations
        self.pni = pni
        self.epsilon = epsilon
        self.strategy_probs = strategy_probs if strategy_probs is not None else [1/3, 1/3, 1/3]
        
        self.population = self.initialize_population()
        self.fitness = np.apply_along_axis(self.func, 1, self.population)
        self.best_idx = np.argmin(self.fitness)
        self.best = self.population[self.best_idx]
        self.fbest = self.fitness[self.best_idx]
        self.history = []

    def initialize_population(self) -> np.ndarray:
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        population = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(self.population_size, self.dimensions))
        return population

    def mutate(self, idx: int) -> np.ndarray:
        while True:
            a = int(random.random() * self.population_size)
            if a != idx :
                break
        while True:
            b = int(random.random() * self.population_size)
            if b != idx :
                break
        while True:
            c = int(random.random() * self.population_size)
            if c != idx :
                break
        x_a = self.population[a]
        x_b = self.population[b]
        x_c = self.population[c]
        mutant = x_a + self.F * (x_b - x_c)
        mutant = np.clip(mutant, self.bounds[:, 0], self.bounds[:, 1])
        return mutant

    def crossover_exponential(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        trial = np.copy(target)
        j_rand = random.randint(0, self.dimensions - 1)
        for j in range(self.dimensions):
            if random.random() <= self.CR or j == j_rand:
                trial[j] = mutant[j]
            else:
                break
        return trial

    def either_or(self, target: np.ndarray) -> np.ndarray:
        mutant = self.mutate(0)  # Mutante
        recombined = self.mutate(0)  # Vector recombinado usando la mutación
        trial = np.copy(target)
        
        for j in range(self.dimensions):
            if random.random() <= self.CR:
                trial[j] = mutant[j]
            else:
                trial[j] = recombined[j]
        return trial

    def select(self, target_idx: int, trial: np.ndarray):
        f_trial = self.func(trial)
        if f_trial <= self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = f_trial
            if f_trial < self.fbest:
                self.fbest = f_trial
                self.best = trial

    def optimize(self):
        iteration = 0
        prev_best = self.fbest
        stagnant_iterations = 0
        
        while iteration < self.max_iterations:
            for i in range(self.population_size):
                target = self.population[i]
                strategy = random.choices(
                    population=['strategy1', 'crossover_exponential', 'either_or'],
                    weights=self.strategy_probs,
                    k=1
                )[0]
                
                if strategy == 'strategy1':
                    mutant = self.mutate(i)
                    trial = self.crossover_exponential(target, mutant)  # Se usa crossover binomial en strategy1
                elif strategy == 'crossover_exponential':
                    mutant = self.mutate(i)
                    trial = self.crossover_exponential(target, mutant)
                elif strategy == 'either_or':
                    trial = self.either_or(target)
                
                self.select(i, trial)
            
            self.history.append(self.fbest)
            
            if iteration % self.pni == 0 and iteration != 0:
                improvement = abs(prev_best - self.fbest)
                if improvement < self.epsilon:
                    print(f"Criterio de parada alcanzado en la iteración {iteration}: mejora {improvement} menor que epsilon {self.epsilon}")
                    break
                prev_best = self.fbest
            iteration += 1
        
        return self.best, self.fbest

