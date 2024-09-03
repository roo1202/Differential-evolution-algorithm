import numpy as np
import random
import math
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
from rana_function import rana_function

class DifferentialEvolution:
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
        seed: int = None
    ):
        """
        Inicializa el optimizador de Evolución Diferencial.

        Args:
            func (Callable): Función objetivo a minimizar.
            bounds (List[Tuple[float, float]]): Límites inferiores y superiores para cada dimensión.
            population_size (int): Tamaño de la población (K).
            mutation_factor (float): Factor de escala (F).
            crossover_probability (float): Probabilidad de cruce (Cr).
            max_iterations (int): Número máximo de iteraciones.
            pni (int): Número de iteraciones para evaluar mejora (PNI).
            epsilon (float): Umbral de mejora mínima requerida (ε).
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
        
        # Inicializar la población
        self.population = self.initialize_population()
        self.fitness = [self.func(individue) for individue in self.population]
        self.best_idx = np.argmin(self.fitness)
        self.best = self.population[self.best_idx]
        self.fbest = self.fitness[self.best_idx]
        self.history = []

    def initialize_population(self) -> np.ndarray:
        """
        Genera la población inicial dentro de los límites especificados.

        Returns:
            np.ndarray: Población inicial de tamaño (population_size, dimensions).
        """
        lower_bounds = self.bounds[:, 0]
        upper_bounds = self.bounds[:, 1]
        population = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(self.population_size, self.dimensions))
        return population

    def mutate(self, idx: int) -> np.ndarray:
        """
        Realiza la mutación para un individuo específico.

        Args:
            idx (int): Índice del individuo objetivo.

        Returns:
            np.ndarray: Vector mutado.
        """
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

    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Realiza el cruce entre el vector objetivo y el vector mutado.

        Args:
            target (np.ndarray): Vector objetivo.
            mutant (np.ndarray): Vector mutado.

        Returns:
            np.ndarray: Vector trial resultante del cruce.
        """
        trial = np.copy(target)
        j_rand = random.randint(0, self.dimensions - 1)
        for j in range(self.dimensions):
            if random.random() <= self.CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def select(self, target_idx: int, trial: np.ndarray):
        """
        Realiza la selección entre el vector objetivo y el vector trial.

        Args:
            target_idx (int): Índice del vector objetivo.
            trial (np.ndarray): Vector trial.

        Updates:
            Actualiza la población y el fitness si el trial es mejor.
        """
        f_trial = self.func(trial)
        if f_trial <= self.fitness[target_idx]:
            self.population[target_idx] = trial
            self.fitness[target_idx] = f_trial
            if f_trial < self.fbest:
                self.fbest = f_trial
                self.best = trial

    def optimize(self):
        """
        Ejecuta el proceso de optimización.

        Returns:
            Tuple[np.ndarray, float]: Mejor solución encontrada y su valor de función objetivo.
        """
        iteration = 0
        prev_best = self.fbest
        stagnant_iterations = 0
        
        while iteration < self.max_iterations:
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                self.select(i, trial)
    
            # Guardar el historial de la mejor fitness
            self.history.append(self.fbest)
            
            # Verificar criterio de parada basado en PNI y epsilon
            if iteration % self.pni == 0 and iteration != 0:
                improvement = np.abs(prev_best - self.fbest)
                if improvement < self.epsilon:
                    print(f"Criterio de parada alcanzado en la iteración {iteration}: mejora {improvement} menor que epsilon {self.epsilon}")
                    break
                prev_best = self.fbest
            iteration += 1
        
        return self.best, self.fbest

