import tkinter as tk
from tkinter import messagebox, filedialog
from typing import Optional, List
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from random import randint
import random
import json
from collections import defaultdict
from abc import ABC, abstractmethod
from matplotlib.colors import ListedColormap
import numpy as np

# ==============================================================================
# Randomizer - Controle de Aleatoriedade
# ==============================================================================
class Randomizer:
    SEED = 1111
    _rand = random.Random(SEED)
    USE_SHARED = True

    @staticmethod
    def get_random():
        if Randomizer.USE_SHARED:
            return Randomizer._rand
        return random.Random()

    @staticmethod
    def reset(seed=None):
        if Randomizer.USE_SHARED:
            if seed is not None:
                Randomizer.SEED = seed
            Randomizer._rand.seed(Randomizer.SEED)

# ==============================================================================
# Location - Representa coordenadas na grade
# ==============================================================================
class Location:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __eq__(self, other):
        return isinstance(other, Location) and self.row == other.row and self.col == other.col

    def __hash__(self):
        return hash((self.row, self.col))

    def __str__(self):
        return f"Location({self.row}, {self.col})"

    __repr__ = __str__

# ==============================================================================
# Animal (Abstract)
# ==============================================================================
class Animal(ABC):
    def __init__(self, location: Location, random_age: bool = False):
        self._location = location
        self._alive = True
        self._age = 0
        self._rand = Randomizer.get_random()
        if random_age:
            self._age = self._rand.randint(0, 1000)

    def get_location(self) -> Location:
        return self._location

    def set_location(self, new_location: Location):
        self._location = new_location

    def is_alive(self) -> bool:
        return self._alive

    def set_dead(self):
        self._alive = False
        #self._location = None

    @abstractmethod
    def run(self, current_field, next_field_state):
        pass

# ==============================================================================
# Humano - Representa um indivíduo humano na simulação
# ==============================================================================

class Humano(Animal):
    SAUDAVEL = "Saudável"
    INFECTADO = "Infectado"
    RECUPERADO = "Recuperado"
    IMUNE = "Imune"
    MORTO_POR_DOENCA = "Morto por Doença"
    MORTO_POR_IDADE = "Morto por Idade"

    MAX_AGE = 27900
    BREEDING_PROBABILITY_BASE = 0.0001
    MIN_BREEDING_AGE = 18 * 365
    MAX_BREEDING_AGE = 50 * 365

    INFECTION_PROBABILITY = 0.3
    RECOVERY_PROBABILITY_BASE = 0.1
    DEATH_PROBABILITY_BASE = 0.01
    IMMUNITY_PROBABILITY = 0.95
    
    INITIAL_IMMUNITY_PROBABILITY = 0.95
    IMMUNITY_DECAY_RATE = 0.005 # Taxa de decaimento por passo

    def __init__(self, random_age: bool, location: Location):
        super().__init__(location, random_age)
        self.health_state = Humano.SAUDAVEL
        self.days_infected = 0
        self.current_immunity_probability = 0.0 # Nova variável para rastrear a probabilidade atual
        if random_age:
            # Idade inicial baseada em uma distribuição normal
            # Média de 30 anos (10950 dias), desvio padrão de 15 anos (5475 dias)
            # Garante que a idade fique entre 0 e MAX_AGE
            age_in_days = int(np.random.normal(10950, 5475))
            self._age = max(0, min(age_in_days, self.MAX_AGE - 1))

    def set_infected(self):
        if self.health_state == Humano.SAUDAVEL:
            self.health_state = Humano.INFECTADO
            self.days_infected = 0

    def run(self, current_field, next_field_state):
        if not self.is_alive():
            return

        self._increment_age()
        if not self.is_alive():
            return

        if self.health_state == Humano.INFECTADO:
            self.days_infected += 1
            self._check_for_recovery_or_death()

        if self.is_alive():
            self._give_birth(next_field_state, current_field.get_free_adjacent_locations(self._location))
            self._move(current_field, next_field_state)

        # Lógica para imunidade decrescente
        if self.health_state == Humano.IMUNE:
            self.current_immunity_probability -= self.IMMUNITY_DECAY_RATE
            if self.current_immunity_probability <= 0 or self._rand.random() > self.current_immunity_probability:
                self.health_state = Humano.SAUDAVEL
                self.current_immunity_probability = 0.0

    def _check_for_recovery_or_death(self):
        # Verifica se o humano se recupera ou morre
        recovery_prob = self.RECOVERY_PROBABILITY_BASE
        death_prob = self.DEATH_PROBABILITY_BASE
        age_in_years = self._age / 365
        if age_in_years > 60:
            death_prob *= 3.0
            recovery_prob *= 0.5
        elif age_in_years < 20:
            death_prob *= 0.5
            recovery_prob *= 1.5

        if self._rand.random() < death_prob:
            self.set_dead()
            self.health_state = Humano.MORTO_POR_DOENCA
        elif self._rand.random() < recovery_prob:
            self.health_state = Humano.RECUPERADO
            # O indivíduo só se torna imune se uma verificação aleatória for bem-sucedida
            if self._rand.random() < self.INITIAL_IMMUNITY_PROBABILITY:
                self.health_state = Humano.IMUNE
                self.current_immunity_probability = self.INITIAL_IMMUNITY_PROBABILITY

    def _can_breed(self):
        return self._age >= self.MIN_BREEDING_AGE and self._age <= self.MAX_BREEDING_AGE and self.health_state == Humano.SAUDAVEL

    def _give_birth(self, next_field_state, free_locations):
        if self._can_breed() and free_locations:
            if self._rand.random() < self.BREEDING_PROBABILITY_BASE:
                num_births = self._rand.randint(1, 2)
                for _ in range(num_births):
                    if free_locations:
                        loc = free_locations.pop(0)
                        baby = Humano(False, loc)
                        next_field_state.place_animal(baby, loc)

    def _increment_age(self):
        self._age += 1
        age_in_years = self._age / 365
        death_prob = 0.0001
        
        # Aumenta a probabilidade de morte drasticamente após os 60 anos
        if age_in_years > 60:
            death_prob *= (age_in_years - 60) * 0.05
        
        if self._rand.random() < death_prob:
            self.set_dead()
            self.health_state = Humano.MORTO_POR_IDADE

    def _move(self, current_field, next_field_state):
        # Obtém uma lista de locais adjacentes que estão livres NO CAMPO ATUAL
        free = current_field.get_free_adjacent_locations(self._location)
        
        # Filtra a lista para garantir que os locais também estão livres
        # no campo de estado seguinte (next_field_state), para evitar colisões
        valid_moves = [loc for loc in free if next_field_state.get_object_at(loc) is None]

        new_loc = self._location
        if valid_moves:
            # Escolhe um local aleatório da lista de movimentos válidos
            new_loc = self._rand.choice(valid_moves)

        # Coloca o animal no novo local no campo de estado seguinte
        next_field_state.place_animal(self, new_loc)

# ==============================================================================
# Counter - Contador para rastrear populações
# ==============================================================================
class Counter:
    def __init__(self, name):
        self.name = name
        self.count = 0
    def get_name(self):
        return self.name
    def get_count(self):
        return self.count
    def increment(self):
        self.count += 1
    def reset(self):
        self.count = 0

# ==============================================================================
# FieldStats - Estatísticas do campo
# ==============================================================================
class FieldStats:
    def __init__(self):
        self.counters: dict[str, Counter] = {}
        self.counts_valid = False

    def get_population_details(self, field) -> str:
        if not self.counts_valid:
            self._generate_counts(field)
        return " ".join([f"{c.get_name()}: {c.get_count()}" for c in self.counters.values()])

    def reset(self):
        self.counts_valid = False
        self.counters.clear()

    def _generate_counts(self, field):
        self.reset()
        for animal in field.get_animals():
            if isinstance(animal, Humano):
                state = animal.health_state
                if state not in self.counters:
                    self.counters[state] = Counter(state)
                self.counters[state].increment()
        self.counts_valid = True

    def is_viable(self, field) -> bool:
        self._generate_counts(field)
        total_population = sum(c.get_count() for s, c in self.counters.items() if s not in [Humano.MORTO_POR_DOENCA, Humano.MORTO_POR_IDADE])
        infected_count = self.counters.get(Humano.INFECTADO, Counter("tmp")).get_count()
        return total_population > 0 and infected_count > 0

# ==============================================================================
# Field - Representa a grade da simulação
# ==============================================================================

class Field:
    def __init__(self, depth: int, width: int):
        self.depth = depth
        self.width = width
        self.field: dict[Location, Animal] = {}
        self._rand = Randomizer.get_random()

    def place_animal(self, animal: Animal, location: Location):
        assert location is not None
        self.field[location] = animal
        animal.set_location(location)

    def remove_animal(self, location: Location):
        if location in self.field:
            animal = self.field.pop(location)
            animal.set_dead()
            return animal
        return None

    def get_object_at(self, location: Location) -> Optional[Animal]:
        return self.field.get(location)

    def clear(self):
        self.field.clear()

    def get_animals(self) -> List[Animal]:
        return list(self.field.values())

    def get_adjacent_locations(self, location: Location) -> List[Location]:
        locations = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = location.row + dr, location.col + dc
                if 0 <= nr < self.depth and 0 <= nc < self.width:
                    locations.append(Location(nr, nc))
        self._rand.shuffle(locations)
        return locations

    def get_free_adjacent_locations(self, location: Location) -> List[Location]:
        return [loc for loc in self.get_adjacent_locations(location) if self.get_object_at(loc) is None]

    def get_occupied_adjacent_locations(self, location: Location) -> List[Location]:
        return [loc for loc in self.get_adjacent_locations(location) if self.get_object_at(loc) is not None]

    def get_depth(self) -> int:
        return self.depth

    def get_width(self) -> int:
        return self.width

# ==============================================================================
# SimulatorView - Exibição gráfica
# ==============================================================================

class SimulatorView:
    def __init__(self, depth, width):
        self.stats = FieldStats()
        self.depth = depth
        self.width = width
        self.state_to_idx = {
            Humano.SAUDAVEL: 0,
            Humano.INFECTADO: 1,
            Humano.RECUPERADO: 2,
            Humano.IMUNE: 3,
            Humano.MORTO_POR_DOENCA: 4,
            Humano.MORTO_POR_IDADE: 5,
            "Vazio": 6 
        }
        self.idx_to_color = ["green", "red", "blue", "cyan", "darkred", "black", "white"]
        self.cm = ListedColormap(self.idx_to_color)

    def show_status(self, step, field: Field, ax):
        self.stats._generate_counts(field)
        grid_data = [[self.state_to_idx["Vazio"]] * self.width for _ in range(self.depth)]
        for y in range(self.depth):
            for x in range(self.width):
                animal = field.get_object_at(Location(y, x))
                if isinstance(animal, Humano):
                    grid_data[y][x] = self.state_to_idx.get(animal.health_state, self.state_to_idx["Vazio"])
        ax.imshow(grid_data, cmap=self.cm, interpolation='nearest', vmin=0, vmax=len(self.idx_to_color)-1)
        
        # Cria a legenda manual
        handles = [plt.Rectangle((0, 0), 1, 1, fc=self.idx_to_color[self.state_to_idx[s]]) for s in self.state_to_idx if s != "Vazio"]
        labels = [s for s in self.state_to_idx if s != "Vazio"]
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        ax.set_title(f"Passo: {step}\nPopulação: {self.stats.get_population_details(field)}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

    def is_viable(self, field: Field):
        return self.stats.is_viable(field)

# ==============================================================================
# Simulator - Núcleo da lógica da simulação
# ==============================================================================

class Simulator:
    def __init__(self, field: Field, view: SimulatorView, seed: Optional[int] = None):
        self.field = field
        self.view = view
        self.step = 0
        self.initial_seed = seed
        self._rand = Randomizer.get_random()
        self.initial_field_state = None
        self.reset()

    def _get_field_snapshot(self):
        snapshot = []
        for animal in self.field.get_animals():
            if isinstance(animal, Humano) and animal.is_alive():
                loc = animal.get_location()
                snapshot.append({
                    'type': 'humano',
                    'row': loc.row,
                    'col': loc.col,
                    'health': animal.health_state,
                    'age': animal._age,
                    'days_infected': animal.days_infected
                })
        return snapshot

    def _populate(self, population_size, initial_infected_count):
        self.field.clear()
        all_locations = [Location(r, c) for r in range(self.field.get_depth()) for c in range(self.field.get_width())]
        self._rand.shuffle(all_locations)
        num_to_populate = min(population_size, len(all_locations))
        for i in range(num_to_populate):
            loc = all_locations.pop(0)
            new_h = Humano(True, loc)
            self.field.place_animal(new_h, loc)
        infected_indices = self._rand.sample(range(num_to_populate), k=min(initial_infected_count, num_to_populate))
        animals = self.field.get_animals()
        for i in infected_indices:
            if isinstance(animals[i], Humano):
                animals[i].set_infected()

    def reset(self):
        Randomizer.reset(self.initial_seed)
        self.step = 0
        initial_pop_size = (self.field.get_depth() * self.field.get_width()) // 2
        initial_infected = 1
        self._populate(initial_pop_size, initial_infected)
        self.view.stats.reset()
        self.initial_field_state = self._get_field_snapshot()

    def simulate_one_step(self):
        # Primeiro remove os mortos "antigos" (que já ficaram visíveis no passo anterior)
        self.field.field = {
            loc: a for loc, a in self.field.field.items()
            if a.is_alive() or a.health_state in [Humano.MORTO_POR_DOENCA, Humano.MORTO_POR_IDADE]
        }

        self.step += 1
        next_field_state = Field(self.field.get_depth(), self.field.get_width())
        animals_to_process = list(self.field.get_animals())
        self._rand.shuffle(animals_to_process)
        processed_locations = set()

        for animal in animals_to_process:
            loc = animal.get_location()
            if loc not in processed_locations and animal.is_alive():
                # verificação de contágio
                if animal.health_state == Humano.SAUDAVEL:
                    for neighbor_loc in self.field.get_occupied_adjacent_locations(loc):
                        neighbor = self.field.get_object_at(neighbor_loc)
                        if isinstance(neighbor, Humano) and neighbor.health_state == Humano.INFECTADO:
                            if self._rand.random() < animal.INFECTION_PROBABILITY:
                                animal.set_infected()
                                break
                animal.run(self.field, next_field_state)
                if animal.is_alive():
                    processed_locations.add(animal.get_location())
            elif loc not in processed_locations and not animal.is_alive():
                # mortos também são transferidos para o próximo campo (para aparecerem nesse passo)
                next_field_state.place_animal(animal, loc)

        # Reconstrói o campo com vivos e mortos recentes
        self.field.clear()
        for a in next_field_state.get_animals():
            if a.get_location() is not None:
                self.field.place_animal(a, a.get_location())
    
        

    def save_initial_field(self, file_path: str) -> bool:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'depth': self.field.get_depth(),
                    'width': self.field.get_width(),
                    'step': self.step,
                    'entities': self._get_field_snapshot()
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def load_field_from_file(self, file_path: str) -> bool:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            depth = payload.get('depth')
            width = payload.get('width')
            self.field = Field(depth, width)
            for ent in payload.get('entities', []):
                if ent.get('type') == 'humano':
                    loc = Location(ent['row'], ent['col'])
                    h = Humano(False, loc)
                    h._age = ent.get('age', 0)
                    h.days_infected = ent.get('days_infected', 0)
                    h.health_state = ent.get('health', Humano.SAUDAVEL)
                    if h.health_state == Humano.MORTO:
                        h.set_dead()
                    self.field.place_animal(h, loc)
            self.view.stats.reset()
            self.step = payload.get('step', 0)
            return True
        except Exception:
            return False

# ==============================================================================
# GUI - Interface Gráfica
# ==============================================================================

class SimulationGUI(tk.Frame):
    """
    Interface gráfica para controlar e visualizar a simulação.
    """
    def __init__(self, master, seed=None, return_callback=None, humano_params=None):
        super().__init__(master)
        if humano_params:
            Humano.INFECTION_PROBABILITY = humano_params['INFECTION_PROBABILITY']
            Humano.RECOVERY_PROBABILITY_BASE = humano_params['RECOVERY_PROBABILITY_BASE']
            Humano.DEATH_PROBABILITY_BASE = humano_params['DEATH_PROBABILITY_BASE']
            Humano.INITIAL_IMMUNITY_PROBABILITY = humano_params['INITIAL_IMMUNITY_PROBABILITY']
            Humano.IMMUNITY_DECAY_RATE = humano_params['IMMUNITY_DECAY_RATE']
        
        
        self.master = master
        self.return_callback = return_callback
        self.is_running_many = False
        self.is_paused = False
        self.population_history = defaultdict(list)

        depth, width = 60, 60
        self.field_instance = Field(depth, width)
        self.view_instance = SimulatorView(depth, width)
        self.simulator = Simulator(self.field_instance, self.view_instance, seed)

        self._update_population_history(initial=True)

        # Menus
        self.menubar = tk.Menu(master)
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Salvar Campo Inicial", command=self.save_initial_field)
        filemenu.add_command(label="Carregar Campo Salvo", command=self.load_initial_field)
        filemenu.add_separator()
        filemenu.add_command(label="Mostrar Gráfico de População", command=self.show_population_plot)
        filemenu.add_separator()
        filemenu.add_command(label="Sair", command=master.quit)
        self.menubar.add_cascade(label="Arquivo", menu=filemenu)
        master.config(menu=self.menubar)

        # Layout
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.label_step = tk.Label(control_frame, text=f"Dias: {self.simulator.step}", font=("Arial", 12))
        self.label_step.pack(side=tk.LEFT, padx=10)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)
        if self.return_callback:
            tk.Button(button_frame, text="Voltar", command=self.return_and_destroy).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Próximo", command=self.next_step).pack(side=tk.LEFT, padx=5)
        self.btn_pause = tk.Button(button_frame, text="Pausar", command=self.pause_resume_simulation, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Resetar", command=self.reset_simulation).pack(side=tk.LEFT, padx=5)

        custom_frame = tk.Frame(main_frame, pady=5)
        custom_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(custom_frame, text="Dias:").pack(side=tk.LEFT)
        self.steps_entry = tk.Entry(custom_frame, width=10)
        self.steps_entry.insert(0, "100")
        self.steps_entry.pack(side=tk.LEFT, padx=5)
        self.btn_simulate = tk.Button(custom_frame, text="Iniciar", command=self.start_custom_simulation)
        self.btn_simulate.pack(side=tk.LEFT, padx=5)
        
        # Novo Frame para os atalhos de tempo
        shortcuts_frame = tk.LabelFrame(main_frame, text="Atalhos de Tempo", padx=5, pady=5)
        shortcuts_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        tk.Button(shortcuts_frame, text="1 Mês", command=lambda: self.run_shortcut_simulation(30)).pack(side=tk.LEFT, padx=5)
        tk.Button(shortcuts_frame, text="3 Meses", command=lambda: self.run_shortcut_simulation(90)).pack(side=tk.LEFT, padx=5)
        tk.Button(shortcuts_frame, text="6 Meses", command=lambda: self.run_shortcut_simulation(180)).pack(side=tk.LEFT, padx=5)
        tk.Button(shortcuts_frame, text="1 Ano", command=lambda: self.run_shortcut_simulation(365)).pack(side=tk.LEFT, padx=5)
        tk.Button(shortcuts_frame, text="2 Anos", command=lambda: self.run_shortcut_simulation(730)).pack(side=tk.LEFT, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.draw_grid()

    def return_and_destroy(self):
        """Volta para a tela inicial e destrói o frame da simulação."""
        if self.is_running_many:
            self.is_running_many = False
        self.destroy()
        self.return_callback()

    def _update_population_history(self, initial=False):
        counts = {s: 0 for s in [Humano.SAUDAVEL, Humano.INFECTADO, Humano.RECUPERADO, Humano.IMUNE, Humano.MORTO_POR_DOENCA, Humano.MORTO_POR_IDADE]} 
        for a in self.simulator.field.get_animals():
            if isinstance(a, Humano):
                counts[a.health_state] += 1
        for s, c in counts.items():
            self.population_history[s].append(c)
        self.population_history['steps'].append(self.simulator.step if not initial else 0)

    def show_population_plot(self):
        # Acessa 'steps' de forma segura. Se 'steps' não existir, retorna uma lista vazia,
        # o que faz a verificação 'if not steps:' funcionar corretamente.
        steps = self.population_history.get('steps', []) 
        if not steps:
            messagebox.showinfo("Gráfico", "Sem dados para exibir.")
            return

        total_population = []
        # O cálculo da população total agora inclui todos os estados de saúde, incluindo os mortos.
        all_states = [
            Humano.SAUDAVEL, 
            Humano.INFECTADO, 
            Humano.RECUPERADO, 
            Humano.IMUNE, 
            Humano.MORTO_POR_DOENCA, 
            Humano.MORTO_POR_IDADE
        ]
        
        for i, step in enumerate(steps):
            current_total = 0
            for state in all_states:
                if state in self.population_history:
                    # Garantimos que o índice 'i' existe na lista de histórico para evitar erros.
                    if i < len(self.population_history[state]):
                        current_total += self.population_history[state][i]
            total_population.append(current_total)
        
        # Exibe o gráfico da evolução da população
        plt.figure(figsize=(10, 6))
        
        # O 'state_map' já inclui todos os estados de saúde e morte.
        state_map = {
            Humano.SAUDAVEL: "green",
            Humano.INFECTADO: "red",
            Humano.RECUPERADO: "blue",
            Humano.IMUNE: "cyan",
            Humano.MORTO_POR_DOENCA: "darkred", 
            Humano.MORTO_POR_IDADE: "black"
        }
        
        # Plotamos as linhas para cada estado
        for state, color in state_map.items():
            if state in self.population_history:
                plt.plot(steps, self.population_history[state], label=state, color=color)

        # Adiciona a linha de população total ao gráfico
        plt.plot(steps, total_population, label="População Total", color="grey", linestyle='--')

        plt.xlabel("Passo")
        plt.ylabel("Número de Indivíduos")
        plt.title("Evolução da População e Contagem de Mortos")
        plt.legend()
        plt.grid(True)
        plt.show()

    def draw_grid(self):
        self.ax.clear()
        self.view_instance.show_status(self.simulator.step, self.field_instance, self.ax)
        self.canvas.draw()
        self.label_step.config(text=f"Passo: {self.simulator.step} | {self.view_instance.stats.get_population_details(self.field_instance)}")

    def next_step(self):
        self.simulator.simulate_one_step()
        self.draw_grid()
        self._update_population_history()
    
    # Novo método para atalhos de tempo
    def run_shortcut_simulation(self, steps):
        """Inicia a simulação por um número de passos pré-definido."""
        self.start_custom_simulation(steps)

    def start_custom_simulation(self, steps=None):
        if self.is_running_many:
            return
        if steps is None:
            try:
                steps = int(self.steps_entry.get())
                if steps <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Erro", "Digite um número válido.")
                return
        self.steps_to_run = steps
        self.is_running_many = True
        self.is_paused = False
        self.btn_pause.config(text="Pausar", state=tk.NORMAL)
        self.master.after(1, self._run_many_steps)

    def pause_resume_simulation(self):
        self.is_paused = not self.is_paused
        self.btn_pause.config(text="Continuar" if self.is_paused else "Pausar")
        if not self.is_paused:
            self.master.after(1, self._run_many_steps)

    def _run_many_steps(self):
        if not self.is_running_many or self.is_paused:
            return
        if self.steps_to_run > 0 and self.view_instance.is_viable(self.field_instance):
            self.next_step()
            self.steps_to_run -= 1
            self.master.after(1, self._run_many_steps)
        else:
            self.is_running_many = False
            self.btn_pause.config(state=tk.DISABLED)
            self.show_population_plot()

    def reset_simulation(self):
        self.is_running_many = False
        self.is_paused = False
        self.btn_pause.config(text="Pausar", state=tk.DISABLED)
        self.simulator.reset()
        self.population_history.clear()
        self.population_history['steps'] = []
        self._update_population_history(initial=True)
        self.draw_grid()

    def save_initial_field(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if file_path:
            if self.simulator.save_initial_field(file_path):
                messagebox.showinfo("Sucesso", "Campo salvo com sucesso.")
            else:
                messagebox.showerror("Erro", "Não foi possível salvar o campo.")

    def load_initial_field(self):
        file_path = filedialog.askopenfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if file_path:
            if self.simulator.load_field_from_file(file_path):
                self.population_history.clear()
                self.population_history['steps'] = []
                self._update_population_history(initial=True)
                self.draw_grid()
                messagebox.showinfo("Sucesso", "Campo carregado com sucesso.")
            else:
                messagebox.showerror("Erro", "Não foi possível carregar o campo.")

# ==============================================================================
# AppController - Tela inicial e controle principal
# ==============================================================================

class AppController:
    def __init__(self, master):
        self.master = master
        self.master.title("Simulador de Infecção")
        self.current_frame = None
        self.initial_seed = Randomizer.get_random().randint(0, 10000)
        self.setup_start_screen()

    def setup_start_screen(self):
        if self.current_frame:
            self.current_frame.destroy()

        self.start_frame = tk.Frame(self.master, padx=20, pady=20)
        self.current_frame = self.start_frame
        self.start_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.start_frame, text="Simulador de Infecção", font=("Arial", 20, "bold")).pack(pady=10)

        seed_frame = tk.LabelFrame(self.start_frame, text="Semente Aleatória", padx=10, pady=10)
        seed_frame.pack(pady=5)
        tk.Label(seed_frame, text="Semente:").pack(side=tk.LEFT, padx=5)
        self.seed_entry = tk.Entry(seed_frame)
        self.seed_entry.insert(0, str(self.initial_seed))
        self.seed_entry.pack(side=tk.LEFT, padx=5)

        humano_frame = tk.LabelFrame(self.start_frame, text="Parâmetros do Humano", padx=10, pady=10)
        humano_frame.pack(pady=10)

        tk.Label(humano_frame, text="Prob. Infecção:").grid(row=0, column=0, sticky="w")
        self.entry_infection = tk.Entry(humano_frame)
        self.entry_infection.insert(0, str(Humano.INFECTION_PROBABILITY))
        self.entry_infection.grid(row=0, column=1)

        tk.Label(humano_frame, text="Prob. Recuperação:").grid(row=1, column=0, sticky="w")
        self.entry_recovery = tk.Entry(humano_frame)
        self.entry_recovery.insert(0, str(Humano.RECOVERY_PROBABILITY_BASE))
        self.entry_recovery.grid(row=1, column=1)

        tk.Label(humano_frame, text="Prob. Morte:").grid(row=2, column=0, sticky="w")
        self.entry_death = tk.Entry(humano_frame)
        self.entry_death.insert(0, str(Humano.DEATH_PROBABILITY_BASE))
        self.entry_death.grid(row=2, column=1)

        tk.Label(humano_frame, text="Prob. Imunidade:").grid(row=3, column=0, sticky="w")
        self.entry_immunity = tk.Entry(humano_frame)
        self.entry_immunity.insert(0, str(Humano.IMMUNITY_PROBABILITY))
        self.entry_immunity.grid(row=3, column=1)

        tk.Label(humano_frame, text="Prob. Imunidade Inicial:").grid(row=3, column=0, sticky="w")
        self.entry_immunity_initial = tk.Entry(humano_frame)
        self.entry_immunity_initial.insert(0, str(Humano.INITIAL_IMMUNITY_PROBABILITY))
        self.entry_immunity_initial.grid(row=3, column=1)

        tk.Label(humano_frame, text="Taxa de Decaimento de Imunidade:").grid(row=4, column=0, sticky="w")
        self.entry_immunity_decay = tk.Entry(humano_frame)
        self.entry_immunity_decay.insert(0, str(Humano.IMMUNITY_DECAY_RATE))
        self.entry_immunity_decay.grid(row=4, column=1)

        start_button = tk.Button(self.start_frame, text="Iniciar Simulação", command=self.start_simulation)
        start_button.pack(pady=10)

    def start_simulation(self):
        try:
            seed = int(self.seed_entry.get())
        except ValueError:
            seed = None
        params = {
            'INFECTION_PROBABILITY': float(self.entry_infection.get()),
            'RECOVERY_PROBABILITY_BASE': float(self.entry_recovery.get()),
            'DEATH_PROBABILITY_BASE': float(self.entry_death.get()),
            'INITIAL_IMMUNITY_PROBABILITY': float(self.entry_immunity_initial.get()),
            'IMMUNITY_DECAY_RATE': float(self.entry_immunity_decay.get())
        }
        self.show_simulation(seed, params)

    def show_simulation(self, seed, humano_params):
        if self.current_frame:
            self.current_frame.destroy()
        self.current_frame = SimulationGUI(self.master, seed, self.setup_start_screen, humano_params)
        self.current_frame.pack(fill=tk.BOTH, expand=True)

# ==============================================================================
# Execução principal
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AppController(root)
    root.mainloop()