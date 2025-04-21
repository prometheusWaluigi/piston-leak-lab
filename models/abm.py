#!/usr/bin/env python3

"""
Agent-Based Model for Piston Leak Lab
=====================================

Implements an Ising-like network of belief agents that interact
through influence dynamics and respond to the macro TNP state.

See paper §4.2 for mathematical formulation.
"""

import numpy as np
import networkx as nx
from enum import Enum, auto
from typing import Any, List, Tuple, Callable, Optional, Union
from dataclasses import dataclass, field

# Import the ODE model to enable coupling
try:
    from .core_ode import PistonLeakODE, PistonLeakParams
except ImportError:
    from core_ode import PistonLeakODE, PistonLeakParams


class BeliefState(Enum):
    """Possible belief states of agents in the network."""
    BELIEVER = auto()   # Believes institutional narrative
    SKEPTIC = auto()    # Rejects institutional narrative
    AGNOSTIC = auto()   # Undecided/uncertain


@dataclass
class ABMParams:
    """Parameters for the Agent-Based Model."""
    # Network parameters
    network_type: str = "watts_strogatz"  # Type of network topology
    n_agents: int = 3000                  # Number of agents in the model
    k: int = 6                            # Mean degree for network generation
    p_rewire: float = 0.2                 # Rewiring probability (small-world)
    
    # Influence weights by belief state
    believer_weight: float = 1.0    # Influence weight of believers
    skeptic_weight: float = 1.2     # Influence weight of skeptics
    agnostic_weight: float = 0.8    # Influence weight of agnostics
    
    # Transition dynamics
    base_transition_rate: float = 0.05  # Base rate of state changes
    temperature: float = 1.0            # Temperature for Glauber dynamics
    
    # Initial distribution (probabilities sum to 1)
    init_believer_prob: float = 0.6  # Initial fraction of believers
    init_skeptic_prob: float = 0.2   # Initial fraction of skeptics
    init_agnostic_prob: float = 0.2  # Initial fraction of agnostics
    
    # Coupling to macro TNP state
    coupling_strength: float = 0.3  # How strongly agents respond to macro state
    
    def __post_init__(self):
        """Validate parameter constraints."""
        assert self.n_agents > 0, "Number of agents must be positive"
        assert self.k < self.n_agents, "Mean degree must be less than n_agents"
        assert 0 <= self.p_rewire <= 1, "Rewiring probability must be in [0,1]"
        assert (
            self.init_believer_prob + 
            self.init_skeptic_prob + 
            self.init_agnostic_prob
        ) == 1.0, "Initial state probabilities must sum to 1"


class PistonLeakABM:
    """
    Agent-Based Model implementation for Piston Leak.
    
    Implements an Ising-like network of belief agents with
    three possible states and coupling to the macro TNP dynamics.
    """
    
    def __init__(self, 
                 params: ABMParams = None,
                 ode_model: PistonLeakODE = None,
                 seed: int = 42):
        """
        Initialize the Agent-Based Model.
        
        Args:
            params: ABM parameters
            ode_model: The ODE model to couple with
            seed: Random seed for reproducibility
        """
        self.params = params or ABMParams()
        self.ode_model = ode_model or PistonLeakODE()
        self.rng = np.random.RandomState(seed)
        
        # Generate the social network
        self.network = self._generate_network()
        
        # Initialize agent states
        self.states = self._initialize_states()
        
        # Storage for simulation results
        self.history = []
        self.state_counts = {
            BeliefState.BELIEVER: [],
            BeliefState.SKEPTIC: [],
            BeliefState.AGNOSTIC: []
        }
    
    def _generate_network(self) -> nx.Graph:
        """
        Generate the social network topology.
        
        Returns:
            NetworkX graph representing agent connections
        """
        n, k, p = self.params.n_agents, self.params.k, self.params.p_rewire
        
        if self.params.network_type == "watts_strogatz":
            # Small-world network (clustered but with shortcuts)
            G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=self.rng.randint(1000))
        elif self.params.network_type == "barabasi_albert":
            # Scale-free network (few hubs, many peripheral nodes)
            m = min(k, n-1)  # Can't attach to more nodes than exist
            G = nx.barabasi_albert_graph(n=n, m=m, seed=self.rng.randint(1000))
        elif self.params.network_type == "erdos_renyi":
            # Random network (Poisson degree distribution)
            p_edge = k / (n - 1)  # Convert mean degree to probability
            G = nx.erdos_renyi_graph(n=n, p=p_edge, seed=self.rng.randint(1000))
        else:
            raise ValueError(f"Unknown network type: {self.params.network_type}")
        
        return G
    
    def _initialize_states(self) -> dict[int, BeliefState]:
        """
        Initialize agent belief states according to parameters.
        
        Returns:
            Dictionary mapping node ID to belief state
        """
        states = {}
        probs = [
            self.params.init_believer_prob,
            self.params.init_skeptic_prob,
            self.params.init_agnostic_prob
        ]
        state_options = [
            BeliefState.BELIEVER,
            BeliefState.SKEPTIC,
            BeliefState.AGNOSTIC
        ]
        
        for node in self.network.nodes():
            # Random choice based on initial probabilities
            states[node] = self.rng.choice(state_options, p=probs)
        
        return states
    
    def _get_influence_weight(self, state: BeliefState) -> float:
        """
        Get the influence weight for a given belief state.
        
        Args:
            state: Belief state of the agent
            
        Returns:
            Influence weight
        """
        if state == BeliefState.BELIEVER:
            return self.params.believer_weight
        elif state == BeliefState.SKEPTIC:
            return self.params.skeptic_weight
        else:  # AGNOSTIC
            return self.params.agnostic_weight
    
    def _calculate_energy_delta(self, node: int, new_state: BeliefState, 
                               tnp_state: tuple[float, float, float]) -> float:
        """
        Calculate the energy change for a state transition.
        
        Follows Ising-like dynamics with neighbor influence and
        coupling to the macro TNP state.
        
        Args:
            node: Node ID
            new_state: Proposed new belief state
            tnp_state: Current (T, N, P) macro state
            
        Returns:
            Energy delta for the proposed transition
        """
        old_state = self.states[node]
        if old_state == new_state:
            return 0.0
        
        # Neighbor influence component
        neighbor_influence = 0.0
        for neighbor in self.network.neighbors(node):
            neighbor_state = self.states[neighbor]
            neighbor_weight = self._get_influence_weight(neighbor_state)
            
            # Same-state neighbors reduce energy (favorable)
            if neighbor_state == new_state:
                neighbor_influence -= neighbor_weight
            # Different-state neighbors increase energy (unfavorable)
            else:
                neighbor_influence += neighbor_weight
        
        # Coupling to macro TNP state
        T, N, P = tnp_state
        tnp_coupling = 0.0
        
        # Trust (T) makes BELIEVER state more favorable
        if new_state == BeliefState.BELIEVER:
            tnp_coupling -= T
        # Narrative Entropy (N) makes SKEPTIC state more favorable
        elif new_state == BeliefState.SKEPTIC:
            tnp_coupling -= N
        # Suppression Pressure (P) makes AGNOSTIC state more favorable
        else:  # AGNOSTIC
            tnp_coupling -= P
        
        # Combine influences with coupling strength
        energy_delta = neighbor_influence + self.params.coupling_strength * tnp_coupling
        return energy_delta
    
    def _transition_probability(self, energy_delta: float) -> float:
        """
        Calculate transition probability using Glauber dynamics.
        
        Args:
            energy_delta: Energy change for the proposed transition
            
        Returns:
            Transition probability
        """
        # Base rate * Glauber formula (T-dependent sigmoid)
        base_rate = self.params.base_transition_rate
        T = self.params.temperature
        return base_rate * (1.0 / (1.0 + np.exp(energy_delta / T)))
    
    def update(self, tnp_state: tuple[float, float, float]) -> Dict:
        """
        Update the agent states for one timestep.
        
        Args:
            tnp_state: Current (T, N, P) macro state
            
        Returns:
            Dictionary with updated state counts
        """
        # Shuffle node order for random update sequence
        nodes = list(self.network.nodes())
        self.rng.shuffle(nodes)
        
        for node in nodes:
            current_state = self.states[node]
            
            # Randomly select a different state to transition to
            possible_states = [s for s in BeliefState if s != current_state]
            proposed_state = self.rng.choice(possible_states)
            
            # Calculate energy change and transition probability
            energy_delta = self._calculate_energy_delta(node, proposed_state, tnp_state)
            p_transition = self._transition_probability(energy_delta)
            
            # Attempt state transition
            if self.rng.random() < p_transition:
                self.states[node] = proposed_state
        
        # Count states
        counts = {state: 0 for state in BeliefState}
        for state in self.states.values():
            counts[state] += 1
        
        # Update history
        for state in BeliefState:
            self.state_counts[state].append(counts[state])
        
        # Copy current states to history
        self.history.append(self.states.copy())
        
        return counts
    
    def get_network_entropy(self) -> float:
        """
        Calculate the entropy of the belief state distribution.
        
        Returns:
            Shannon entropy of the belief distribution
        """
        counts = {state: 0 for state in BeliefState}
        for state in self.states.values():
            counts[state] += 1
        
        n = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / n
                entropy -= p * np.log2(p)
        
        return entropy
    
    def get_clustering_coefficient(self) -> dict[BeliefState, float]:
        """
        Calculate clustering coefficients by belief state.
        
        Returns:
            Dictionary mapping belief states to clustering coefficients
        """
        state_nodes = {state: [] for state in BeliefState}
        for node, state in self.states.items():
            state_nodes[state].append(node)
        
        clustering = {}
        for state, nodes in state_nodes.items():
            if nodes:
                subgraph = self.network.subgraph(nodes)
                clustering[state] = nx.average_clustering(subgraph)
            else:
                clustering[state] = 0.0
        
        return clustering
    
    def get_state_distribution(self) -> dict[BeliefState, float]:
        """
        Get the current distribution of belief states.
        
        Returns:
            Dictionary mapping belief states to fractions
        """
        counts = {state: 0 for state in BeliefState}
        for state in self.states.values():
            counts[state] += 1
        
        n = sum(counts.values())
        return {state: count / n for state, count in counts.items()}
    
    def simulate(self, timesteps: int, ode_results: Dict = None) -> Dict:
        """
        Run the ABM simulation for a given number of timesteps.
        
        Args:
            timesteps: Number of timesteps to simulate
            ode_results: Results from ODE simulation (for coupling)
            
        Returns:
            Dictionary with simulation results
        """
        if ode_results is None:
            # Run the ODE model first if results not provided
            ode_results = self.ode_model.simulate((0, timesteps), dt=1.0)
        
        # Extract TNP trajectories
        times = ode_results['times']
        T = ode_results['trust']
        N = ode_results['entropy']
        P = ode_results['pressure']
        
        # Ensure we have enough ODE timesteps
        assert len(times) >= timesteps, "Not enough ODE timesteps for ABM simulation"
        
        # Reset history
        self.history = []
        self.state_counts = {state: [] for state in BeliefState}
        
        # Run ABM for each timestep
        for t in range(timesteps):
            tnp_state = (T[t], N[t], P[t])
            self.update(tnp_state)
        
        # Calculate state distributions over time
        state_fractions = {
            state: [count / self.params.n_agents for count in counts]
            for state, counts in self.state_counts.items()
        }
        
        # Calculate network metrics
        entropy_over_time = [self.get_network_entropy() for _ in range(timesteps)]
        clustering_over_time = [self.get_clustering_coefficient() for _ in range(timesteps)]
        
        return {
            'times': times[:timesteps],
            'state_counts': self.state_counts,
            'state_fractions': state_fractions,
            'entropy': entropy_over_time,
            'clustering': clustering_over_time,
            'final_states': self.states.copy()
        }


# Example usage
if __name__ == "__main__":
    # Set up the ODE model
    ode_model = PistonLeakODE()
    
    # Set up the ABM
    abm_params = ABMParams(
        n_agents=1000,  # Smaller for testing
        init_believer_prob=0.7,
        init_skeptic_prob=0.2,
        init_agnostic_prob=0.1
    )
    abm_model = PistonLeakABM(params=abm_params, ode_model=ode_model)
    
    # Run ODE simulation
    ode_results = ode_model.simulate((0, 100), dt=1.0)
    
    # Run ABM simulation with ODE results
    abm_results = abm_model.simulate(100, ode_results)
    
    # Print some results
    print("Final belief distribution:")
    final_dist = abm_model.get_state_distribution()
    for state, fraction in final_dist.items():
        print(f"  {state.name}: {fraction:.3f}")
    
    print(f"Final network entropy: {abm_results['entropy'][-1]:.3f}")
