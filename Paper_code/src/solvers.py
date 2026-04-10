import networkx as nx
import time
import random
import math

# =============================================================================
# Helper: Knowledge-Based Potentials (Unchanged)
# =============================================================================

def get_knowledge_based_weight(lig_type, rec_type):
    """
    Returns the interaction weight for a pharmacophore feature pair.
    Values directly from Table S1 in arXiv:1902.00462 (PDBbind 2015 + DrugScore, reflected for max-clique).
    Higher value = stronger preference.
    Matrix is symmetric; metal interactions prioritized separately as paper omits metals.
    """
    
    # Exact values from paper's Table S1 (symmetric matrix filled)
    KB_POTENTIAL = {
        # Ionic / Charge interactions
        tuple(sorted(("NegIonizable", "NegIonizable"))): 0.2953,
        tuple(sorted(("PosIonizable", "PosIonizable"))): 0.1596,
        tuple(sorted(("PosIonizable", "NegIonizable"))): 0.6459,
        
        # H-bond
        tuple(sorted(("Donor", "Donor"))): 0.5244,
        tuple(sorted(("Acceptor", "Acceptor"))): 0.5478,
        tuple(sorted(("Donor", "Acceptor"))): 0.6686,
        tuple(sorted(("Donor", "PosIonizable"))): 0.4781,
        tuple(sorted(("Donor", "NegIonizable"))): 0.7114,
        tuple(sorted(("Acceptor", "PosIonizable"))): 0.7029,
        tuple(sorted(("Acceptor", "NegIonizable"))): 0.6450,
        
        # Hydrophobic
        tuple(sorted(("Hydrophobe", "Hydrophobe"))): 0.0504,
        tuple(sorted(("Hydrophobe", "PosIonizable"))): 0.0679,
        tuple(sorted(("Hydrophobe", "NegIonizable"))): 0.1802,
        tuple(sorted(("Hydrophobe", "Donor"))): 0.1453,
        tuple(sorted(("Hydrophobe", "Acceptor"))): 0.2317,
        
        # Aromatic
        tuple(sorted(("Aromatic", "Aromatic"))): 0.1943,
        tuple(sorted(("Aromatic", "PosIonizable"))): 0.1555,
        tuple(sorted(("Aromatic", "NegIonizable"))): 0.0,
        tuple(sorted(("Aromatic", "Donor"))): 0.1091,
        tuple(sorted(("Aromatic", "Acceptor"))): 0.0770,
        tuple(sorted(("Aromatic", "Hydrophobe"))): 0.0795,
    }

    # Normalize Feature Names (handle synonyms from your code)
    type_map = {
        "ZnBinder": "Metal",
        "CuBinder": "Metal",
        "Metal": "Metal",
        "LumpedHydrophobe": "Hydrophobe",
        "Hydrophobe": "Hydrophobe",
        "Aromatic": "Aromatic",
        "Donor": "Donor",           # H-bond donor
        "Acceptor": "Acceptor",     # H-bond acceptor
        "PosIonizable": "PosIonizable",
        "NegIonizable": "NegIonizable"
    }
    
    t1 = type_map.get(lig_type, lig_type)
    t2 = type_map.get(rec_type, rec_type)

    # Metal override: Paper omits metals, but coordination is critical in many cases
    if "Metal" in (t1, t2):
        if any(x in (t1, t2) for x in ["Donor", "Acceptor", "NegIonizable", "Aromatic", "PosIonizable"]):
            return 10.0  # Strong coordination priority
        return 0.1  # Weak otherwise

    # Lookup (symmetric key)
    key = tuple(sorted((t1, t2)))
    return KB_POTENTIAL.get(key, 0.05)  # Default for unseen pairs


# =============================================================================
# Solver Class (Updated: Greedy + SA Refinement)
# =============================================================================

class MaxCliqueSolver:
    def __init__(self, graph):
        self.graph = graph

    def solve_greedy(self, node_weights=None):
        """
        Deterministic Greedy using NetworkX's optimized max_weight_clique.
        This provides a solid baseline solution very quickly.
        """
        start_time = time.time()
        
        # 1. Apply weights to graph attributes so NetworkX can use them
        if node_weights:
            for n, w in node_weights.items():
                self.graph.nodes[n]['weight'] = w
            weight_attr = 'weight'
        else:
            weight_attr = None # Unweighted

        try:
            # NetworkX returns (clique_nodes, total_weight)
            clique, _ = nx.algorithms.clique.max_weight_clique(self.graph, weight=weight_attr)
            
            # Recalculate value manually to ensure consistency
            total_val = sum(node_weights.get(n, 1.0) for n in clique) if node_weights else len(clique)
            
            return {
                "method": "NetworkX_Greedy",
                "size": len(clique),
                "value": total_val,
                "clique": sorted(clique),
                "time": time.time() - start_time
            }
        except Exception as e:
            print(f" [Solver Error] Greedy failed: {e}")
            return {'clique': [], 'size': 0, 'value': 0, 'time': 0}

    def solve_sa(self, node_weights=None, iterations=2000, temp=1.0, cooling_rate=0.995):
        """
        Simulated Annealing Refinement.
        STRATEGY:
        1. Initialize with the Greedy solution (don't start from scratch).
        2. Perturb: Kick out 1-3 random nodes.
        3. Refill: Greedily add compatible nodes in random order.
        4. Accept/Reject based on score improvement or Metropolis criterion.
        """
        start_time = time.time()
        
        # 1. Initialization (Stand on the shoulders of Greedy)
        greedy_res = self.solve_greedy(node_weights)
        current_clique = list(greedy_res['clique'])
        best_clique = list(current_clique)
        
        # Helper to calculate score
        def get_score(nodes):
            if node_weights:
                return sum(node_weights.get(n, 1.0) for n in nodes)
            return len(nodes)

        current_score = get_score(current_clique)
        best_score = current_score
        
        nodes_list = list(self.graph.nodes())
        
        # 2. Annealing Loop
        for i in range(iterations):
            # A. Perturbation (The "Shake")
            candidate = list(current_clique)
            
            # Kick out 1 to 3 nodes randomly to make space for new ones
            if len(candidate) > 0:
                num_to_remove = random.randint(1, min(3, len(candidate)))
                for _ in range(num_to_remove):
                    if candidate:
                        candidate.pop(random.randint(0, len(candidate)-1))
            
            # B. Refill (The "Stochastic Repair")
            # Shuffle potential nodes to introduce randomness
            potential_nodes = [n for n in nodes_list if n not in candidate]
            random.shuffle(potential_nodes)
            
            for n in potential_nodes:
                is_compatible = True
                # Fast check against current candidate members
                for existing in candidate:
                    if not self.graph.has_edge(n, existing):
                        is_compatible = False
                        break
                
                if is_compatible:
                    candidate.append(n)
            
            # C. Evaluation
            candidate_score = get_score(candidate)
            delta = candidate_score - current_score
            
            # D. Acceptance Logic
            if delta > 0:
                # Improvement: Always Accept
                current_clique = candidate
                current_score = candidate_score
                
                if current_score > best_score:
                    best_clique = list(current_clique)
                    best_score = current_score
            else:
                # Worsening: Metropolis Criterion (Accept with prob exp(delta/T))
                try:
                    prob = math.exp(delta / temp)
                except OverflowError:
                    prob = 0
                
                if random.random() < prob:
                    current_clique = candidate
                    current_score = candidate_score
            
            # E. Cooling
            temp *= cooling_rate
            
        return {
            "method": "SA_Refined",
            "size": len(best_clique),
            "value": best_score,
            "clique": sorted(best_clique),
            "time": time.time() - start_time
        }