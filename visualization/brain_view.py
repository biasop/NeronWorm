"""Cached graph layout with live spike and membrane activity overlays."""
from math import hypot
import networkx as nx
import pygame

from brain import Brain, Connectome

ROLE_COLORS = {"sensory": (116, 192, 239), "interneuron": (182, 154, 228), "motor": (115, 226, 178)}


class BrainView:
    def __init__(self, connectome: Connectome, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.connectome = connectome
        self.font = pygame.font.Font(None, 19)
        self.small = pygame.font.Font(None, 16)
        self.title = pygame.font.Font(None, 28)
        self.graph = nx.DiGraph()
        layers = {"sensory": 0, "interneuron": 1, "motor": 2}
        for nid, metadata in connectome.neurons.items():
            self.graph.add_node(nid, subset=layers[metadata["role"]])
        for synapse in connectome.chemical_synapses:
            self.graph.add_edge(synapse.source_id, synapse.target_id)
        layout = nx.multipartite_layout(self.graph, subset_key="subset")
        xs, ys = [p[0] for p in layout.values()], [p[1] for p in layout.values()]
        self.positions = {
            nid: (int(45 + (p[0] - min(xs)) / max(1e-9, max(xs) - min(xs)) * (width - 110)),
                  int(110 + (p[1] - min(ys)) / max(1e-9, max(ys) - min(ys)) * (height - 365)))
            for nid, p in layout.items()
        }
        # NetworkX can order a layer through sets; sort positions by ID for a
        # reproducible display across Python hash seeds and library versions.
        for layer in layers.values():
            ids = sorted(nid for nid in self.graph if self.graph.nodes[nid]["subset"] == layer)
            positions = sorted((self.positions[nid] for nid in ids), key=lambda p: p[1])
            for nid, pos in zip(ids, positions):
                self.positions[nid] = pos

    def draw(self, surface: pygame.Surface, brain: Brain, enabled: bool = True) -> None:
        surface.fill((22, 32, 47))
        surface.blit(self.title.render("NEURAL CIRCUIT", True, (228, 238, 247)), (22, 22))
        subtitle = f"{len(brain.neurons)} neurons  /  {len(self.connectome.chemical_synapses)} synapses"
        surface.blit(self.font.render(subtitle, True, (145, 163, 186)), (22, 52))
        if not enabled:
            surface.blit(self.font.render("Brain view hidden · press B", True, (182, 199, 216)), (22, 105))
            return
        for x, role in ((25, "sensory"), (177, "interneuron"), (337, "motor")):
            surface.blit(self.small.render(role.upper(), True, ROLE_COLORS[role]), (x, 80))
        for synapse in self.connectome.chemical_synapses:
            a, b = self.positions[synapse.source_id], self.positions[synapse.target_id]
            color = (45, 83, 72) if synapse.weight > 0 else (112, 53, 71)
            pygame.draw.line(surface, color, a, b, 1)
            dx, dy = b[0] - a[0], b[1] - a[1]
            distance = hypot(dx, dy)
            if distance:
                ux, uy = dx / distance, dy / distance
                tip = (b[0] - 11 * ux, b[1] - 11 * uy)
                back = (tip[0] - 5 * ux, tip[1] - 5 * uy)
                pygame.draw.polygon(surface, color, [tip, (back[0] - 3 * uy, back[1] + 3 * ux),
                                                     (back[0] + 3 * uy, back[1] - 3 * ux)])
        for junction in self.connectome.gap_junctions:
            pygame.draw.line(surface, (71, 141, 194), self.positions[junction.neuron_a_id],
                             self.positions[junction.neuron_b_id], 2)
        for nid, neuron in brain.neurons.items():
            pos = self.positions[nid]
            fraction = max(0.0, min(1.0, (neuron.membrane_potential - neuron.resting_potential) /
                                          (neuron.threshold - neuron.resting_potential)))
            color = tuple(int(c * (0.4 + 0.6 * fraction)) for c in ROLE_COLORS[neuron.role])
            recent = brain.time_ms - neuron.last_spike_time < 45
            if recent:
                pygame.draw.circle(surface, (249, 215, 117), pos, 12, 2)
                color = (255, 233, 153)
            if neuron.role == "motor":
                pygame.draw.rect(surface, color, (pos[0] - 7, pos[1] - 7, 14, 14), border_radius=3)
            else:
                pygame.draw.circle(surface, color, pos, 7)
            label = neuron.name.replace("_", " ")[:16]
            surface.blit(self.small.render(label, True, (187, 200, 216)), (pos[0] - 35, pos[1] + 13))
        y = self.height - 224
        surface.blit(self.small.render("Green: excitation   Red: inhibition   Blue: gap", True, (156, 173, 192)), (22, y))
        y += 26
        for nid in ("motor_left", "motor_right", "food_left", "food_right", "danger_left", "danger_right"):
            mapping = self.connectome.metadata.get("motor_mapping" if nid.startswith("motor") else "sensor_mapping", {})
            mapped = mapping.get(nid, nid)
            if mapped not in brain.neurons:
                continue
            activity = brain.get_activity(mapped)
            color = ROLE_COLORS[brain.neurons[mapped].role]
            label = f"{nid.replace('_', ' ').title():12} {activity:.2f}"
            surface.blit(self.font.render(label, True, color), (22, y))
            pygame.draw.rect(surface, (41, 54, 70), (205, y + 2, self.width - 230, 9), border_radius=4)
            pygame.draw.rect(surface, color, (205, y + 2, int((self.width - 230) * min(1, activity)), 9), border_radius=4)
            y += 23
        surface.blit(self.small.render("Node brightness: voltage   Gold ring: recent spike", True, (146, 164, 184)), (22, self.height - 24))
