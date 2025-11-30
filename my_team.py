# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

def create_team(first_index, second_index, is_red, first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):
    #Utiliza un algoritmo greedy com look de un paso, mira las acciones posibles siguientes evalua el estado resultante y elige segun el que tenga mejor puntuación
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.target_food = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions and len(legal_actions) > 1:
            legal_actions.remove(Directions.STOP)
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            return self.get_action_to_home(game_state, legal_actions)
        if my_state.num_carrying >= 5:
            return self.get_action_to_home(game_state, legal_actions)
        if my_state.num_carrying >= 3:
            boundary_x = game_state.data.layout.width // 2
            if self.red:
                boundary_x -= 1
            boundary_positions = [(boundary_x, y) for y in range(game_state.data.layout.height) if not game_state.has_wall(boundary_x, y)]
            if boundary_positions:
                min_home_dist = min([self.get_maze_distance(my_pos, pos) for pos in boundary_positions])
                if min_home_dist <= 5:
                    return self.get_action_to_home(game_state, legal_actions)
        best_action = None
        best_value = float('-inf')
        for action in legal_actions:
            successor = game_state.generate_successor(self.index, action)
            value = self.evaluate_action(game_state, successor)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action if best_action else random.choice(legal_actions)

    def get_action_to_home(self, game_state, legal_actions):
        my_pos = game_state.get_agent_position(self.index)
        boundary_x = game_state.data.layout.width // 2
        if self.red:
            boundary_x -= 1
        boundary_positions = [(boundary_x, y) for y in range(game_state.data.layout.height) if not game_state.has_wall(boundary_x, y)]
        if not boundary_positions:
            return random.choice(legal_actions)
        best_action = None
        best_dist = float('inf')
        for action in legal_actions:
            successor = game_state.generate_successor(self.index, action)
            new_pos = successor.get_agent_position(self.index)
            min_dist = min([self.get_maze_distance(new_pos, pos) for pos in boundary_positions])
            if min_dist < best_dist:
                best_dist = min_dist
                best_action = action
        return best_action if best_action else random.choice(legal_actions)

    def evaluate_action(self, current_state, successor_state):
        my_state = successor_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        if my_pos is not None:
            my_pos = (int(my_pos[0]), int(my_pos[1]))
        score = 0
        food_list = self.get_food(successor_state).as_list()
        carrying = my_state.num_carrying
        #Hemos añadido esta linea de codigo ya que veiamos que priorizaba recoger alrededor de 5 frutas siempre en lugar de ir poco a poco, cuando ya tenia diversas recolectadas. Esto causaba que tuviera que recorrer demasiado perdiendo la mayoria de veces por el camino
        #Es decir: si Pacman lleva mucha comida encima, el peso de buscar más comida baja drásticamente.
        if len(food_list) > 0:
            min_food_dist = min([self.get_maze_distance(my_pos, food) for food in food_list])
            if carrying >= 10:
                score -= min_food_dist * 0.05
            elif carrying >= 7:
                score -= min_food_dist * 0.15
            elif carrying >= 5:
                score -= min_food_dist * 0.25
            else:
                score -= min_food_dist * 0.4
        if carrying < 10:
            score -= len(food_list) * 12
        else:
            score -= len(food_list) * 2
        enemies = [successor_state.get_agent_state(i) for i in self.get_opponents(successor_state)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]
        if len(ghosts) > 0 and my_state.is_pacman:
            ghost_positions = [(int(g.get_position()[0]), int(g.get_position()[1])) for g in ghosts]
            ghost_distances = [self.get_maze_distance(my_pos, pos) for pos in ghost_positions]
            min_ghost_dist = min(ghost_distances)
            normal_ghosts = [g for g in ghosts if g.scared_timer == 0]
            if len(normal_ghosts) > 0:
                normal_positions = [(int(g.get_position()[0]), int(g.get_position()[1])) for g in normal_ghosts]
                normal_distances = [self.get_maze_distance(my_pos, pos) for pos in normal_positions]
                min_normal_dist = min(normal_distances)
                if min_normal_dist <= 1:
                    score -= 20
                elif min_normal_dist <= 3:
                    score -= 15
                elif min_normal_dist <= 5:
                    score -= 0.5
        capsules = self.get_capsules(successor_state)
        if len(capsules) > 0 and len(ghosts) > 0:
            min_capsule_dist = min([self.get_maze_distance(my_pos, cap) for cap in capsules])
            score -= min_capsule_dist * 0.7
        if my_state.is_pacman:
            score += 1.2
        score += self.get_score(successor_state) * 50
        return score

class DefensiveAgent(CaptureAgent):
    #Minimax con poda de alphabeta que minimiza el daño del atacante
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        #Solo mira un turno hacia el futuro, ya que sino el tiempo de computo hacia que no fuera fluido
        self.depth = 1

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if Directions.STOP in legal_actions and len(legal_actions) > 1:
            legal_actions.remove(Directions.STOP)
        best_action = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        num_agents = game_state.get_num_agents()
        next_agent = (self.index + 1) % num_agents
        for action in legal_actions:
            successor = game_state.generate_successor(self.index, action)
            value = self.alpha_beta(successor, self.depth, next_agent, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
        return best_action

    def alpha_beta(self, game_state, depth, agent_index, alpha, beta):
        if game_state.is_over() or depth == 0:
            return self.evaluate_state(game_state)
        agent_state = game_state.get_agent_state(agent_index)
        if agent_state.get_position() is None:
            return self.evaluate_state(game_state)
        legal_actions = game_state.get_legal_actions(agent_index)
        if not legal_actions:
            return self.evaluate_state(game_state)
        opponents = self.get_opponents(game_state)
        is_opponent = agent_index in opponents
        num_agents = game_state.get_num_agents()
        next_agent = (agent_index + 1) % num_agents
        next_depth = depth - 1 if next_agent == self.index else depth
        if is_opponent:
            next_state = game_state.get_agent_state(agent_index)
            if next_state.get_position() is None:
                return self.alpha_beta(game_state, next_depth, next_agent, alpha, beta)
        if not is_opponent:
            max_value = float('-inf')
            for action in legal_actions:
                successor = game_state.generate_successor(agent_index, action)
                value = self.alpha_beta(successor, next_depth, next_agent, alpha, beta)
                max_value = max(max_value, value)
                if max_value > beta:
                    return max_value
                alpha = max(alpha, max_value)
            return max_value
        else:
            min_value = float('inf')
            limited_actions = legal_actions[:3]
            for action in limited_actions:
                successor = game_state.generate_successor(agent_index, action)
                value = self.alpha_beta(successor, next_depth, next_agent, alpha, beta)
                min_value = min(min_value, value)
                if min_value < alpha:
                    return min_value
                beta = min(beta, min_value)
            return min_value

    def evaluate_state(self, game_state):
        features = util.Counter()
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        if my_pos is not None:
            my_pos = (int(my_pos[0]), int(my_pos[1]))
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            inv_positions = [(int(inv.get_position()[0]), int(inv.get_position()[1])) for inv in invaders]
            dists = [self.get_maze_distance(my_pos, pos) for pos in inv_positions]
            features['invader_distance'] = min(dists)
        else:
            boundary_x = game_state.data.layout.width // 2
            if not self.red:
                boundary_x += 1
            boundary_positions = [(boundary_x, y) for y in range(game_state.data.layout.height) if not game_state.has_wall(boundary_x, y)]
            if len(boundary_positions) > 0:
                min_boundary_dist = min([self.get_maze_distance(my_pos, pos) for pos in boundary_positions])
                features['patrol_distance'] = min_boundary_dist
        features['on_defense'] = 1 if not my_state.is_pacman else 0
        food_defending = self.get_food_you_are_defending(game_state).as_list()
        if len(food_defending) > 0:
            min_food_dist = min([self.get_maze_distance(my_pos, food) for food in food_defending])
            features['food_defense_distance'] = min_food_dist
        features['scared'] = 1 if my_state.scared_timer > 0 else 0
        #Pesos que hemos modificado para que obtenga el mejor resultado dado esta depth = 1
        weights = {'num_invaders': -50, 'invader_distance': -5, 'on_defense': 10, 'patrol_distance': -2.5, 'food_defense_distance': -0.25, 'capsule_defense_distance': -0.1, 'scared': -25}
        score = sum(features[key] * weights[key] for key in features.keys())

        return score


