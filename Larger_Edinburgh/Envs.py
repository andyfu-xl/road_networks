import gym
import traci
import torch
import random
import copy
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from utils import *
from collections import deque, namedtuple


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RoutingGym(gym.Env):
    def __init__(self, sumoCmd, max_steps=1100, n_nodes=57, max_routing_steps=30, max_n_neighbors=6):
        self.sumoCmd = sumoCmd
        self.step_counter = 0
        self.max_steps = max_steps
        self.n_nodes = n_nodes
        self.start_node = None
        self.end_node = None
        self.current_node = None
        self.node_features = None
        self.adj_matrix = None
        self.edge_index = None
        self.hop_thresh = None
        self.routing_done = False
        self.routing_steps = 0
        self.min_n_hops = None
        self.end_node_indicator = torch.zeros(n_nodes)
        self.max_routing_steps = max_routing_steps
        self.n_hop_matrix = None
        self.neighbors_indicator = None
        self.action_space = None
        self.to_remove_indices = None
        self.prunned_adj_matrix = None
        self.prunned_n_hop_matrix = None
        self.state = None
        self.max_n_neighbors = max_n_neighbors

        self.xs = None
        self.ys = None
        self.norm_x = None
        self.norm_y = None

    def reset(self):
        try:
            traci.close()
        except:
            pass
        traci.start(self.sumoCmd)
        self.step_counter = 0

        while self.step_counter < 400:
            traci.simulationStep()
            self.step_counter += 1

    def node_pruning(self):
        self.prunned_adj_matrix = copy.deepcopy(self.adj_matrix)
        self.prunned_n_hop_matrix = copy.deepcopy(self.n_hop_matrix)
        neighbor_indices = np.where(self.adj_matrix[self.current_node] == 1)[0]
        if len(neighbor_indices) >= self.max_n_neighbors:
            two_hop_neighbours_indices = np.where(self.n_hop_matrix[self.current_node] == 2)[0]
            two_hop_neighbours_mask = (self.n_hop_matrix[self.current_node] == 2).type(torch.int)
            # direct neighbours connectivities with two hop neighbours
            neighbour_dict = {}
            for neighbour_index in neighbor_indices:
                neighbour_dict[neighbour_index] = two_hop_neighbours_indices[np.where(self.adj_matrix[neighbour_index][two_hop_neighbours_indices] == 1)[0]]
            # sort by the number of two hop neighbours
            neighbour_dict = dict(sorted(neighbour_dict.items(), key=lambda item: len(item[1]), reverse=True))

            self.to_remove_indices = []
            action_space = 0
            for neighbour_index, two_hop_neighbours_indices in neighbour_dict.items():
                mask_sum_before = torch.sum(two_hop_neighbours_mask)
                two_hop_neighbours_mask[two_hop_neighbours_indices] = 0
                mask_sum_after = torch.sum(two_hop_neighbours_mask)
                if mask_sum_after < mask_sum_before:
                    action_space += 1
                else:
                    self.to_remove_indices.append(neighbour_index)
            if action_space < self.max_n_neighbors:
                self.to_remove_indices = random.sample(self.to_remove_indices, len(self.to_remove_indices) - (self.max_n_neighbors - action_space))
            self.prunned_adj_matrix[self.to_remove_indices, :] = 0
            self.prunned_adj_matrix[:, self.to_remove_indices] = 0
            self.prunned_n_hop_matrix[self.to_remove_indices, :] = -100
            self.prunned_n_hop_matrix[:, self.to_remove_indices] = -100

    def step(self):
        traci.simulationStep()
        self.routing_done = False
        self.routing_steps = 0
        self.step_counter += 1
        self.adj_matrix, self.xs, self.ys = intervehicleConnectivity_xs_ys(800)
        self.adj_matrix = simplify_graph(self.adj_matrix)
        self.select_start_end_nodes()
        self.current_node = self.start_node
        self.adj_matrix = F.pad(self.adj_matrix, (0, self.n_nodes - self.adj_matrix.size(0), 
                                                  0, self.n_nodes - self.adj_matrix.size(1)), "constant", 0)
        self.n_hop_matrix = F.pad(self.n_hop_matrix, (0, self.n_nodes - self.n_hop_matrix.size(0), 
                                                      0, self.n_nodes - self.n_hop_matrix.size(1)), "constant", -100)
        self.node_pruning()
        # set diagonal to 0
        self.prunned_n_hop_matrix = self.prunned_n_hop_matrix - torch.diag(torch.diag(self.prunned_n_hop_matrix))
        curr_node_indicator = torch.zeros(self.n_nodes)
        curr_node_indicator[self.current_node] = 1
        self.end_node_indicator = self.prunned_adj_matrix[self.end_node]
        distances = self.prunned_n_hop_matrix[self.end_node]
        distances[distances == -100] = 8
        distances[distances > 7] = 7
        one_hot_distances = F.one_hot(distances.long(), num_classes=8).type(torch.float32)
        neighbour_indicator = self.prunned_adj_matrix[self.current_node]
        self.node_features = torch.cat((one_hot_distances, self.end_node_indicator.unsqueeze(1), 
                                        curr_node_indicator.unsqueeze(1), neighbour_indicator.unsqueeze(1), 
                                        self.norm_x, self.norm_y), dim=1).to(device)
        self.state = Data(x=self.node_features, edge_index=self.get_edge_index())
        return self.state
    
    def refresh(self):
        self.select_start_end_nodes(True)
        self.routing_done = False
        self.routing_steps = 0
        self.current_node = self.start_node
        self.node_pruning()
        self.prunned_n_hop_matrix = self.prunned_n_hop_matrix - torch.diag(torch.diag(self.prunned_n_hop_matrix))
        curr_node_indicator = torch.zeros(self.n_nodes)
        curr_node_indicator[self.current_node] = 1
        self.end_node_indicator = self.prunned_adj_matrix[self.end_node]
        neighbour_indicator = self.prunned_adj_matrix[self.current_node]
        distances = self.prunned_n_hop_matrix[self.end_node]
        distances[distances == -100] = 8
        distances[distances > 7] = 7
        one_hot_distances = F.one_hot(distances.long(), num_classes=8).type(torch.float32)
        neighbour_indicator = self.prunned_adj_matrix[self.current_node]
        self.node_features = torch.cat((one_hot_distances, self.end_node_indicator.unsqueeze(1),
                                            curr_node_indicator.unsqueeze(1), neighbour_indicator.unsqueeze(1), 
                                            self.norm_x, self.norm_y), dim=1).to(device)
        self.state = Data(x=self.node_features, edge_index=self.get_edge_index())
        return self.state


    def select_start_end_nodes(self, refresh=False):
        if not refresh:
            self.n_hop_matrix = bfs_distance(self.adj_matrix)
            self.hop_thresh = min(self.n_hop_matrix.max(), 5)
        starts, ends = torch.where(self.hop_thresh == self.n_hop_matrix)
        starts = starts.tolist()
        ends = ends.tolist()
        self.start_node, self.end_node = random.choice(list(zip(starts, ends)))
        # minimal number of hops between start and end nodes
        self.min_n_hops = self.n_hop_matrix[self.start_node, self.end_node]

        start_x = self.xs[self.start_node]
        start_y = self.ys[self.start_node]
        end_x = self.xs[self.end_node]
        end_y = self.ys[self.end_node]
        self.norm_x = (self.xs - end_x) / (start_x - end_x)
        self.norm_y = (self.ys - end_y) / (start_y - end_y)
        self.norm_x = F.pad(self.norm_x, (0, 0, 0, self.n_nodes - self.norm_x.size(0)), "constant", 0)
        self.norm_y = F.pad(self.norm_y, (0, 0, 0, self.n_nodes - self.norm_y.size(0)), "constant", 0)

    def act(self, neighbor_index):
        self.routing_steps += 1
        neighbors = torch.where(self.prunned_adj_matrix[self.current_node] == 1)[0]
        valid_action_size = len(neighbors)
        if valid_action_size <= neighbor_index:
            self.routing_done = self.routing_steps >= self.max_routing_steps
            if self.routing_done:
                # print("Failed, ", self.min_n_hops)
                return self.state, torch.tensor(-1).to(device), self.routing_done
            return self.state, torch.tensor(-0.15).to(device), self.routing_done
        else:
            next_hop = neighbors[neighbor_index]
            reward = self.compute_reward(next_hop)
            self.current_node = next_hop
            self.node_pruning()
            curr_node_indicator = torch.zeros(self.n_nodes)
            curr_node_indicator[self.current_node] = 1
            distances = self.prunned_n_hop_matrix[self.end_node]
            distances[distances == -100] = 8
            distances[distances > 7] = 7
            one_hot_distances = F.one_hot(distances.long(), num_classes=8).type(torch.float32)
            neighbour_indicator = self.prunned_adj_matrix[self.current_node]
            self.node_features = torch.cat((one_hot_distances, self.end_node_indicator.unsqueeze(1),
                                             curr_node_indicator.unsqueeze(1), neighbour_indicator.unsqueeze(1),
                                               self.norm_x, self.norm_y), dim=1).to(device)
            self.state = Data(x=self.node_features, edge_index=self.get_edge_index())
            return self.state, torch.tensor(reward).to(device), self.routing_done

    def get_adj_matrix(self):
        return copy.deepcopy(self.adj_matrix).to(device)
    
    def get_edge_index(self):
        self.edge_index, _ = dense_to_sparse(self.prunned_adj_matrix)
        return copy.deepcopy(self.edge_index).to(device)
        
    def compute_reward(self, next_hop):
        if self.routing_steps >= self.max_routing_steps:
            # print("Failed, ", self.min_n_hops)
            self.routing_done = True
            return -1
        elif self.adj_matrix[self.current_node, self.end_node] == 1:
            # print("Routing done, number of hops: ", self.routing_steps, " minimum number of hops: ", self.min_n_hops)
            self.routing_done = True
            return (self.min_n_hops / self.routing_steps)
        elif self.n_hop_matrix[self.current_node, self.end_node] > self.n_hop_matrix[next_hop, self.end_node]:
            return 0.1
        else:
            return -0.15

    def get_action_mask(self):
        action_mask = copy.deepcopy(self.prunned_adj_matrix[self.current_node])
        action_mask = F.pad(action_mask, (0, self.n_nodes - action_mask.size(0)), "constant", 0).to(device)
        return action_mask
        
    def sim_done(self):
        """
        function: get the done state of simulation.
        """
        return not (shouldContinueSim() and self.step_counter <= self.max_steps)




class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition',
                        ('data', 'action', 'next_state', 'reward', 'shuffle_indices'))

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)