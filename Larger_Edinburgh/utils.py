import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from sumolib import checkBinary
import traci
import torch.nn.functional as F
import io
from contextlib import redirect_stdout
import pandas as pd
from collections import deque

if 'SUMO_HOME' in os.environ:
    print('SUMO_HOME found')
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

sumoBinary = checkBinary('sumo')
roadNetwork = "./config/osm.sumocfg"
sumoCmd = [sumoBinary, "-c", roadNetwork, "--start", "--quit-on-end"]

def randomTrips(dur=10000, density=12):
    os.system("python $SUMO_HOME/tools/randomTrips.py -n config/osm.net.xml.gz -r config/osm.passenger.trips.xml -e " + str(dur) + " -l --insertion-density=" + str(density))

def shouldContinueSim():
    numVehicles = traci.simulation.getMinExpectedNumber()
    return True if numVehicles > 0 else False

def restart(dur, density):
    with io.StringIO() as buf, redirect_stdout(buf):
        try:
            traci.close()
        except:
            pass
        randomTrips(dur, density)
        traci.start(sumoCmd)
        step = 0
        for i in range(100):
            traci.simulationStep()
            step += 1

def quickstart(cmd):
    try:
        traci.close()
    except:
        pass
    traci.start(cmd)

def close():
    traci.close()

def get_trajectories():
    trajectories = {}
    traci.start(sumoCmd)
    while shouldContinueSim():
        traci.simulationStep()
        for vehicle in traci.vehicle.getIDList():
            if vehicle not in trajectories:
                trajectories[vehicle] = []
            x, y = traci.vehicle.getPosition(vehicle)
            speed = traci.vehicle.getSpeed(vehicle)
            direction = np.deg2rad(traci.vehicle.getAngle(vehicle))
            velocity_x = speed * np.sin(direction)
            velocity_y = speed * np.cos(direction)
            trajectories[vehicle] += [x, y, velocity_x, velocity_y]
    traci.close()
    return trajectories

# plot the first 10 trajectories
def plot_trajectories(trajectories, n=10):
    for i, vehicle in enumerate(trajectories.keys()):
        if i > n:
            break
        trajectory = trajectories[vehicle]
        x = trajectory[::4]
        y = trajectory[1::4]
        plt.plot(x, y)
    plt.show()

def padding_dict(dict):
    max_len = max([len(dict[key]) for key in dict.keys()])
    for key in dict.keys():
        dict[key] += [0] * (max_len - len(dict[key]))
    return dict

def padding_dict2(dict):
    max_len = max([len(dict[key]) for key in dict.keys()])
    for key in dict.keys():
        dict[key] += [[0,0]] * (max_len - len(dict[key]))
    return dict

def padding_dict1(dict):
    max_len = max([len(dict[key]) for key in dict.keys()])
    for key in dict.keys():
        dict[key] += [[-1,-1]] * (max_len - len(dict[key]))
    return dict

# pad trajectories to have the same length, then save as csv
def to_csv(trajectories, name='edinburgh_trajectories.csv'):
    trajectories = padding_dict(trajectories)
    df = pd.DataFrame(list(trajectories.values()), index=trajectories.keys())
    df.to_csv(name)

def to_csv_1(trajectories, name='edinburgh_trajectories.csv'):
    trajectories = padding_dict1(trajectories)
    df = pd.DataFrame(list(trajectories.values()), index=trajectories.keys())
    df.to_csv(name)

def get_planned_path():
    planned_path = {}
    quickstart(sumoCmd)
    step = 0
    while shouldContinueSim():
        traci.simulationStep()
        for veh in traci.vehicle.getIDList():
            if veh in planned_path:
                continue
            points = []
            curr_lane_shape = traci.lane.getShape(traci.vehicle.getLaneID(veh))
            for point in curr_lane_shape:
                points.append([point[0], point[1]])
            for link in traci.vehicle.getNextLinks(veh):
                via = link[-4]
                for point in traci.lane.getShape(via):
                    if point[0] == points[-1][0] and point[1] == points[-1][1]:
                        continue
                    points.append([point[0], point[1]])
                lane = link[0]
                for point in traci.lane.getShape(lane):
                    if point[0] == points[-1][0] and point[1] == points[-1][1]:
                        continue
                    points.append([point[0], point[1]])
            planned_path[veh] = points
    traci.close()
    return padding_dict2(planned_path)

def missing(x, y, z, p):
    return torch.tensor(np.repeat(np.random.rand(x * y) < p, z).reshape(x, y, z)).float()

def generate_masks(tensors, min_mask_ratio=0.2, max_mask_ratio=0.4, missing_ratio=0.5, complete_traj_ratio=0.8):
    initial_masks = missing(tensors.shape[0], tensors.shape[1], tensors.shape[2], missing_ratio)
    masks = []
    for initial_mask in initial_masks:
        if np.random.rand() < complete_traj_ratio:
            masks.append(torch.zeros_like(initial_mask).tolist())
            continue
        seq_length = initial_mask.shape[0]
        mask_start = np.random.randint(int(seq_length * min_mask_ratio), int(seq_length * max_mask_ratio))
        mask = torch.zeros_like(initial_mask)
        mask[:, :mask_start] = 1
        mask = initial_mask * mask
        mask[0] = 0
        mask[1] = 0
        masks.append(mask.tolist())
    return torch.tensor(masks)

def project_to_nearest(prediction, planned_path):
    with torch.no_grad():
        starts = planned_path[:, :-1, :]
        to = planned_path[:, 1:, :]

        prediction = prediction.unsqueeze(1).repeat(1, starts.shape[1], 1)
        ap = prediction - starts
        ab = to - starts
        numerator = torch.einsum('ijk,ijk->ij', ap, ab)
        denominator = torch.einsum('ijk,ijk->ij', ab, ab)
        t = numerator / denominator
        t = torch.nan_to_num(t, nan=0.0)
        t = torch.clamp(t, 0, 1)
        projections = starts + t.unsqueeze(2) * ab
        diff = projections - prediction
        distances = torch.norm(diff, dim=2)
        min_indices = torch.argmin(distances, dim=1)
        projections = projections[range(projections.shape[0]), min_indices]
        return projections, min_indices

def project_to_nearest_with_checkpoints(prediction, planned_path):
    with torch.no_grad():
        starts = planned_path[:, :-1, :]
        to = planned_path[:, 1:, :]

        prediction = prediction.unsqueeze(1).repeat(1, starts.shape[1], 1)
        ap = prediction - starts
        ab = to - starts
        numerator = torch.einsum('ijk,ijk->ij', ap, ab)
        denominator = torch.einsum('ijk,ijk->ij', ab, ab)
        t = numerator / denominator
        t = torch.nan_to_num(t, nan=0.0)
        t = torch.clamp(t, 0, 1)
        projections = starts + t.unsqueeze(2) * ab
        diff = projections - prediction
        distances = torch.norm(diff, dim=2)
        min_indices = torch.argmin(distances, dim=1)
        projections = projections[range(projections.shape[0]), min_indices]
        next_checkpoint = to[range(to.shape[0]), min_indices]
        pad_to = F.pad(to, (0, 0, 1, 0), value=0)
        next_next_checkpoint = pad_to[range(to.shape[0]), min_indices + 1]
        projections_with_checkpoints = torch.cat((projections, next_checkpoint, next_next_checkpoint), dim=1)
        return projections_with_checkpoints

def intervehicleConnectivity(threshold = None):
    xs = []
    ys = []
    for vehicle in traci.vehicle.getIDList():
        x, y = traci.vehicle.getPosition(vehicle)
        xs.append(x)
        ys.append(y)
    xs = torch.tensor(xs, dtype=torch.float32).view(-1,1)
    ys = torch.tensor(ys, dtype=torch.float32).view(-1,1)
    intervehicle_distances = torch.sqrt((xs - xs.t())**2 + (ys - ys.t())**2)
    if threshold is not None:
        # make the distances 1 if less than the threshold, 0 otherwise
        connectivity = torch.where(intervehicle_distances < threshold, torch.ones_like(intervehicle_distances), torch.zeros_like(intervehicle_distances))
    assert connectivity.shape[0] == intervehicle_distances.shape[0]
    return connectivity

def intervehicleConnectivity_xs_ys(threshold = None):
    xs = []
    ys = []
    for vehicle in traci.vehicle.getIDList():
        x, y = traci.vehicle.getPosition(vehicle)
        xs.append(x)
        ys.append(y)
    xs = torch.tensor(xs, dtype=torch.float32).view(-1,1)
    ys = torch.tensor(ys, dtype=torch.float32).view(-1,1)
    intervehicle_distances = torch.sqrt((xs - xs.t())**2 + (ys - ys.t())**2)
    if threshold is not None:
        # make the distances 1 if less than the threshold, 0 otherwise
        connectivity = torch.where(intervehicle_distances < threshold, torch.ones_like(intervehicle_distances), torch.zeros_like(intervehicle_distances))
    return connectivity, xs, ys


def bfs_distance(adj_matrix):
    n_hop_matrix = torch.ones_like(adj_matrix) * (-100)
    for start_node in range(adj_matrix.size(0)):
        visited = [0] * adj_matrix.size(0)
        queue = deque([(start_node, 0)])
        visited[start_node] = True
        
        while queue:
            current_node, current_dist = queue.popleft()
            
            for neighbor, connected in enumerate(adj_matrix[current_node]):
                if connected and not visited[neighbor]:
                    queue.append((neighbor, current_dist + 1))
                    visited[neighbor] = True
                    n_hop_matrix[start_node, neighbor] = current_dist + 1
    return n_hop_matrix

def simplify_graph(adj_matrix):
    adj_matrix = adj_matrix - torch.eye(adj_matrix.size(0))
    degrees = torch.sum(adj_matrix, axis=0)
    nodes_to_keep = np.where(degrees > 0)[0]
    new_adj_matrix = adj_matrix[np.ix_(nodes_to_keep, nodes_to_keep)]
    return new_adj_matrix