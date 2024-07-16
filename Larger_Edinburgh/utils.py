import os
import sys
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sumolib import checkBinary
import traci
import torch
import sumolib
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import sys
import io
from contextlib import redirect_stdout
import pandas as pd

if 'SUMO_HOME' in os.environ:
    print('SUMO_HOME found')
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

# sumoBinary = checkBinary('sumo-gui')
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

def check_parallel_graph():
    net = sumolib.net.readNet('./config/osm.net.xml.gz')
    # Extract nodes and edges
    nodes = net.getNodes()
    edges = net.getEdges()

    edges_from_to = {}
    for edge in edges:
        from_node = edge.getFromNode().getID()
        to_node = edge.getToNode().getID()
        if (from_node, to_node) not in edges_from_to:
            edges_from_to[(from_node, to_node)] = [edge]
        else:
            edges_from_to[(from_node, to_node)].append(edge)

    for from_node, to_node in edges_from_to.keys():
        if len(edges_from_to[(from_node, to_node)]) > 1:
            print(from_node, to_node)
            for edge in edges_from_to[(from_node, to_node)]:
                print(edge.getID())
                num_lanes = edge.getLaneNumber()
                speed_limit = edge.getSpeed()
                length = edge.getLength()

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

# pad trajectories to have the same length, then save as csv
def to_csv(trajectories, name='edinburgh_trajectories.csv'):
    trajectories = padding_dict(trajectories)
    df = pd.DataFrame(list(trajectories.values()), index=trajectories.keys())
    df.to_csv(name)

def get_planned_path():
    planned_path = {}
    traci.start(sumoCmd)
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