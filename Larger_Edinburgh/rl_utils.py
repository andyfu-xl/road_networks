import copy

 
class Knowledges:
    def __init__(self, knowledges={}, delays={}):
        self.knowledges = knowledges
        self.delays = delays
    
    def add_observations(self, vehicles, observed_vehicles):
        for vehicle, visibility in zip(vehicles, observed_vehicles):
            if vehicle not in self.knowledges:
                self.knowledges[vehicle] = []
                self.delays[vehicle] = 0
            self.knowledges[vehicle].append(int(visibility))
            if visibility == 0:
                self.delays[vehicle] += 1
            else:
                self.delays[vehicle] = 0
    
    def merge_knowledges(self, new_knowledges, new_delays):
        for vehicle, visibility in new_knowledges.items():
            if vehicle not in self.knowledges:
                self.knowledges[vehicle] = copy.deepcopy(visibility)
                self.delays[vehicle] = new_delays[vehicle]
            else:
                assert len(self.knowledges[vehicle]) == len(visibility)
                self.knowledges[vehicle] = [a | b for a, b in zip(self.knowledges[vehicle], visibility)]
                self.delays[vehicle] = min(self.delays[vehicle], new_delays[vehicle])
        return copy.deepcopy(self.knowledges), copy.deepcopy(self.delays)

    def get_knowledges(self):
        return copy.deepcopy(self.knowledges)
    
    def get_delays(self):
        return copy.deepcopy(self.delays)
    
    def evaluate_knowledge(self):
        missing = 0
        total = 0
        delay = 0
        for vehicle, visibility in self.knowledges.items():
            missing += sum(visibility)
            total += len(visibility)
            delay += self.delays[vehicle]
        return missing / total, delay / total

class Beacon:
    def __init__(self, trace_hidden):
        self.trace_hidden = copy.deepcopy(trace_hidden)
    
    def update(self, trace_hidden):
        self.trace_hidden = copy.deepcopy(trace_hidden)
