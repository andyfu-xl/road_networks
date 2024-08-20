import copy


class Knowledges:
    def __init__(self):
        self.knowledges = {}
        self.delays = {}
    
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
        prev_missing, prev_delay, _, _ = self.evaluate_knowledge()
        for vehicle, visibility in new_knowledges.items():
            if vehicle not in self.knowledges:
                self.knowledges[vehicle] = copy.deepcopy(visibility)
                self.delays[vehicle] = new_delays[vehicle]
            else:
                for i in range(1, len(self.knowledges[vehicle])+1):
                    if i > len(visibility):
                        break
                    self.knowledges[vehicle][-i] = visibility[-i] | self.knowledges[vehicle][-i]
                self.delays[vehicle] = min(self.delays[vehicle], new_delays[vehicle])
        new_missing, new_delay, _, _ = self.evaluate_knowledge()
        return copy.deepcopy(self.knowledges), copy.deepcopy(self.delays), prev_missing - new_missing, prev_delay - new_delay

    def get_knowledges(self):
        return copy.deepcopy(self.knowledges)
    
    def get_delays(self):
        return copy.deepcopy(self.delays)
    
    def evaluate_knowledge(self, large_delay_threshold=10):
        observed = 0
        delay = 0
        large_delay = 0
        num_vehicles = len(self.knowledges)
        whole_missing = 0
        total = 0
        for vehicle, visibility in self.knowledges.items():
            observed += sum(visibility) 
            total += len(visibility)
            delay += self.delays[vehicle]
            if self.delays[vehicle] >= large_delay_threshold or sum(visibility) <= 3:
                large_delay += 1
            if sum(visibility) == 0:
                whole_missing += len(visibility)
        return 1-(observed / total), delay / num_vehicles, large_delay / num_vehicles, whole_missing / total
