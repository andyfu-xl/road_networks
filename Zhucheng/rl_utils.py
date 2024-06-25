import copy

 
class Knowledges:
    def __init__(self, knowledges={}):
        self.knowledges = knowledges
    
    def add_observations(self, observed_vehicles):
        for vehicle, visibility in observed_vehicles.items():
            if vehicle not in self.knowledges:
                self.knowledges[vehicle] = []
            self.knowledges[vehicle].append(visibility)
    
    def merge_knowledges(self, new_knowledges):
        for vehicle, visibility in new_knowledges.items():
            if vehicle not in self.knowledges:
                self.knowledges[vehicle] = visibility.deepcopy()
            else:
                longer_length = max(len(self.knowledges[vehicle]), len(visibility))
                # fill the former knowledge with 0s
                self.knowledges[vehicle] = [0] * (longer_length - len(self.knowledges[vehicle])) + self.knowledges[vehicle]
                # fill the new knowledge with 0s
                visibility = [0] * (longer_length - len(visibility)) + visibility
                # merge the two by or operation
                self.knowledges[vehicle] = [a | b for a, b in zip(self.knowledges[vehicle], visibility)]
        return self.knowledges.deepcopy()

    def get_knowledges(self):
        return self.knowledges.deepcopy()