class Vehicle:
    def __init__(self):
        self.sent = {}
        self.received = 0

    def step(self):
        self.received = 0
        for vehicle, lag in self.sent.items():
            self.sent[vehicle] += 1
        self.sent = {vehicle: lag for vehicle, lag in self.sent.items() if lag < 10}
    
    def receive(self):
        self.received += 1

    # lag: the time since the last communication with a neighbor
    def select(self, neighbors):
        max_lag = 0
        selected = None
        for neighbor in neighbors:
            if neighbor not in self.sent:
                selected = neighbor
                break
            if self.sent[neighbor] > max_lag:
                max_lag = self.sent[neighbor]
                selected = neighbor
        return selected
    
    def send(self, selected_neighbor):
        self.sent[selected_neighbor] = 0