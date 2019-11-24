import random
from IPython.display import Markdown, display

random.seed(9001)

class Agent:

    def __init__(self, ID):
        self.vision = random.randrange(1, 6)
        self.metabolic_rate = random.randrange(1, 4)
        self.age = random.randrange(0, 100)
        self.s_wealth = random.randrange(5, 25)
        self.sugar_collected = 0
        self.ID = ID
        self.visual = "\033[1mX\033[0m"

    def get_vision(self):
        return self.vision

    def get_visual(self):
        return self.visual

    def get_metabolic_rate(self):
        return self.metabolic_rate

    def get_age(self):
        return self.age

    def get_s_wealth(self):
        return self.s_wealth

    def calculate_s_wealth(self):
        self.s_wealth = self.sugar_collected - self.metabolic_rate

    def collect_sugar(self, environment_cell_sugar):
        self.sugar_collected = self.sugar_collected + environment_cell_sugar

    def get_ID(self):
        return self.ID
