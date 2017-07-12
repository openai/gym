class Board:
    def __init__(self):
        self.state = [0] * 9

    def reset(self):
        self.state = [0] * 9

    def getvalidmoves(self):
        Valid = []
        for i in range(9):
            if self.state[i] == 0:
                Valid.append(i)
        return Valid

    def move(self, position, label):
        self.state[position] = label


    def rowcolumn(self):
        if ''.join(map(str, self.state[0:3])) == '111' or ''.join(map(str, self.state[0:3])) == '222':
            return True
        elif ''.join(map(str, self.state[3:6])) == '111' or ''.join(map(str, self.state[3:6])) == '222':
            return True
        elif ''.join(map(str, self.state[6:9])) == '111' or ''.join(map(str, self.state[6:9])) == '222':
            return True
        elif ''.join(map(str, (self.state[x] for x in [0, 3, 6]))) == '111' or ''.join(
                map(str, (self.state[x] for x in [0, 3, 6]))) == '222':
            return True
        elif ''.join(map(str, (self.state[x] for x in [1, 4, 7]))) == '111' or ''.join(
                map(str, (self.state[x] for x in [1, 4, 7]))) == '222':
            return True
        elif ''.join(map(str, (self.state[x] for x in [2, 5, 8]))) == '111' or ''.join(
                map(str, (self.state[x] for x in [2, 5, 8]))) == '222':
            return True
        elif ''.join(map(str, (self.state[x] for x in [0, 4, 8]))) == '111' or ''.join(
                map(str, (self.state[x] for x in [0, 4, 8]))) == '222':
            return True
        elif ''.join(map(str, (self.state[x] for x in [2, 4, 6]))) == '111' or ''.join(
                map(str, (self.state[x] for x in [2, 4, 6]))) == '222':
            return True

    def full_posit(self):
        if 0 not in self.state:
            return True

    def gameOver(self):
        if self.rowcolumn():
            return False
        elif self.full_posit():
            return False
        else:
            return True
