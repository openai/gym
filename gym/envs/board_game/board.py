'''
Author: P.V.Aravind Reddy
Board state is a list consisting of 0's ,1's , 2's 
where 0 is blank
1 for player1 move i.e X
2 for player2 move i.e O
'''


class Board:
    def __init__(self):
        self.state = [0] * 9

    def reset(self):
        # initializing board with 0's
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
        # check if there are three 1's or three 2's in a row or column or diagonal
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
        # check if the game is draw
        if 0 not in self.state:
            return True

    def gameOver(self):
        # check if game is over
        if self.rowcolumn():
            return False
        elif self.full_posit():
            return False
        else:
            return True
