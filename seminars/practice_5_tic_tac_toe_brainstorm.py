"""
Programming 2022
Seminar 5

Brainstorm from the lecture on designing a TicTacToe game
"""


class Move:
    def __init__(self, row, col, label):
        self.row = row
        self.col = col
        self.label = label


class Player:
    def __init__(self, label):
        self.label = label

    def make_move(self):
        return Move(row=2, col=2, label=self.label)


class Game:
    def __init__(self, players, board_size):
        self._moves = []
        self._players = players
        self._win_states = []
        self._current_player_idx = 0
        self._size = board_size

    def _check_move(self, move: Move):
        for i in self.moves:
            if i.row == move.row and i.col == move.col:
                return False
        if move.row >= self.size or move.col >= self.size:
            return False
        return True

    def _next_player(self):
        if self.current_player_idx == 0:
            self.current_player_idx = 1
        else:
            self.current_player_idx = 0

    def play_game(self):
        while True:
            move = self.players[self.current_player_idx].make_move()

            if self.check_move(move):
                self.moves.append(move)

            self.next_player()


def main():
    player1 = Player("X")
    player2 = Player("O")

    game = Game(players=(player1, player2), board_size=3)
    game.play_game()


if __name__ == "__main__":
    main()
