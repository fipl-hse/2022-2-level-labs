"""
Programming 2022
Seminar 5

Brainstorm from the lecture on designing a TicTacToe game
"""
from typing import Literal

MarkerType = Literal["X", "O"]


class Move:
    def __init__(self, row: int, col: int, label: MarkerType) -> None:
        self.row = row
        self.col = col
        self.label = label


class Player:
    def __init__(self, label: MarkerType) -> None:
        self.label = label

    def make_move(self) -> Move:
        return Move(row=2, col=2, label=self.label)


class Game:
    def __init__(self, players: tuple[Player, ...], board_size: int) -> None:
        self._moves = []
        self._players = players
        self._win_states = []
        self._current_player_idx = 0
        self._size = board_size

    def _check_move(self, move: Move) -> bool:
        for i in self._moves:
            if i.row == move.row and i.col == move.col:
                return False
        if move.row >= self._size or move.col >= self._size:
            return False
        return True

    def _next_player(self) -> None:
        if self._current_player_idx == 0:
            self._current_player_idx = 1
        else:
            self._current_player_idx = 0

    def play_game(self) -> None:
        # num_steps should be eliminated further, here just to make sure program does not
        # run forever
        num_steps = 0

        while num_steps < 3:  # while True:
            print(f"Current move no. {num_steps+1}: Player {self._current_player_idx} decides...")
            move = self._players[self._current_player_idx].make_move()

            if self._check_move(move):
                print("Move accepted")
                self._moves.append(move)
            else:
                print("Move rejected")

            self._next_player()

            num_steps += 1


def main() -> None:
    player1 = Player("X")
    player2 = Player("O")

    game = Game(players=(player1, player2), board_size=3)
    game.play_game()


if __name__ == "__main__":
    main()
