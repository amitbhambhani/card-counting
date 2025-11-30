# Casino Card Counting Detection - 3CD

A program that can be split into 2 main phases:

1. Image detection via a custom CNN built using PyTorch; images are captured and uploaded to the model.
2. Player/dealer move selection based on basic strategy, optimized by Q-learning, in addition to deviations resulting from hi-low card counting.

## Installation

### Clone the repository and run

```bash
git clone https://github.com/amitbhambhani/card-counting.git
cd src/
python3 blackjack.py
```

## Usage

### General

After the introduction, enter the number of decks used in the game

```
Welcome to 3CD
The game is Blackjack
How many decks of cards are in use?: 1
```

The game begins with the player being dealt 2 cards

```
Hand 1
Please provide the filename(s) of the card(s) the player was dealt: c1.jpg c2.jpg
```

The dealer is also dealt 2 cards, but the player can only see one of them. This is the dealer's face-up card (aka upcard)

```
Please provide the filename(s) of the card(s) the dealer was dealt (dealer upcard first): c3.jpg c4.jpg
```

The program then selects the best move for the player and dealer based on the supplied cards

```
Player's turn 1: stay
Dealer's turn 1: hit
```

The program then asks for additional cards from the player if they chose to hit. If they stay, the dealer is prompted for additional cards (given the dealer chose to hit)

```
Which card did the dealer draw next?: c5.jpg
```

Moves are suggested based on real-time analysis of the current card

```
Dealer's turn 2: stay
```

Once the player and dealer have chosen to stay, or either one busts, the hand ends. At this point, you can choose to continue with the next hand or exit

```
Next hand (1) | End (0): 1
```

Choosing to continue the game maintains the running count from the previous hand

```
Hand 2
Please provide the filename(s) of the card(s) the player was dealt: c6.jpg c7.jpg
```

### Bust (Player)

### Bust (Dealer)
