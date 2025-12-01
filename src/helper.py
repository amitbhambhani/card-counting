import numpy as np

from script import predict


def pollModel(cards: dict[str, str], cardFile: str) -> str:  # (ex. model outputs "52" and function returns "King of Hearts")
    # Call the model and retrieve the class associated with cardFile
    classNum = str(predict(cardFile))
    return cards[classNum]


# ====================== PROCESS CARDS ========================#


def processPlayerCards(cards: dict[str, str], playerCards: list[int], fileCount: int) -> tuple[int, int]:
    # Keep track of the running count and if the player has an Ace
    aceFlag = 0
    runningCount = 0
    # Cards drawn used for printing
    chosenCards = []

    # The player draws 2 cards at the start of the game
    if fileCount == 2:    
        # Input example: "card1.jpg card2.jpg"
        selection = input("Please provide the filename(s) of the card(s) the player was dealt: ")
        selection = selection.split()

        # Ask the model what the first card is
        card1 = pollModel(cards, selection[0])
        chosenCards.append(card1)
        # Get the card's numerical value
        card1 = cardValue(card1)
        # Update the running count based on this card
        runningCount += addRunningCount(card1)
        # Add the card to the player's hand
        playerCards.append(card1)

        # Second card
        card2 = pollModel(cards, selection[1])
        chosenCards.append(card2)
        card2 = cardValue(card2)
        runningCount += addRunningCount(card2)
        playerCards.append(card2)

        # Does the player have an Ace?
        if card1 == 11 or card2 == 11:
            aceFlag = 1

        print(f"The player drew: {chosenCards[0]} and {chosenCards[1]}")

    # The player draws 1 card for every other turn
    else:
        # Example input: "card3.jpg"
        selection = input("Which card did the player draw next?: ")

        card0 = pollModel(cards, selection)
        chosenCards.append(card0)
        card0 = cardValue(card0)
        runningCount += addRunningCount(card0)
        playerCards.append(card0)

        # Is this card an Ace?
        if card0 == 11:
            aceFlag = 1

        print(f"The player drew: {chosenCards[0]}")

    return aceFlag, runningCount


def processDealerCards(cards: dict[str, str], dealerCards: list[int], fileCount: int) -> int:
    # Only concerned with the running count when it comes to the dealer's cards
    runningCount = 0
    # Cards drawn used for printing
    chosenCards = []
    # The dealer draws 2 cards at the start of the game
    if fileCount == 2:
        selection = input("Please provide the filename(s) of the card(s) the dealer was dealt (dealer upcard first): ")
        selection = selection.split()

        # The dealer's upcard
        card3 = pollModel(cards, selection[0])
        chosenCards.append(card3)
        card3 = cardValue(card3)
        runningCount += addRunningCount(card3)
        dealerCards.append(card3)

        # Second card
        card4 = pollModel(cards, selection[1])
        chosenCards.append(card4)
        card4 = cardValue(card4)
        runningCount += addRunningCount(card4)
        dealerCards.append(card4)

        print(f"The dealer drew: {chosenCards[0]} and {chosenCards[1]}")
    else:
        selection = input("Which card did the dealer draw next?: ")

        card5 = pollModel(cards, selection)
        chosenCards.append(card5)
        card5 = cardValue(card5)
        runningCount += addRunningCount(card5)
        dealerCards.append(card5)

        print(f"The dealer drew: {chosenCards[0]}")

    return runningCount


# Get a card's numerical value
def cardValue(card: str) -> int:
    # ex. King of Diamonds -> ['King', 'of', 'Diamonds']
    cardSplit = card.split()
    if cardSplit[0] == "Ace":
        return 11
    elif cardSplit[0] in ["Jack", "Queen", "King"]:
        return 10
    else:
        return int(cardSplit[0])


# Compute the hi-low value for a given card
# Used the following card-counting tutorial: (Reference #1)
def addRunningCount(value: int) -> int:
    if 2 <= value <= 6:
        return 1
    elif 7 <= value <= 9:
        return 0
    else:
        return -1


# ===================== EVALUATE OPTIONS =========================#


def evaluateOptions(aceFlag: int, runningCount: int, numDecks: int, playerCards: list[int], dealerCards: list[int]) -> int:
    # Technically, the count used for deviations uses running count / number of decks
    move = 0
    totalCount = runningCount / numDecks

    # Used Q-learning enhanced probabilities for sum 13-21: (Reference #2)
    # Used basic strategy probabilities for sum 8-12: (Reference #3)
    # Assumed the player should hit for all cases within sum 2-7
    hardProbs = np.array(
        [
            # Each row represents the player's sum
            # Each column represents the dealer's upcard
          # Sum 2  3  4  5  6  7  8  9  10 A
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [12, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
            [13, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],
            [14, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [15, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [16, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    softProbs = np.array(
        [
            # Same format as hard probabilities
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [14, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [17, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [18, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [19, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    # Calculate the player's current sum and get the dealer's upcard
    playerSum = sum(playerCards)
    dealerUp = dealerCards[0]
    # Based on those 2 values, calculate the table indicies
    tableX, tableY = findCoordinate(playerSum, dealerUp)
    # If the player has an Ace, they should choose from the soft totals
    if aceFlag == 0:
        move = hardProbs[tableY, tableX]
    else:
        move = softProbs[tableY, tableX]

    # Based on the running count, the player may need to go against the basic strategy recommendation
    deviate = deviationRules(aceFlag, totalCount, playerSum, dealerUp)
    if deviate:
        move = not move

    return move


# The following 3 functions are used to select a value from one of the tables
def findCoordinate(playerSum: int, dealerUp: int) -> tuple[int, int]:
    x = convertDealer(dealerUp)
    y = convertPlayer(playerSum)
    return x, y


def convertDealer(dealerUp: int) -> int:
    return dealerUp - 1


def convertPlayer(playerSum: int) -> int:
    return playerSum - 2


# Don Schlesinger wrote a book about the programming simulations he used 
    # to create the rules for when one should deviate from basic strategy (Reference #4)
# The most influential of these simulations were condensed into the "Illustrious 18" (Reference #5)
# I used a diagram to help visualize these 18 deviations and build the softProbs and hardProbs tables (Reference #6)
    # Our program doesn't incorporate insurance, so we excluded the 18th deviation from our implementation

def deviationRules(aceFlag: int, totalCount: float, playerSum: int, dealerUp: int) -> bool:
    # Deviations for the hard totals
    if aceFlag == 0:
        if playerSum == 16 and dealerUp == 10 and totalCount >= 0:
            return True
        elif playerSum == 16 and dealerUp == 9 and totalCount >= 4:
            return True
        elif playerSum == 15 and dealerUp == 10 and totalCount >= 4:
            return True
        elif playerSum == 13 and dealerUp == 2 and totalCount <= -1:
            return True
        elif playerSum == 12 and dealerUp == 4 and totalCount <= 0:
            return True
        elif playerSum == 12 and dealerUp == 3 and totalCount >= 2:
            return True
        elif playerSum == 12 and dealerUp == 2 and totalCount >= 3:
            return True
        elif playerSum == 11 and dealerUp == 11 and totalCount >= 1:
            return True
        elif playerSum == 10 and dealerUp == 11 and totalCount >= 4:
            return True
        elif playerSum == 10 and dealerUp == 10 and totalCount >= 4:
            return True
        elif playerSum == 9 and dealerUp == 7 and totalCount >= 3:
            return True
        elif playerSum == 9 and dealerUp == 2 and totalCount >= 1:
            return True
        elif playerSum == 8 and dealerUp == 6 and totalCount >= 2:
            return True
    # Deviations for the soft totals
    else:
        if playerSum == 19 and dealerUp == 6 and totalCount >= 1:
            return True
        elif playerSum == 19 and dealerUp == 5 and totalCount >= 1:
            return True
        elif playerSum == 19 and dealerUp == 4 and totalCount >= 3:
            return True
        elif playerSum == 17 and dealerUp == 2 and totalCount >= 1:
            return True
    return False


# ===================== REFERENCES ====================#
# Code/logic derived from references will be marked with (Reference #)
#
# 1. Card counting tutorial - https://www.youtube.com/watch?app=desktop&v=KAyA_XTHi-g
# 2. Q-Learning optimization of basic strategy (13-21) - https://web.stanford.edu/class/aa228/reports/2020/final17.pdf
# 3. Extended (8-12) basic strategy probabilities - https://www.blackjackapprenticeship.com/wp-content/uploads/2019/07/BJA_S17.pdf
# 4. Don Schlesinger's blackjack simulations - https://www.casinocenter.com/master-class-the-hi-lo-card-counting-system/
# 5. The Illustrious 18 - https://wizardofodds.com/games/blackjack/card-counting/high-low/
# 6. Diagram of the 18 - https://www.blackjackapprenticeship.com/wp-content/uploads/2019/07/BJA_S17.pdf
