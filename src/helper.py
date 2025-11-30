import numpy as np


def pollModel(cards: dict[str, str], cardFile: str) -> str:  # (ex. model outputs "52" and function returns "King of Hearts")
    # all model logic here
    # output = net(cardFile)
    # _, predicted = torch.max(output, 1)
    # return cards[classes[predicted[0]]]
    pass


# ====================== PROCESS CARDS ========================#


def processPlayerCards(cards: dict[str, str], playerCards: list[int], fileCount: int) -> tuple[int, int]:
    # keep track of if the player has an ace, and what the running count of these cards are
    aceFlag = 0
    runningCount = 0

    # start of the game, player draws 2 cards
    if fileCount == 2:
        selection = input("Please provide the filename(s) of the card(s) the player was dealt: ")
        selection = selection.split()
        # selection = card6.png card7.png
        # selection.split() = ["card6.png", "card7.png"]
        card1 = "Jack of Diamonds"  # pollModel(cards, selection[0])
        card1 = cardValue(card1)
        runningCount += addRunningCount(card1)
        playerCards.append(card1)

        card2 = "7 of Clubs"  # pollModel(cards, selection[1])
        card2 = cardValue(card2)
        runningCount += addRunningCount(card2)
        playerCards.append(card2)

        if card1 == 11 or card2 == 11:
            aceFlag = 1
    # player draws 1 card the rest of the game
    else:
        selection = input("Which card did the player draw next?: ")
        selection = selection.split()

        card0 = "2 of Spades"  # pollModel(cards, selection[0])
        card0 = cardValue(card0)
        runningCount += addRunningCount(card0)
        playerCards.append(card0)
        if card0 == 11:
            aceFlag = 1

    return aceFlag, runningCount


def processDealerCards(cards: dict[str, str], dealerCards: list[int], fileCount: int) -> int:
    runningCount = 0

    if fileCount == 2:
        selection = input("Please provide the filename(s) of the card(s) the dealer was dealt (dealer upcard first): ")
        selection = selection.split()

        card3 = "8 of Diamonds"  # pollModel(cards, selection[0])
        card3 = cardValue(card3)
        # don't add the dealer's upcard to the running count yet
        # after the player ends their turn, the dealer will draw until stay or bust
        # the running count from this function will get added to the overall count at the end of the hand
        runningCount += addRunningCount(card3)
        dealerCards.append(card3)

        card4 = "3 of Hearts"  # pollModel(cards, selection[1])
        card4 = cardValue(card4)
        runningCount += addRunningCount(card4)
        dealerCards.append(card4)
    else:
        selection = input("Which card did the dealer draw next?: ")
        selection = selection.split()

        card5 = "King of Hearts"  # pollModel(cards, selection[0])
        card5 = cardValue(card5)
        runningCount += addRunningCount(card5)
        dealerCards.append(card5)

    return runningCount


# convert the card description into a numerical value
def cardValue(card: str) -> int:
    cardSplit = card.split()
    # King of Diamonds
    # [King, of, Diamonds]
    if cardSplit[0] == "Ace":
        return 11
    elif cardSplit[0] in ["Jack", "Queen", "King"]:
        return 10
    else:
        return int(cardSplit[0])


# convert the card into a number specified by hi-low card counting
# used card counting tutorial (Reference #1)
def addRunningCount(value: int) -> int:
    if 2 <= value <= 6:
        return 1
    elif 7 <= value <= 9:
        return 0
    else:
        return -1


# ===================== EVALUATE OPTIONS =========================#


def evaluateOptions(aceFlag: int, runningCount: int, numDecks: int, playerCards: list[int], dealerCards: list[int]) -> int:
    move = 0
    totalCount = runningCount / numDecks

    # Used Q-learning enhanced probabilities (Reference #2)
    # Used extended probabilities, since Q-learning data stops at sum 13 and we need sum 8-12 (Reference #3)
    hardProbs = np.array(
        [
            # columns (player score, dealer upcards): score 2 3 4 5 6 7 8 9 10 A
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
    playerSum = sum(playerCards)
    dealerUp = dealerCards[0]
    tableX, tableY = findCoordinate(playerSum, dealerUp)
    if aceFlag == 0:
        move = hardProbs[tableY, tableX]
    else:
        move = softProbs[tableY, tableX]

    deviate = deviationRules(aceFlag, totalCount, playerSum, dealerUp)
    if deviate:
        move = not move

    return move


# the writer of the book that contains data from simulations: https://www.casinocenter.com/master-class-the-hi-lo-card-counting-system/
# the condensed versions of those rules down to the 'illustrious 18': https://wizardofodds.com/games/blackjack/card-counting/high-low/
# visual representation of the illustrious 18: https://www.blackjackapprenticeship.com/wp-content/uploads/2019/07/BJA_S17.pdf
# Note: the 18th deviation dictates when the player should take insurance; our implementation won't address this


def findCoordinate(playerSum: int, dealerUp: int) -> tuple[int, int]:
    x = convertDealer(dealerUp)
    y = convertPlayer(playerSum)
    return x, y


def convertDealer(dealerUp: int) -> int:
    return dealerUp - 1


def convertPlayer(playerSum: int) -> int:
    return playerSum - 2


def deviationRules(aceFlag: int, totalCount: float, playerSum: int, dealerUp: int) -> bool:
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
