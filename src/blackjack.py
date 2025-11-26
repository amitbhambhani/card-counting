import time

from helper import evaluateOptions, processDealerCards, processPlayerCards


def gameLoop() -> int:
    # decode model output
    cards = {
        "01": "Ace of Spades",
        "02": "Ace of Clubs",
        "03": "Ace of Diamonds",
        "04": "Ace of Hearts",
        "05": "2 of Spades",
        "06": "2 of Clubs",
        "07": "2 of Diamonds",
        "08": "2 of Hearts",
        "09": "3 of Spades",
        "10": "3 of Clubs",
        "11": "3 of Diamonds",
        "12": "3 of Hearts",
        "13": "4 of Spades",
        "14": "4 of Clubs",
        "15": "4 of Diamonds",
        "16": "4 of Hearts",
        "17": "5 of Spades",
        "18": "5 of Clubs",
        "19": "5 of Diamonds",
        "20": "5 of Hearts",
        "21": "6 of Spades",
        "22": "6 of Clubs",
        "23": "6 of Diamonds",
        "24": "6 of Hearts",
        "25": "7 of Spades",
        "26": "7 of Clubs",
        "27": "7 of Diamonds",
        "28": "7 of Hearts",
        "29": "8 of Spades",
        "30": "8 of Clubs",
        "31": "8 of Diamonds",
        "32": "8 of Hearts",
        "33": "9 of Spades",
        "34": "9 of Clubs",
        "35": "9 of Diamonds",
        "36": "9 of Hearts",
        "37": "10 of Spades",
        "38": "10 of Clubs",
        "39": "10 of Diamonds",
        "40": "10 of Hearts",
        "41": "Jack of Spades",
        "42": "Jack of Clubs",
        "43": "Jack of Diamonds",
        "44": "Jack of Hearts",
        "45": "Queen of Spades",
        "46": "Queen of Clubs",
        "47": "Queen of Diamonds",
        "48": "Queen of Hearts",
        "49": "King of Spades",
        "50": "King of Clubs",
        "51": "King of Diamonds",
        "52": "King of Hearts",
    }

    # cards that the player has drawn, and all top cards the dealer has drawn
    dealerCards = []
    playerCards = []

    # number of decks used (assume 4 for now)
    numDecks = 0

    # 2-6 = -1, 7-9 = 0, T (10, J, Q, K) and A = +1
    runningCount = 0

    # Is there an ace on the board (player hand or dealer up card)?
    aceFlag = 0

    print("Welcome to 3CD")
    time.sleep(2)
    print("The game is Blackjack")
    time.sleep(2)
    playerNum = int(input("How many players are at the table?: "))
    numDecks = int(input("How many decks of cards will be used?: "))
    numDecks = 1

    turn = 0
    while True:
        if turn == 0:
            playerAce, playerCount = processPlayerCards(cards, playerCards, 2)
            dealerAce, dealerCount = processDealerCards(cards, dealerCards, 2)

            runningCount += playerCount
            runningCount += dealerCount

            if playerAce == 1 or dealerAce == 1:
                aceFlag = 1

            move = evaluateOptions(aceFlag, runningCount, numDecks, playerCards, dealerCards)
            if move == 1:
                print(f"Turn {turn}: hit")
            else:
                print(f"Turn {turn}: stay")

            if sum(playerCards) > 21 or sum(dealerCards) > 21:
                print("The round is over")
                time.sleep(0.5)
                again = input("Next round (1) | End (0): ")
                if again == 1:
                    turn = 0
                else:
                    return 0
        else:
            playerAce = processPlayerCards(cards, playerCards, 1)
            dealerAce = processDealerCards(cards, dealerCards, 1)
        turn += 1


gameLoop()
