import time

from helper import (
    addRunningCount,
    evaluateOptions,
    processDealerCards,
    processPlayerCards,
)

print("Welcome to 3CD: Blackjack Edition")
time.sleep(1.5)

# Assume 1 player
# playerNum = int(input("How many players are at the table?: "))

# For now, assume 1 deck
numDecks = int(input("How many decks of cards will be used?: "))
numDecks = 1

hands = 1

# Decode model output
cards = {
    "1": "Ace of Spades",
    "2": "Ace of Clubs",
    "3": "Ace of Diamonds",
    "4": "Ace of Hearts",
    "5": "2 of Spades",
    "6": "2 of Clubs",
    "7": "2 of Diamonds",
    "8": "2 of Hearts",
    "9": "3 of Spades",
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

# This represents the player's hi-low count of all cards they can see at any given time
    # 2-6 = +1
    # 7-9 = 0
    # T (10, J, Q, K) and A = -1
runningCount = 0

def gameLoop(cards: dict[str, str], numDecks: int, hands: int, runningCount: int) -> int:
    print(f"Hand {hands}")
    time.sleep(1.5)
    
    # Cards in the player's and dealer's hands
    dealerCards: list[int] = []
    playerCards: list[int] = []

    # 0 if player doesn't have an Ace, 1 if they do
    aceFlag = 0

    # Flag if the hand has ended
    end = False

    # Track when the player or dealer chooses to stay/busts
    playerStay = 0
    playerBust = 0
    dealerStay = 0
    dealerBust = 0

    # Turn 1: deal 2 cards; turn 2+: deal 1 card
    turn = 1
    # Keep track of individual turns for printing
    playerTurn = 2
    dealerTurn = 2

    while True:
        if turn == 1:
            # Add the player's cards to their hand
            # Assess the hi-low count of their cards
            # See if any of the player's cards are an Ace
            playerAce, playerCount = processPlayerCards(cards, playerCards, 2)
            # Add the dealer's cards to their hand
            # Assess the hi-low count of their cards
            dealerCount = processDealerCards(cards, dealerCards, 2)

            # Update the running count with the player's cards
            runningCount += playerCount
            # Update the running count with the dealer's upcard
            runningCount += addRunningCount(dealerCards[0])

            # An Ace determines whether we use hard or soft totals
            if playerAce == 1:
                aceFlag = 1

            # Account for the rare case where the player is dealt 2 Aces 
            if playerCards[0] == 11 and playerCards[1] == 11:
                playerCards[0] = 1

            # Select the most accurate move based on the player's hand sum and dealer's upcard
            playerMove = evaluateOptions(aceFlag, runningCount, numDecks, playerCards, dealerCards)
            # Display the player's move
            if playerMove == 1:
                print(f"Player's turn {turn}: hit")
                time.sleep(1.5)
            else:
                playerStay = 1
                print(f"Player's turn {turn}: stay")
                time.sleep(1.5)

        else:
            while playerStay == 0 and playerBust == 0:
                # The player chose to hit
                # They can only be dealt 1 card from now on
                playerAce, playerCount = processPlayerCards(cards, playerCards, 1)

                # Add the new card to the running count
                runningCount += playerCount

                # Determine if the new card is an Ace
                if playerAce == 1:
                    aceFlag = 1

                # Check if this new card made the player bust
                if sum(playerCards) > 21:
                    # Without an Ace, the player busts
                    if aceFlag == 0:
                        print("The player busts! Dealer wins.")
                        time.sleep(1.5)
                        playerBust = 1
                        # Assume the dealer shows their hole card, now that the hand is over
                        # The player would add that card to the running count
                        runningCount += addRunningCount(dealerCards[1])
                        # Mark the end of the hand
                        end = True
                        # The dealer won't draw cards in this scenario
                        break
                    # With an Ace, the player is saved
                    else:
                        found = False
                        for i in range(len(playerCards)):
                            if found == False and playerCards[i] == 11:
                                # A player might have multiple Aces. Did we find one of them?
                                found = True
                                # This Ace needs to switch from 11 to 1
                                playerCards[i] = 1
                                # Assume the player has no more Aces now...
                                aceFlag = 0
                            # ... but keep scanning their hand to see if another Ace exists
                            if found == True and playerCards[i] == 11:
                                # The player does have another Ace, so reactivate the flag
                                aceFlag = 1

                # Select the most accurate move based on the player's hand sum and dealer's upcard
                playerMove = evaluateOptions(aceFlag, runningCount, numDecks, playerCards, dealerCards)
                # Display the player's move
                if playerMove == 1:
                    print(f"Player's turn {playerTurn}: hit")
                    time.sleep(1.5)
                    playerTurn += 1
                else:
                    playerStay = 1
                    print(f"Player's turn {playerTurn}: stay")
                    time.sleep(1.5)

            while dealerStay == 0 and dealerBust == 0:
                # If the player busts, the dealer won't need to draw cards
                if end:
                    break
                # The dealer takes their first turn
                if turn == 2:
                    # The player can now see the dealer's hole card
                    runningCount += addRunningCount(dealerCards[1])
                    
                    # Many casinos employ the "soft 17" rule (Reference #1)
                    # Dealers will hit if they're below a sum of 17, and stay otherwise
                        # However, if the player has a sum greater than the dealer, and 
                        # the dealer's sum is at least 17, the dealer will hit again
                    if sum(dealerCards) >= 17:
                        if sum(playerCards) > sum(dealerCards):
                            print(f"Dealer's turn 1: hit")
                            time.sleep(1.5)
                            turn += 1
                            continue
                        print(f"Dealer's turn 1: stay")
                        time.sleep(1.5)
                        dealerStay = 1
                        # Mark the end of the hand
                        end = True 
                        break
                    else:
                        print(f"Dealer's turn 1: hit")
                        time.sleep(1.5)
                        turn += 1
                        continue

                # The dealer chose to hit
                dealerCount = processDealerCards(cards, dealerCards, 1)

                # Add the new card to the running count
                runningCount += dealerCount

                # Check if this new card made the dealer bust
                if sum(dealerCards) > 21:
                    found = False
                    for i in range(len(dealerCards)):
                        # Does the dealer have an Ace?
                        if dealerCards[i] == 11:
                            found = True
                            # Switch this Ace from 11 to 1
                            playerCards[i] = 1
                    # If the dealer doesn't have an Ace, they bust
                    if found == False:
                        print("The dealer busts! Player wins.")
                        time.sleep(1.5)
                        dealerBust = 1
                        # Mark the end of the hand
                        end = True
                        break

                # Determine the dealer's next move
                if sum(dealerCards) >= 17:
                    if sum(playerCards) > sum(dealerCards):
                            print(f"Dealer's turn {dealerTurn}: hit")
                            time.sleep(1.5)
                            turn += 1
                            continue
                    print(f"Dealer's turn {dealerTurn}: stay")
                    time.sleep(1.5)
                    dealerStay = 1
                    # Mark the end of the hand
                    end = True
                    break
                else:
                    print(f"Dealer's turn {dealerTurn}: hit")
                    time.sleep(1.5)
                    dealerTurn += 1

        # The end of the current hand
        if end:
            # Choose to start the next hand or end the game
            again = input("Next hand (1) | End (0): ")
            if again == "1":
                hands += 1
                gameLoop(cards, numDecks, hands, runningCount)
                return 0
            else:
                print("Thank you for choosing 3CD")
                time.sleep(1.5)
                return 0
        turn += 1

# Start
gameLoop(cards, numDecks, hands, runningCount)


# ===================== REFERENCES ====================#
# Code/logic derived from references will be marked with (Reference #)
#
# 1. Soft 17 rule for blackjack dealers - https://www.cachecreek.com/blackjack-odds#:~:text=Dealer%20Stands%20on%20Soft%2017,win%20in%20the%20long%20run.
