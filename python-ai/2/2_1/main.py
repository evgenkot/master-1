import random


class Warrior:
    def __init__(self, name):
        self.name = name
        self.health = 100
        print(f"Warrior {self.name} created with {self.health} hp!")

    def attack(self, opponent: 'Warrior'):
        opponent.health -= 20
        print(f"{self.name} attack {opponent.name}!")

    def is_alive(self) -> bool:
        return self.health > 0

    def __del__(self):
        print(f"Warrior {self.name} left us...")


def battle(warrior1: 'Warrior', warrior2: 'Warrior'):
    while warrior1.is_alive() and warrior2.is_alive():
        attacker = random.choice([warrior1, warrior2])
        if attacker == warrior1:
            warrior1.attack(warrior2)
        else:
            warrior2.attack(warrior1)

        print(f"{warrior1.name} has {warrior1.health} hp left.")
        print(f"{warrior2.name} has {warrior2.health} hp left.")
        print()  # Print a blank line for better readability

    if warrior1.is_alive():
        print(f"{warrior1.name} wins!")
    else:
        print(f"{warrior2.name} wins!")


warrior1 = Warrior("Gladiator")
warrior2 = Warrior("Hero")

battle(warrior1, warrior2)
