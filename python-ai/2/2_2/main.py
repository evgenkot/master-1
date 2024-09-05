class Snow:
    def __init__(self, flakes: int) -> None:
        self.flakes = flakes

    def __add__(self, other: 'Snow') -> 'Snow':
        return Snow(self.flakes + other.flakes)

    def __sub__(self, other: 'Snow') -> 'Snow':
        return Snow(self.flakes - other.flakes)

    def __mul__(self, n: int) -> 'Snow':
        return Snow(self.flakes * n)

    def __truediv__(self, n: int) -> 'Snow':
        if n != 0:
            return Snow(round(self.flakes / n))
        raise ValueError("Division by zero")

    def makeSnow(self, row_flakes: int) -> str:
        if row_flakes <= 0:
            raise ValueError("Count of flakes must be positive")
        rows: int = self.flakes // row_flakes
        line = '*' * row_flakes
        return ''.join([line for _ in range(rows)])

    def __call__(self, new_flakes: int) -> None:
        self.flakes = new_flakes

    def __str__(self) -> str:
        return f"Flakes count: {self.flakes}"


snow1 = Snow(10)
snow2 = Snow(5)

print(snow1 + snow2)
print(snow1 - snow2)
print(snow1 * 2)
print(snow1 / 3)
print(snow1.makeSnow(3))
snow1(20)
print(snow1)
