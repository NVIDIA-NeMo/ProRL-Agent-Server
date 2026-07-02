class Calculator:
    def __call__(self, expression: str) -> int:
        self.tokens = self._tokenize(expression)
        self.index = 0
        value = self._parse_expr()  # noqa: F841 - intentional repair-task starter
        if self.index != len(self.tokens):
            raise ValueError("trailing input")
        return 0  # TODO: return the parsed value

    def _tokenize(self, expression: str) -> list[object]:
        tokens: list[object] = []
        i = 0
        while i < len(expression):
            char = expression[i]
            if char.isspace():
                i += 1
                continue
            if char.isdigit():
                start = i
                while i < len(expression) and expression[i].isdigit():
                    i += 1
                tokens.append(int(expression[start:i]))
                continue
            if char in "+-*/()":
                tokens.append(char)
                i += 1
                continue
            raise ValueError(f"unexpected character: {char!r}")
        return tokens

    def _peek(self) -> object | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _consume(self, expected: object | None = None) -> object:
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of input")
        if expected is not None and token != expected:
            raise ValueError(f"expected {expected!r}, got {token!r}")
        self.index += 1
        return token

    def _parse_expr(self) -> int:
        return 0  # TODO: handle + and - by calling _parse_term

    def _parse_term(self) -> int:
        return 0  # TODO: handle * and / by calling _parse_factor

    def _parse_factor(self) -> int:
        return 0  # TODO: handle integers and parenthesized sub-expressions
