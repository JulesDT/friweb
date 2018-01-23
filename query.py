possible_operators = '|&~'


class QueryParser:

    def __init__(self, query):
        self.string_query = query.replace(' ', '')

    def parse(self):
        if len(self.string_query) == 0:
            return Query()
        if self.string_query[0] == '(':
            depth = 1
            i = 1
            while depth != 0:
                char = self.string_query[i]
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                i += 1
            print(self.string_query, i)
            operator = self.string_query[i]
            # print(self.string_query, i, self.string_query[i + 1])
            return Query(operator, [QueryParser(self.string_query[1:i - 1]).parse(), QueryParser(self.string_query[i + 1:]).parse()])
        elif self.string_query[0] not in possible_operators:
            i = 0
            while i < len(self.string_query):
                if self.string_query[i] in possible_operators:
                    break
                i += 1
            if i == len(self.string_query):
                return Query(value=self.string_query)
            else:
                return Query(operator=self.string_query[i],
                            childrens=[Query(value=self.string_query[:i]), QueryParser(self.string_query[i + 1:]).parse()])


class Query:

    def __init__(self, operator=None, childrens=[], value=None):
        self.operator = operator
        self.childrens = childrens
        self.value = value

    def __str__(self):
        if len(self.childrens) == 2:
            return str(self.childrens[0]) + self.operator + str(self.childrens[1])
        else:
            return self.value

a = QueryParser('(A & B) | C')

parsed = a.parse()

print(parsed.childrens, parsed.operator)
print(parsed.childrens[0].childrens, parsed.childrens[0].operator)
print(str(parsed.childrens[0].childrens[0]), str(parsed.childrens[0].childrens[1]))