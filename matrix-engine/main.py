# add, subtract, multiply and transposition a matrices A & B
# 2x2 first then 3x3

a = [[2, 4], [3, 9]]  # len(a) -> row, len(a[0]) -> column
b = [[1, 2], [4, 8]]
c = [[1, 4, 2], [3, 2, 6], [5, 4, 8]]  # 3x3 matrix

# add
result = [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]
print("Addition result", result)

# subtract
result = [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]
print("Subtraction result", result)

# multiply
result = [[0, 0], [0, 0]]

for i in range(2):  # rows of a as i
    for j in range(2):  # columns of b as j
        total = 0
        for k in range(2):
            total += (
                a[i][k] * b[k][j]
            )  # a[i][k] -> find row i(0/1) and element k(0/1) in matrix a, b[k][j] -> find element k(0/1) in column j(0/1)
            result[i][j] = total
            print("Multiplication result", result)

# transposition
for i in range(len(a[0])):  # loop through column
    result = []
    for j in range(len(a)):  # loop through row
        result.append(a[j][i])  # swap positions
        print("Transposition result", result)

# 3x3 matrix solution
