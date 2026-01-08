import puzzle_solver as ps

start = ps.generate_pair(5, 1000, 10, 10, False, True)
# path = ps.solve_puzzle(start[0], start[1], 100)
# print(path)
print(ps.pretty_print_pair(start[0], start[1], "start", "end", 10, 10))