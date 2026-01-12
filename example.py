import puzzle_solver as ps


# problems = ps.generate_pairs([2,3,4,5], 1000, 3, 3, False, False)
problems = ps.generate_pairs_with_backtracking([1,1,1,1], 3, 3)
# path = ps.solve_puzzle(start[0], start[1], 100)
# print(path)

for problem in problems:
    print(problem)
    print(ps.pretty_print_pair(problem[0], problem[1], "start", "end", 3, 3))