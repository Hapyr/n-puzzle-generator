# N-Puzzle Problem Generator and Solver

Fast Rust-based Python extension for N-puzzle problem generation and solving. Uses A* with Manhattan distance + linear conflict heuristics. Parallelized via Rayon.

## Installation

```bash
maturin build --release
pip install .
```

## API Reference

All functions accept `size_x` and `size_y` parameters (default: 3×3). States are `list[int]` with `0` as blank tile.

### Solving

```python
solve_puzzle(start_state, goal_state, max_depth=100) -> Optional[list[int]]
solve_puzzle_with_stats(start_state, goal_state, max_depth=100, size_x=3, size_y=3) -> tuple[Optional[list[int]], int]
compute_heuristic(state, goal_state, size_x, size_y) -> int
```

### Pair Generation

```python
generate_pair(target_distance, max_attempts=1000, size_x=3, size_y=3, verify=True, fixed_goal=False) -> tuple[list, list, int]
generate_pairs(target_distances, max_attempts_per_pair=1000, size_x=3, size_y=3, verify=True, fixed_goal=False) -> list[tuple]
generate_pair_with_backtracking(target_distance, size_x=3, size_y=3) -> tuple[list, list, int]
generate_pair_same_blank(target_distance, max_attempts=1000, size_x=3, size_y=3) -> tuple[list, list, int]
generate_pairs_same_blank(target_distances, max_attempts_per_pair=1000, size_x=3, size_y=3) -> list[tuple]
```

### Dataset Generation

```python
generate_dataset(n_samples=1000, p_successor=0.4, p_extended=0.3, size_x=3, size_y=3) -> list[tuple]
compare_datasets(d1_n_samples=1000, d2_n_samples=1000) -> int
compare_datasets_parallel(n=10, d1_n_samples=1000, d2_n_samples=1000) -> list[int]
```

### Visualization

```python
pretty_print_state(state, size_x=3, size_y=3) -> str
pretty_print_pair(state1, state2, label1="State 1", label2="State 2", size_x=3, size_y=3) -> str
```

## Actions

`0=up, 1=right, 2=down, 3=left`

## State Representation

Flat list, row-major order. Example 3×3:
```
1 2 3
4 5 6    →  [1, 2, 3, 4, 5, 6, 7, 8, 0]
7 8 _
```
