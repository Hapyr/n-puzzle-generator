use indicatif::{ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// State representation for 8-puzzle (3x3 grid as flat array)
#[derive(Clone, Hash, Eq, PartialEq)]
struct PuzzleState {
    tiles: Vec<u8>,
    blank_pos: usize,
    size_x: usize,
    size_y: usize,
}

impl PuzzleState {
    fn from_flat(tiles: &[u8], size_x: usize, size_y: usize) -> Self {
        let blank_pos = tiles.iter().position(|&t| t == 0).unwrap();
        PuzzleState {
            tiles: tiles.to_vec(),
            blank_pos,
            size_x,
            size_y,
        }
    }

    fn goal(size_x: usize, size_y: usize) -> Self {
        let tiles = (0..size_x * size_y)
            .map(|i| {
                if i == size_x * size_y - 1 {
                    0
                } else {
                    (i + 1) as u8
                }
            })
            .collect::<Vec<u8>>();
        Self::from_flat(&tiles, size_x, size_y)
    }

    fn random(size_x: usize, size_y: usize) -> Self {
        let mut state = Self::goal(size_x, size_y);
        state.shuffle();
        state
    }

    fn shuffle(&mut self) {
        // Fisher-Yates shuffle
        for i in (1..self.tiles.len()).rev() {
            let j = fastrand::usize(..=i);
            self.tiles.swap(i, j);
        }
        self.blank_pos = self.tiles.iter().position(|&t| t == 0).unwrap();
    }

    /// Get valid moves: 0=up, 1=right, 2=down, 3=left
    fn get_valid_moves(&self) -> Vec<u8> {
        let row = self.blank_pos / self.size_x;
        let col = self.blank_pos % self.size_x;
        let mut moves = Vec::with_capacity(4);

        if row > 0 {
            moves.push(0);
        } // up
        if col < self.size_x - 1 {
            moves.push(1);
        } // right
        if row < self.size_y - 1 {
            moves.push(2);
        } // down
        if col > 0 {
            moves.push(3);
        } // left

        moves
    }

    /// Get the opposite/reverse action
    fn opposite_action(action: u8) -> u8 {
        match action {
            0 => 2, // up -> down
            1 => 3, // right -> left
            2 => 0, // down -> up
            3 => 1, // left -> right
            _ => action,
        }
    }

    /// Apply a move and return new state
    fn apply_move(&self, action: u8) -> Self {
        let new_blank = match action {
            0 => self.blank_pos - self.size_x, // up
            1 => self.blank_pos + 1,           // right
            2 => self.blank_pos + self.size_x, // down
            3 => self.blank_pos - 1,           // left
            _ => self.blank_pos,
        };

        let mut new_tiles = self.tiles.clone();
        new_tiles.swap(self.blank_pos, new_blank);

        PuzzleState {
            tiles: new_tiles,
            blank_pos: new_blank,
            size_x: self.size_x,
            size_y: self.size_y,
        }
    }

    /// Make n random moves (avoiding immediate backtracking)
    fn random_walk(&self, n: usize) -> (Self, Vec<u8>) {
        let mut last_action: Option<u8> = None;
        let mut state = self.clone();
        let mut path = Vec::new();

        for _ in 0..n {
            let moves = state.get_valid_moves();
            // Filter out the reverse of the last action to avoid backtracking
            let valid_moves: Vec<u8> = moves
                .into_iter()
                .filter(|&m| last_action.map_or(true, |la| m != Self::opposite_action(la)))
                .collect();

            let action = valid_moves[fastrand::usize(..valid_moves.len())];
            state = state.apply_move(action);
            last_action = Some(action);
            path.push(action);
        }

        (
            PuzzleState {
                tiles: state.tiles,
                blank_pos: state.blank_pos,
                size_x: self.size_x,
                size_y: self.size_y,
            },
            path,
        )
    }

    fn random_walk_with_backtracking(&self, n: usize) -> (Self, Option<u8>) {
        let mut state = self.clone();
        let mut first_action: Option<u8> = None;

        for _ in 0..n {
            let moves = state.get_valid_moves();
            let action = moves[fastrand::usize(..moves.len())];
            state = state.apply_move(action);
            if first_action.is_none() {
                first_action = Some(action);
            }
        }

        (
            PuzzleState {
                tiles: state.tiles,
                blank_pos: state.blank_pos,
                size_x: self.size_x,
                size_y: self.size_y,
            },
            first_action,
        )
    }
}

/// Compute goal positions for a given goal state
fn compute_positions(state: &PuzzleState) -> Vec<(usize, usize)> {
    let mut positions = Vec::new();
    for (i, &_tile) in state.tiles.iter().enumerate() {
        positions.push((i / state.size_x, i % state.size_x));
    }
    positions
}

/// Manhattan distance + Linear conflict heuristic
fn heuristic(state: &PuzzleState, positions: &Vec<(usize, usize)>) -> u32 {
    let mut distance: u32 = 0;

    // Manhattan distance (skip blank tile)
    for (i, &tile) in state.tiles.iter().enumerate() {
        if tile == 0 {
            continue;
        }

        let curr_row = i / state.size_x;
        let curr_col = i % state.size_x;
        let (goal_row, goal_col) = positions[tile as usize];

        distance += (curr_row as i32 - goal_row as i32).unsigned_abs()
            + (curr_col as i32 - goal_col as i32).unsigned_abs();
    }

    // Linear conflicts in rows
    for row in 0..state.size_y {
        let mut tiles_in_row: Vec<(usize, usize)> = Vec::new();
        for col in 0..state.size_x {
            let tile = state.tiles[row * state.size_x + col];
            if tile != 0 && positions[tile as usize].0 == row {
                tiles_in_row.push((col, positions[tile as usize].1));
            }
        }

        for i in 0..tiles_in_row.len() {
            for j in (i + 1)..tiles_in_row.len() {
                let (curr_col_i, goal_col_i) = tiles_in_row[i];
                let (curr_col_j, goal_col_j) = tiles_in_row[j];
                if (curr_col_i < curr_col_j) != (goal_col_i < goal_col_j) {
                    distance += 2;
                }
            }
        }
    }

    // Linear conflicts in columns
    for col in 0..state.size_x {
        let mut tiles_in_col: Vec<(usize, usize)> = Vec::new();
        for row in 0..state.size_y {
            let tile = state.tiles[row * state.size_x + col];
            if tile != 0 && positions[tile as usize].1 == col {
                tiles_in_col.push((row, positions[tile as usize].0));
            }
        }

        for i in 0..tiles_in_col.len() {
            for j in (i + 1)..tiles_in_col.len() {
                let (curr_row_i, goal_row_i) = tiles_in_col[i];
                let (curr_row_j, goal_row_j) = tiles_in_col[j];
                if (curr_row_i < curr_row_j) != (goal_row_i < goal_row_j) {
                    distance += 2;
                }
            }
        }
    }

    distance
}

/// Node in the priority queue
#[derive(Clone, Eq, PartialEq)]
struct SearchNode {
    f_score: u32,
    g_score: u32,
    state: PuzzleState,
    path: Vec<u8>,
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (lower f_score = higher priority)
        other
            .f_score
            .cmp(&self.f_score)
            .then_with(|| other.g_score.cmp(&self.g_score))
    }
}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A* search implementation - returns (path, nodes_visited) or None
fn a_star_search(
    start: PuzzleState,
    goal: PuzzleState,
    max_depth: u32,
) -> (Option<Vec<u8>>, usize) {
    let positions = compute_positions(&goal);

    let mut heap = BinaryHeap::new();
    let mut visited = HashSet::new();

    let h_start = heuristic(&start, &positions);
    heap.push(SearchNode {
        f_score: h_start,
        g_score: 0,
        state: start,
        path: Vec::new(),
    });

    while let Some(node) = heap.pop() {
        // Check depth limit
        if node.g_score > max_depth {
            continue;
        }

        // Goal check
        if node.state == goal {
            return (Some(node.path), visited.len());
        }

        // Skip if visited
        if !visited.insert(node.state.clone()) {
            continue;
        }

        // Expand neighbors
        for action in node.state.get_valid_moves() {
            let next_state = node.state.apply_move(action);

            if !visited.contains(&next_state) {
                let new_g = node.g_score + 1;
                let new_h = heuristic(&next_state, &positions);
                let new_f = new_g + new_h;

                let mut new_path = node.path.clone();
                new_path.push(action);

                heap.push(SearchNode {
                    f_score: new_f,
                    g_score: new_g,
                    state: next_state,
                    path: new_path,
                });
            }
        }
    }

    (None, visited.len())
}

/// Python function: solve 8-puzzle using A*
/// Returns: action sequence or None
#[pyfunction]
#[pyo3(signature = (start_state, goal_state, max_depth=100))]
fn solve_puzzle(start_state: Vec<u8>, goal_state: Vec<u8>, max_depth: u32) -> Option<Vec<u8>> {
    let start = PuzzleState::from_flat(&start_state, start_state.len() as usize / 3, 3);
    let goal = PuzzleState::from_flat(&goal_state, goal_state.len() as usize / 3, 3);

    a_star_search(start, goal, max_depth).0
}

/// Python function: solve 8-puzzle using A* and return node count
/// Returns: (action_sequence, nodes_visited) or (None, nodes_visited)
#[pyfunction]
#[pyo3(signature = (start_state, goal_state, max_depth=100, size_x=3, size_y=3))]
fn solve_puzzle_with_stats(
    start_state: Vec<u8>,
    goal_state: Vec<u8>,
    max_depth: u32,
    size_x: usize,
    size_y: usize,
) -> (Option<Vec<u8>>, usize) {
    let start = PuzzleState::from_flat(&start_state, size_x, size_y);
    let goal = PuzzleState::from_flat(&goal_state, size_x, size_y);
    a_star_search(start, goal, max_depth)
}

/// Python function: compute heuristic value between two states
#[pyfunction]
fn compute_heuristic(state: Vec<u8>, goal_state: Vec<u8>, size_x: usize, size_y: usize) -> u32 {
    let s = PuzzleState::from_flat(&state, size_x, size_y);
    let g = PuzzleState::from_flat(&goal_state, size_x, size_y);
    let positions = compute_positions(&g);

    heuristic(&s, &positions)
}

/// Generate a state pair by a random walk without backtracking.
///
/// Args:
///     target_distance: The exact optimal distance between the states
///     max_attempts: Maximum attempts before giving up (default 1000, only used when verify is True)
///     size_x: Width of the puzzle (default 3)
///     size_y: Height of the puzzle (default 3)
///     verify: Whether to verify the distance (default True)
///     fixed_goal: Whether to use a fixed goal state (default False)
///
/// Returns:
///     Tuple of (start_state, end_state, action) as flattened lists, or None
fn generate_pair_internal(
    target_distance: u32,
    max_attempts: u32,
    size_x: usize,
    size_y: usize,
    verify: bool,
    fixed_goal: bool,
) -> Option<(Vec<u8>, Vec<u8>, u8)> {
    for _ in 0..max_attempts {
        let mut end = PuzzleState::random(size_x, size_y);
        if fixed_goal {
            end = PuzzleState::goal(size_x, size_y);
        }
        let (start, path) = end.random_walk(target_distance as usize);

        // verification only for 3x3
        if size_x != 3 || size_y != 3 {
            eprintln!("\x1b[1;33m[warning]\x1b[0m \x1b[33mMinimal distance verification is only implemented for 3x3 puzzles. Returning random state pair without minimal distance guarantee.\x1b[0m");
            return Some((
                start.tiles,
                end.tiles,
                PuzzleState::opposite_action(path[path.len() - 1]),
            ));
        }

        if !verify {
            return Some((
                start.tiles,
                end.tiles,
                PuzzleState::opposite_action(path[path.len() - 1]),
            ));
        }

        // Verify actual distance with A*
        if let (Some(path), _) = a_star_search(start.clone(), end.clone(), target_distance + 5) {
            if path.len() as u32 == target_distance {
                return Some((start.tiles.clone(), end.tiles.clone(), path[0]));
            }
        }
    }
    None
}

#[pyfunction]
#[pyo3(signature = (target_distance, max_attempts=1000, size_x=3, size_y=3, verify=true, fixed_goal=false))]
fn generate_pair(
    target_distance: u32,
    max_attempts: u32,
    size_x: usize,
    size_y: usize,
    verify: bool,
    fixed_goal: bool,
) -> Option<(Vec<u8>, Vec<u8>, u8)> {
    generate_pair_internal(
        target_distance,
        max_attempts,
        size_x,
        size_y,
        verify,
        fixed_goal,
    )
}

/// Generate a single pair of states with backtracking. Therefore the provided target distance is not minimal.
///
/// Args:
///     target_distance: The exact optimal distance between the states
///
/// Returns:
///     Tuple of (start_state, end_state, action) as flattened lists, or None
#[pyfunction]
#[pyo3(signature = (target_distance, size_x=3, size_y=3))]
fn generate_pair_with_backtracking(
    target_distance: u32,
    size_x: usize,
    size_y: usize,
) -> Option<(Vec<u8>, Vec<u8>, u8)> {
    let start = PuzzleState::random(size_x, size_y);
    let (end, first_action) = start.random_walk_with_backtracking(target_distance as usize);
    if let Some(first_action) = first_action {
        return Some((start.tiles, end.tiles, first_action));
    }
    None
}

#[pyfunction]
#[pyo3(signature = (target_distances, size_x=3, size_y=3))]
fn generate_pairs_with_backtracking(
    target_distances: Vec<u32>,
    size_x: usize,
    size_y: usize,
) -> Vec<(Vec<u8>, Vec<u8>, u8)> {
    target_distances
        .par_iter()
        .filter_map(|&dist| generate_pair_with_backtracking(dist, size_x, size_y))
        .collect()
}

/// Generate multiple pairs of states at exact distance (batch version), parallelized
///
/// Args:
///     target_distances: List of target distances (one per pair to generate)
///     max_attempts_per_pair: Maximum attempts per pair (default 100)
///
/// Returns:
///     List of (start_state, end_state) tuples as lists of tiles
#[pyfunction]
#[pyo3(signature = (target_distances, max_attempts_per_pair=1000, size_x=3, size_y=3, verify=true, fixed_goal=false))]
fn generate_pairs(
    target_distances: Vec<u32>,
    max_attempts_per_pair: u32,
    size_x: usize,
    size_y: usize,
    verify: bool,
    fixed_goal: bool,
) -> Vec<(Vec<u8>, Vec<u8>, u8)> {
    let pb = ProgressBar::new(target_distances.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Generating pairs without backtracking")
        .expect("Failed to set style")
        .progress_chars("##-"));

    let results: Vec<(Vec<u8>, Vec<u8>, u8)> = target_distances
        .par_iter()
        .map_init(
            || pb.clone(),
            |progress, &dist| {
                // Keep trying until we get a valid pair
                let pair = loop {
                    if let Some(pair) = generate_pair_internal(
                        dist,
                        max_attempts_per_pair,
                        size_x,
                        size_y,
                        verify,
                        fixed_goal,
                    ) {
                        break pair;
                    }
                };
                progress.inc(1);
                pair
            },
        )
        .collect();
    pb.finish_with_message("Pairs generation without backtracking done.");
    results
}

/// Limited DFS to find states at exact depth with same blank position
/// Returns all states found at the target depth with matching blank position
fn dfs_same_blank(
    start: &PuzzleState,
    target_depth: u32,
    current_state: PuzzleState,
    current_depth: u32,
    last_action: Option<u8>,
    results: &mut Vec<PuzzleState>,
    max_results: usize,
) {
    // If we've collected enough results, stop
    if results.len() >= max_results {
        return;
    }

    // At target depth, check if blank is in same position and state is different
    if current_depth == target_depth {
        if current_state.blank_pos == start.blank_pos && current_state.tiles != start.tiles {
            results.push(current_state);
        }
        return;
    }

    // Continue DFS
    let mut moves = current_state.get_valid_moves();
    fastrand::shuffle(&mut moves);
    for action in moves {
        // Avoid immediate backtracking
        if last_action.map_or(false, |la| action == PuzzleState::opposite_action(la)) {
            continue;
        }

        let next_state = current_state.apply_move(action);
        dfs_same_blank(
            start,
            target_depth,
            next_state,
            current_depth + 1,
            Some(action),
            results,
            max_results,
        );

        if results.len() >= max_results {
            return;
        }
    }
}

/// Internal function to generate a pair using DFS to find state with same blank position at exact distance
fn generate_pair_same_blank_internal(
    target_distance: u32,
    max_attempts: u32,
    size_x: usize,
    size_y: usize,
) -> Option<(Vec<u8>, Vec<u8>, u8)> {
    // warning
    if size_x != 3 || size_y != 3 {
        eprintln!("\x1b[1;33m[warning]\x1b[0m \x1b[33mIts not recommended to use this function for puzzles other than 3x3 due to computational complexity.\x1b[0m");
    }
    for _ in 0..max_attempts {
        let start_init = PuzzleState::random(size_x, size_y);
        let mut start = start_init.clone();
        // Make a random valid action to further randomize the starting state
        let moves = start.get_valid_moves();
        let _action = if !moves.is_empty() {
            let a = moves[fastrand::usize(..moves.len())];
            start = start_init.apply_move(a);
            Some((a + 2) % 4)
        } else {
            None
        };

        // Use DFS to find candidates at target depth with same blank position
        let mut candidates = Vec::new();
        dfs_same_blank(
            &start,
            target_distance,
            start_init.clone(),
            1,
            None,
            &mut candidates,
            100, // Collect up to 100 candidates per attempt
        );

        // Shuffle candidates and try to find one at exact distance
        if !candidates.is_empty() {
            // Randomly pick from candidates
            for _ in 0..candidates.len().min(10) {
                let idx = fastrand::usize(..candidates.len());
                let end = candidates[idx].clone();

                // Verify actual distance with A*
                if let (Some(path), _) =
                    a_star_search(start.clone(), end.clone(), target_distance + 2)
                {
                    if path.len() as u32 == target_distance {
                        return Some((start_init.tiles.clone(), end.tiles.clone(), path[1]));
                    }
                }
            }
        }
    }
    None
}

/// Generate a pair using DFS to find state with same blank position at exact distance
///
/// Args:
///     target_distance: The exact optimal distance between the states
///     max_attempts: Maximum random starts before giving up (default 1000)
///
/// Returns:
///     Tuple of (start_state, end_state, action) as flattened lists, or None
#[pyfunction]
#[pyo3(signature = (target_distance, max_attempts=1000, size_x=3, size_y=3))]
fn generate_pair_same_blank(
    target_distance: u32,
    max_attempts: u32,
    size_x: usize,
    size_y: usize,
) -> Option<(Vec<u8>, Vec<u8>, u8)> {
    generate_pair_same_blank_internal(target_distance + 1, max_attempts, size_x, size_y)
}

/// Generate multiple pairs using DFS method (same blank position), parallelized
///
/// Args:
///     target_distances: List of target distances (one per pair to generate)
///     max_attempts_per_pair: Maximum attempts per pair (default 100)
///
/// Returns:
///     List of (start_state, end_state, action) tuples as flattened lists
#[pyfunction]
#[pyo3(signature = (target_distances, max_attempts_per_pair=1000, size_x=3, size_y=3))]
fn generate_pairs_same_blank(
    target_distances: Vec<u32>,
    max_attempts_per_pair: u32,
    size_x: usize,
    size_y: usize,
) -> Vec<(Vec<u8>, Vec<u8>, u8)> {
    // Fixed version: No extraneous code block, just the collection and progress bar in the function body

    let pb = ProgressBar::new(target_distances.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Generating same blank pairs",
            )
            .expect("Failed to set style")
            .progress_chars("##-"),
    );

    let result = target_distances
        .par_iter()
        .enumerate()
        .filter_map(|(_i, &dist)| {
            // Keep trying until we get a valid pair
            loop {
                if let Some(pair) = generate_pair_same_blank_internal(
                    dist + 1,
                    max_attempts_per_pair,
                    size_x,
                    size_y,
                ) {
                    pb.inc(1);
                    return Some(pair);
                }
            }
        })
        .collect();
    pb.finish_with_message("Same blank pairs generation done.");
    result
}

/// Generate a dataset of state pairs for training and evaluation.
/// Distances are optimal and verified with A* search.
/// 
/// Args:
///     n_samples: Number of samples to generate
///     p_successor: Ratio of directly connected pairs to generate (one step distance)
///     p_extended: Ratio of pairs with a random walk distance without immediate backtracking
///     size_x: Width of the puzzle
///     size_y: Height of the puzzle
/// 
/// Returns:
///     List of (start_state, end_state, action, distance) tuples as flattened lists
#[pyfunction]
#[pyo3(signature = (n_samples=1000, p_successor=0.4, p_extended=0.3, size_x=3, size_y=3))]
fn generate_dataset(
    n_samples: usize,
    p_successor: f64,
    p_extended: f64,
    size_x: usize,
    size_y: usize,
) -> Vec<(Vec<u8>, Vec<u8>, u8, u32)> {
    let mut dataset = Vec::new();

    let valid_pairs_target = (n_samples as f64 * p_successor) as usize;

    let pb = ProgressBar::new(valid_pairs_target as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .expect("Failed to set style")
            .progress_chars("##-"),
    );

    for _ in 0..valid_pairs_target {
        if let Some((start_state, end_state, action)) =
            generate_pair_with_backtracking(1, size_x, size_y)
        {
            dataset.push((start_state.to_vec(), end_state.to_vec(), action, 1));
        }
        pb.inc(1);
    }
    pb.finish_with_message("Valid pairs generation done.");

    let number_invalid_typ2_pairs =
        ((n_samples as f64 * (1.0 - p_extended)) as usize) - dataset.len() as usize;

    let distro = [3, 5, 7, 9, 11, 13, 15, 17, 19];
    let distances: Vec<u32> = (0..number_invalid_typ2_pairs)
        .map(|_| distro[fastrand::usize(..distro.len())])
        .collect();
    let pairs = generate_pairs_same_blank(distances, 1000, size_x, size_y);
    for pair in pairs {
        dataset.push((pair.0.to_vec(), pair.1.to_vec(), 4, pair.2 as u32));
    }

    let number_invalid_typ1_pairs = n_samples - dataset.len() as usize;

    let distro = [
        2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18,
        19,
    ];

    let distances: Vec<u32> = (0..number_invalid_typ1_pairs)
        .map(|_| distro[fastrand::usize(..distro.len())])
        .collect();
    let pairs = generate_pairs(distances, 1000, size_x, size_y, true, false);
    for pair in pairs {
        dataset.push((pair.0.to_vec(), pair.1.to_vec(), 4, pair.2 as u32));
    }

    println!("\x1b[1;32m[info]\x1b[0m \x1b[32mDataset generation done.\x1b[0m");

    dataset
}

/// Pretty print a puzzle state with colors and box-drawing borders
///
/// Args:
///     state: Flattened puzzle state as a list of integers
///     size_x: Width of the puzzle (default 3)
///     size_y: Height of the puzzle (default 3)
///
/// Returns:
///     A beautifully formatted string representation of the puzzle
#[pyfunction]
#[pyo3(signature = (state, size_x=3, size_y=3))]
fn pretty_print_state(state: Vec<u8>, size_x: usize, size_y: usize) -> String {
    let puzzle = PuzzleState::from_flat(&state, size_x, size_y);

    // ANSI color codes
    const RESET: &str = "\x1b[0m";
    const BOLD: &str = "\x1b[1m";
    const DIM: &str = "\x1b[2m";

    // Extended colors for tiles (cycle through for larger puzzles)
    const TILE_COLORS: [&str; 16] = [
        "\x1b[90m",       // 0 (blank) - dark gray
        "\x1b[38;5;196m", // 1 - red
        "\x1b[38;5;208m", // 2 - orange
        "\x1b[38;5;226m", // 3 - yellow
        "\x1b[38;5;46m",  // 4 - green
        "\x1b[38;5;51m",  // 5 - cyan
        "\x1b[38;5;21m",  // 6 - blue
        "\x1b[38;5;129m", // 7 - purple
        "\x1b[38;5;201m", // 8 - magenta
        "\x1b[38;5;160m", // 9 - dark red
        "\x1b[38;5;166m", // 10 - dark orange
        "\x1b[38;5;190m", // 11 - lime
        "\x1b[38;5;42m",  // 12 - teal
        "\x1b[38;5;33m",  // 13 - dodger blue
        "\x1b[38;5;93m",  // 14 - violet
        "\x1b[38;5;199m", // 15 - hot pink
    ];

    // Background for the blank tile
    const BLANK_BG: &str = "\x1b[48;5;236m"; // dark gray background

    // Box-drawing characters (double-line style)
    const TOP_LEFT: &str = "╔";
    const TOP_RIGHT: &str = "╗";
    const BOTTOM_LEFT: &str = "╚";
    const BOTTOM_RIGHT: &str = "╝";
    const HORIZONTAL: &str = "═";
    const VERTICAL: &str = "║";
    const T_DOWN: &str = "╦";
    const T_UP: &str = "╩";
    const T_RIGHT: &str = "╠";
    const T_LEFT: &str = "╣";
    const CROSS: &str = "╬";

    // Border color
    const BORDER_COLOR: &str = "\x1b[38;5;39m"; // bright blue

    // Determine cell width based on max tile number
    let max_tile = size_x * size_y - 1;
    let cell_width = if max_tile >= 10 { 4 } else { 3 };

    let mut result = String::new();

    // Title
    let puzzle_name = format!("{}-Puzzle", size_x * size_y - 1);
    result.push_str(&format!(
        "{}{}  {} State  {}\n",
        BOLD, BORDER_COLOR, puzzle_name, RESET
    ));

    // Helper to create horizontal border segment
    let h_segment: String = (0..cell_width).map(|_| HORIZONTAL).collect();

    // Top border
    result.push_str(&format!("{}{}", BORDER_COLOR, TOP_LEFT));
    for col in 0..size_x {
        result.push_str(&h_segment);
        if col < size_x - 1 {
            result.push_str(T_DOWN);
        }
    }
    result.push_str(&format!("{}{}\n", TOP_RIGHT, RESET));

    for row in 0..size_y {
        // Content row
        result.push_str(&format!("{}{}", BORDER_COLOR, VERTICAL));

        for col in 0..size_x {
            let tile = puzzle.tiles[row * size_x + col];
            let tile_color = TILE_COLORS[(tile as usize) % TILE_COLORS.len()];

            if tile == 0 {
                // Blank tile - show as empty with background
                if cell_width == 4 {
                    result.push_str(&format!(
                        "{}{} ·  {}{}{}",
                        BLANK_BG, DIM, RESET, BORDER_COLOR, VERTICAL
                    ));
                } else {
                    result.push_str(&format!(
                        "{}{} · {}{}{}",
                        BLANK_BG, DIM, RESET, BORDER_COLOR, VERTICAL
                    ));
                }
            } else {
                // Regular tile with color
                if cell_width == 4 {
                    result.push_str(&format!(
                        " {}{}{:>2} {}{}{}",
                        BOLD, tile_color, tile, RESET, BORDER_COLOR, VERTICAL
                    ));
                } else {
                    result.push_str(&format!(
                        " {}{}{} {}{}{}",
                        BOLD, tile_color, tile, RESET, BORDER_COLOR, VERTICAL
                    ));
                }
            }
        }
        result.push_str(&format!("{}\n", RESET));

        // Separator or bottom border
        if row < size_y - 1 {
            // Middle separator
            result.push_str(&format!("{}{}", BORDER_COLOR, T_RIGHT));
            for col in 0..size_x {
                result.push_str(&h_segment);
                if col < size_x - 1 {
                    result.push_str(CROSS);
                }
            }
            result.push_str(&format!("{}{}\n", T_LEFT, RESET));
        } else {
            // Bottom border
            result.push_str(&format!("{}{}", BORDER_COLOR, BOTTOM_LEFT));
            for col in 0..size_x {
                result.push_str(&h_segment);
                if col < size_x - 1 {
                    result.push_str(T_UP);
                }
            }
            result.push_str(&format!("{}{}\n", BOTTOM_RIGHT, RESET));
        }
    }

    result
}

/// Pretty print two puzzle states side by side (for comparing state pairs)
///
/// Args:
///     state1: First flattened puzzle state
///     state2: Second flattened puzzle state
///     label1: Optional label for first state (default "State 1")
///     label2: Optional label for second state (default "State 2")
///     size_x: Width of the puzzle (default 3)
///     size_y: Height of the puzzle (default 3)
///
/// Returns:
///     A beautifully formatted string showing both states side by side
#[pyfunction]
#[pyo3(signature = (state1, state2, label1="State 1", label2="State 2", size_x=3, size_y=3))]
fn pretty_print_pair(
    state1: Vec<u8>,
    state2: Vec<u8>,
    label1: &str,
    label2: &str,
    size_x: usize,
    size_y: usize,
) -> String {
    let puzzle1 = PuzzleState::from_flat(&state1, size_x, size_y);
    let puzzle2 = PuzzleState::from_flat(&state2, size_x, size_y);

    // ANSI color codes
    const RESET: &str = "\x1b[0m";
    const BOLD: &str = "\x1b[1m";
    const DIM: &str = "\x1b[2m";

    // Extended colors for tiles (cycle through for larger puzzles)
    const TILE_COLORS: [&str; 16] = [
        "\x1b[90m",       // 0 (blank) - dark gray
        "\x1b[38;5;196m", // 1 - red
        "\x1b[38;5;208m", // 2 - orange
        "\x1b[38;5;226m", // 3 - yellow
        "\x1b[38;5;46m",  // 4 - green
        "\x1b[38;5;51m",  // 5 - cyan
        "\x1b[38;5;21m",  // 6 - blue
        "\x1b[38;5;129m", // 7 - purple
        "\x1b[38;5;201m", // 8 - magenta
        "\x1b[38;5;160m", // 9 - dark red
        "\x1b[38;5;166m", // 10 - dark orange
        "\x1b[38;5;190m", // 11 - lime
        "\x1b[38;5;42m",  // 12 - teal
        "\x1b[38;5;33m",  // 13 - dodger blue
        "\x1b[38;5;93m",  // 14 - violet
        "\x1b[38;5;199m", // 15 - hot pink
    ];

    const BLANK_BG: &str = "\x1b[48;5;236m";
    const BORDER_COLOR: &str = "\x1b[38;5;39m";

    // Box-drawing characters
    const TOP_LEFT: &str = "╔";
    const TOP_RIGHT: &str = "╗";
    const BOTTOM_LEFT: &str = "╚";
    const BOTTOM_RIGHT: &str = "╝";
    const HORIZONTAL: &str = "═";
    const VERTICAL: &str = "║";
    const T_DOWN: &str = "╦";
    const T_UP: &str = "╩";
    const T_RIGHT: &str = "╠";
    const T_LEFT: &str = "╣";
    const CROSS: &str = "╬";

    // Determine cell width based on max tile number
    let max_tile = size_x * size_y - 1;
    let cell_width = if max_tile >= 10 { 4 } else { 3 };

    // Helper to create horizontal border segment
    let h_segment: String = (0..cell_width).map(|_| HORIZONTAL).collect();

    fn format_tile(
        tile: u8,
        tile_colors: &[&str; 16],
        bold: &str,
        dim: &str,
        blank_bg: &str,
        border_color: &str,
        vertical: &str,
        reset: &str,
        cell_width: usize,
    ) -> String {
        let color_idx = (tile as usize) % tile_colors.len();
        if tile == 0 {
            if cell_width == 4 {
                format!(
                    "{}{} ·  {}{}{}",
                    blank_bg, dim, reset, border_color, vertical
                )
            } else {
                format!(
                    "{}{} · {}{}{}",
                    blank_bg, dim, reset, border_color, vertical
                )
            }
        } else {
            if cell_width == 4 {
                format!(
                    " {}{}{:>2} {}{}{}",
                    bold, tile_colors[color_idx], tile, reset, border_color, vertical
                )
            } else {
                format!(
                    " {}{}{} {}{}{}",
                    bold, tile_colors[color_idx], tile, reset, border_color, vertical
                )
            }
        }
    }

    let mut result = String::new();

    // Calculate label padding to center labels over the puzzles
    let puzzle_width = size_x * (cell_width + 1) + 1; // +1 for each vertical bar
    let label1_padding = (puzzle_width.saturating_sub(label1.len())) / 2;
    let label2_padding = (puzzle_width.saturating_sub(label2.len())) / 2;

    // Labels
    result.push_str(&format!(
        "{}{}{:>width1$}{}{}     {}{}{:>width2$}{}{}\n",
        BOLD,
        BORDER_COLOR,
        "",
        label1,
        RESET,
        BOLD,
        BORDER_COLOR,
        "",
        label2,
        RESET,
        width1 = label1_padding,
        width2 = label2_padding
    ));

    // Top borders
    result.push_str(&format!("{}{}", BORDER_COLOR, TOP_LEFT));
    for col in 0..size_x {
        result.push_str(&h_segment);
        if col < size_x - 1 {
            result.push_str(T_DOWN);
        }
    }
    result.push_str(&format!("{}{}     ", TOP_RIGHT, RESET));
    result.push_str(&format!("{}{}", BORDER_COLOR, TOP_LEFT));
    for col in 0..size_x {
        result.push_str(&h_segment);
        if col < size_x - 1 {
            result.push_str(T_DOWN);
        }
    }
    result.push_str(&format!("{}{}\n", TOP_RIGHT, RESET));

    for row in 0..size_y {
        // Content row - first puzzle
        result.push_str(&format!("{}{}", BORDER_COLOR, VERTICAL));
        for col in 0..size_x {
            let tile = puzzle1.tiles[row * size_x + col];
            result.push_str(&format_tile(
                tile,
                &TILE_COLORS,
                BOLD,
                DIM,
                BLANK_BG,
                BORDER_COLOR,
                VERTICAL,
                RESET,
                cell_width,
            ));
        }
        result.push_str(RESET);

        // Separator between puzzles
        result.push_str("     ");

        // Content row - second puzzle
        result.push_str(&format!("{}{}", BORDER_COLOR, VERTICAL));
        for col in 0..size_x {
            let tile = puzzle2.tiles[row * size_x + col];
            result.push_str(&format_tile(
                tile,
                &TILE_COLORS,
                BOLD,
                DIM,
                BLANK_BG,
                BORDER_COLOR,
                VERTICAL,
                RESET,
                cell_width,
            ));
        }
        result.push_str(&format!("{}\n", RESET));

        // Separator or bottom
        if row < size_y - 1 {
            // Middle separator - first puzzle
            result.push_str(&format!("{}{}", BORDER_COLOR, T_RIGHT));
            for col in 0..size_x {
                result.push_str(&h_segment);
                if col < size_x - 1 {
                    result.push_str(CROSS);
                }
            }
            result.push_str(&format!("{}{}     ", T_LEFT, RESET));
            // Middle separator - second puzzle
            result.push_str(&format!("{}{}", BORDER_COLOR, T_RIGHT));
            for col in 0..size_x {
                result.push_str(&h_segment);
                if col < size_x - 1 {
                    result.push_str(CROSS);
                }
            }
            result.push_str(&format!("{}{}\n", T_LEFT, RESET));
        } else {
            // Bottom border - first puzzle
            result.push_str(&format!("{}{}", BORDER_COLOR, BOTTOM_LEFT));
            for col in 0..size_x {
                result.push_str(&h_segment);
                if col < size_x - 1 {
                    result.push_str(T_UP);
                }
            }
            result.push_str(&format!("{}{}     ", BOTTOM_RIGHT, RESET));
            // Bottom border - second puzzle
            result.push_str(&format!("{}{}", BORDER_COLOR, BOTTOM_LEFT));
            for col in 0..size_x {
                result.push_str(&h_segment);
                if col < size_x - 1 {
                    result.push_str(T_UP);
                }
            }
            result.push_str(&format!("{}{}\n", BOTTOM_RIGHT, RESET));
        }
    }

    result
}

#[pyfunction]
#[pyo3(signature = (d1_n_samples=1000, d2_n_samples=1000))]
fn compare_datasets(d1_n_samples: usize, d2_n_samples: usize) -> u32 {
    println!("Generating dataset 1");
    let dataset1 = generate_dataset(d1_n_samples, 0.4, 0.3, 3, 3);
    println!("Generating dataset 2");
    let dataset2 = generate_dataset(d2_n_samples, 0.4, 0.3, 3, 3);
    println!("Comparing datasets");
    number_of_equal_pairs(&dataset1, &dataset2)
}

#[pyfunction]
#[pyo3(signature = (n=10, d1_n_samples=1000, d2_n_samples=1000))]
fn compare_datasets_parallel(n: usize, d1_n_samples: usize, d2_n_samples: usize) -> Vec<u32> {
    (0..n)
        .into_par_iter()
        .map(|i| {
            println!("Run {}: Generating datasets", i);
            let dataset1 = generate_dataset(d1_n_samples, 0.4, 0.3, 3, 3);
            let dataset2 = generate_dataset(d2_n_samples, 0.4, 0.3, 3, 3);
            println!("Run {}: Comparing datasets", i);
            number_of_equal_pairs(&dataset1, &dataset2)
        })
        .collect()
}

fn number_of_equal_pairs(
    dataset1: &Vec<(Vec<u8>, Vec<u8>, u8, u32)>,
    dataset2: &Vec<(Vec<u8>, Vec<u8>, u8, u32)>,
) -> u32 {
    // calculate the number of equal pairs in the two datasets
    let pb = ProgressBar::new(dataset1.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .expect("Failed to set style")
            .progress_chars("##-"),
    );

    let mut count = 0;
    for (s1, e1, _a1, _d1) in dataset1 {
        for (s2, e2, _a2, _d2) in dataset2 {
            if s1 == s2 && e1 == e2 {
                count += 1;
            }
        }
        pb.inc(1);
    }
    pb.finish_with_message("Number of equal pairs loop done");
    count
}

/// Python module
#[pymodule]
fn puzzle_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_puzzle, m)?)?;
    m.add_function(wrap_pyfunction!(solve_puzzle_with_stats, m)?)?;
    m.add_function(wrap_pyfunction!(compute_heuristic, m)?)?;
    m.add_function(wrap_pyfunction!(generate_pair, m)?)?;
    m.add_function(wrap_pyfunction!(generate_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_pair_same_blank, m)?)?;
    m.add_function(wrap_pyfunction!(generate_pairs_same_blank, m)?)?;
    m.add_function(wrap_pyfunction!(generate_pair_with_backtracking, m)?)?;
    m.add_function(wrap_pyfunction!(generate_pairs_with_backtracking, m)?)?;
    m.add_function(wrap_pyfunction!(generate_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(compare_datasets, m)?)?;
    m.add_function(wrap_pyfunction!(compare_datasets_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(pretty_print_state, m)?)?;
    m.add_function(wrap_pyfunction!(pretty_print_pair, m)?)?;
    Ok(())
}
