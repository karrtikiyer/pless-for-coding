import ast
import builtins
import hashlib
from itertools import combinations

import zss

BUILTIN_NAMES = set(dir(builtins))


class _NameNormalizer(ast.NodeTransformer):
    """Replace user-defined names with positional placeholders (_v0, _v1, ...)."""

    def __init__(self):
        self._mapping: dict[str, str] = {}
        self._counter = 0

    def _normalize(self, name: str) -> str:
        if name in BUILTIN_NAMES:
            return name
        if name not in self._mapping:
            self._mapping[name] = f"_v{self._counter}"
            self._counter += 1
        return self._mapping[name]

    def visit_Name(self, node):
        self.generic_visit(node)
        node.id = self._normalize(node.id)
        return node

    def visit_FunctionDef(self, node):
        node.name = self._normalize(node.name)
        # Normalize argument names
        for arg in node.args.args:
            arg.arg = self._normalize(arg.arg)
        for arg in node.args.posonlyargs:
            arg.arg = self._normalize(arg.arg)
        for arg in node.args.kwonlyargs:
            arg.arg = self._normalize(arg.arg)
        if node.args.vararg:
            node.args.vararg.arg = self._normalize(node.args.vararg.arg)
        if node.args.kwarg:
            node.args.kwarg.arg = self._normalize(node.args.kwarg.arg)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        node.name = self._normalize(node.name)
        self.generic_visit(node)
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname:
                alias.asname = self._normalize(alias.asname)
        return node

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.asname:
                alias.asname = self._normalize(alias.asname)
        return node

    def visit_keyword(self, node):
        # Preserve keyword argument names (they must match function signatures)
        self.generic_visit(node)
        return node

    def visit_attribute(self, node):
        # Preserve attribute names (they refer to external APIs)
        self.generic_visit(node)
        return node


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove string expression statements (docstrings) from the AST."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, (ast.Constant,))
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def ast_fingerprint(code: str) -> str | None:
    """Return a hash fingerprint of the normalized AST, or None on parse failure."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    tree = _strip_docstrings(tree)
    normalizer = _NameNormalizer()
    tree = normalizer.visit(tree)
    dumped = ast.dump(tree)
    return hashlib.sha256(dumped.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pairwise AST edit distance (Zhang-Shasha via zss)
# ---------------------------------------------------------------------------

def _node_label(node: ast.AST) -> str:
    """Create a descriptive label for an AST node.

    Includes the node type and key distinguishing attributes (operator kind,
    constant value, etc.) so that structurally meaningful differences produce
    non-zero edit costs.
    """
    parts = [type(node).__name__]

    # Operators and boolean ops
    if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
                         ast.Pow, ast.FloorDiv, ast.BitAnd, ast.BitOr,
                         ast.BitXor, ast.LShift, ast.RShift, ast.MatMult)):
        pass  # type name alone is sufficient
    elif isinstance(node, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
                           ast.Is, ast.IsNot, ast.In, ast.NotIn)):
        pass
    elif isinstance(node, (ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub)):
        pass
    elif isinstance(node, ast.Constant):
        # Include value for small constants so e.g. `0` vs `1` differ
        val = node.value
        if isinstance(val, (int, float, bool, type(None))):
            parts.append(repr(val))
        elif isinstance(val, str) and len(val) <= 20:
            parts.append(repr(val))
        else:
            parts.append(type(val).__name__)
    elif isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.FunctionDef):
        parts.append(node.name)
    elif isinstance(node, ast.arg):
        parts.append(node.arg)
    elif isinstance(node, ast.Attribute):
        parts.append(node.attr)
    elif isinstance(node, ast.keyword):
        if node.arg:
            parts.append(node.arg)

    return ":".join(parts)


def _ast_to_zss_tree(node: ast.AST) -> zss.Node:
    """Convert a Python AST node into a zss.Node tree for edit distance."""
    zss_node = zss.Node(_node_label(node))
    for child in ast.iter_child_nodes(node):
        zss_node.addkid(_ast_to_zss_tree(child))
    return zss_node


def _count_nodes(node: zss.Node) -> int:
    """Count total nodes in a zss tree."""
    return 1 + sum(_count_nodes(c) for c in node.children)


def _normalize_ast(code: str) -> ast.AST | None:
    """Parse, strip docstrings, normalize names. Returns None on failure."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    tree = _strip_docstrings(tree)
    normalizer = _NameNormalizer()
    tree = normalizer.visit(tree)
    return tree


def normalized_ast_edit_distance(code_a: str, code_b: str) -> float | None:
    """Compute normalized AST edit distance between two code snippets.

    Returns a float in [0, 1] where 0 = identical ASTs and 1 = maximally different.
    Returns None if either snippet fails to parse.
    """
    tree_a = _normalize_ast(code_a)
    tree_b = _normalize_ast(code_b)
    if tree_a is None or tree_b is None:
        return None

    zss_a = _ast_to_zss_tree(tree_a)
    zss_b = _ast_to_zss_tree(tree_b)

    dist = zss.simple_distance(zss_a, zss_b)

    # Normalize by max tree size (upper bound on edit distance)
    max_size = max(_count_nodes(zss_a), _count_nodes(zss_b))
    if max_size == 0:
        return 0.0
    return min(dist / max_size, 1.0)


def pairwise_diversity(
    codes: list[str],
    cluster_threshold: float = 0.8,
) -> dict:
    """Compute pairwise AST edit distances for a list of code snippets.

    Args:
        codes: List of code strings (should be correct solutions).
        cluster_threshold: Distance threshold for clustering. Two solutions
            are in the same cluster if their distance < (1 - cluster_threshold).
            Default 0.8 means solutions with similarity >= 0.8 are clustered.

    Returns:
        dict with keys:
        - mean_distance: average pairwise normalized edit distance
        - num_clusters: number of clusters at the given threshold
        - distances: list of all pairwise distances
    """
    if len(codes) < 2:
        return {
            "mean_distance": 0.0,
            "num_clusters": len(codes),
            "distances": [],
        }

    # Pre-compute normalized ASTs and zss trees, deduplicating by AST fingerprint
    # to avoid expensive edit distance computation on identical trees
    trees = []
    tree_sizes = []
    fp_to_idx: dict[str, int] = {}  # fingerprint → index in trees[]
    code_to_tree_idx: list[int] = []  # maps each valid code → index in trees[]

    for code in codes:
        tree = _normalize_ast(code)
        if tree is None:
            continue
        fp = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
        if fp not in fp_to_idx:
            fp_to_idx[fp] = len(trees)
            zss_tree = _ast_to_zss_tree(tree)
            trees.append(zss_tree)
            tree_sizes.append(_count_nodes(zss_tree))
        code_to_tree_idx.append(fp_to_idx[fp])

    if len(code_to_tree_idx) < 2:
        return {
            "mean_distance": 0.0,
            "num_clusters": len(code_to_tree_idx),
            "distances": [],
        }

    # Compute pairwise distances between unique trees only
    n_unique = len(trees)
    unique_dist: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(n_unique), 2):
        dist = zss.simple_distance(trees[i], trees[j])
        max_size = max(tree_sizes[i], tree_sizes[j])
        norm_dist = min(dist / max_size, 1.0) if max_size > 0 else 0.0
        unique_dist[(i, j)] = norm_dist
        unique_dist[(j, i)] = norm_dist

    # Expand back to all pairwise distances (including duplicates)
    n = len(code_to_tree_idx)
    all_distances = []
    dist_matrix = [[0.0] * n for _ in range(n)]

    for i, j in combinations(range(n), 2):
        ti, tj = code_to_tree_idx[i], code_to_tree_idx[j]
        if ti == tj:
            norm_dist = 0.0
        else:
            norm_dist = unique_dist[(ti, tj)]
        dist_matrix[i][j] = norm_dist
        dist_matrix[j][i] = norm_dist
        all_distances.append(norm_dist)

    mean_distance = float(sum(all_distances) / len(all_distances))

    # Simple single-linkage clustering
    # Two solutions merge if distance < merge_threshold
    merge_threshold = 1.0 - cluster_threshold
    cluster_ids = list(range(n))

    def find(x):
        while cluster_ids[x] != x:
            cluster_ids[x] = cluster_ids[cluster_ids[x]]
            x = cluster_ids[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            cluster_ids[ra] = rb

    for i, j in combinations(range(n), 2):
        if dist_matrix[i][j] < merge_threshold:
            union(i, j)

    num_clusters = len(set(find(i) for i in range(n)))

    return {
        "mean_distance": round(mean_distance, 4),
        "num_clusters": num_clusters,
        "distances": [round(float(d), 4) for d in all_distances],
    }
