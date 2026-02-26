import ast
import builtins
import hashlib

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
