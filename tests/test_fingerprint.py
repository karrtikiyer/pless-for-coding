from bench.eval.fingerprint import ast_fingerprint


def test_syntax_error_returns_none():
    assert ast_fingerprint("def f(") is None


def test_identical_code_same_fingerprint():
    code = "def add(a, b): return a + b"
    assert ast_fingerprint(code) == ast_fingerprint(code)


def test_different_variable_names_same_structure():
    code1 = "def add(a, b): return a + b"
    code2 = "def plus(x, y): return x + y"
    assert ast_fingerprint(code1) == ast_fingerprint(code2)


def test_different_structure_different_fingerprint():
    code1 = "def f(a, b): return a + b"
    code2 = "def f(a, b): return a * b"
    assert ast_fingerprint(code1) != ast_fingerprint(code2)


def test_docstring_stripping():
    code_with_doc = '''def f(x):
    """This does something."""
    return x + 1'''
    code_without_doc = "def f(x):\n    return x + 1"
    assert ast_fingerprint(code_with_doc) == ast_fingerprint(code_without_doc)


def test_builtin_names_preserved():
    """Builtin names like len, print, range should NOT be normalized."""
    code1 = "def f(x): return len(x)"
    code2 = "def g(y): return len(y)"
    assert ast_fingerprint(code1) == ast_fingerprint(code2)


def test_different_builtins_different_fingerprint():
    code1 = "def f(x): return len(x)"
    code2 = "def f(x): return sum(x)"
    assert ast_fingerprint(code1) != ast_fingerprint(code2)


def test_argument_order_matters():
    code1 = "def f(a, b): return a - b"
    code2 = "def f(a, b): return b - a"
    # These are structurally different: first arg minus second vs second minus first
    # After normalization both args become _v1, _v2 but the order in subtraction differs
    assert ast_fingerprint(code1) != ast_fingerprint(code2)


def test_multiline_functions():
    code1 = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    code2 = """
def fact(x):
    if x <= 1:
        return 1
    return x * fact(x - 1)
"""
    assert ast_fingerprint(code1) == ast_fingerprint(code2)


def test_class_docstring_stripped():
    code_with = '''
class Foo:
    """A class."""
    def method(self):
        return 1
'''
    code_without = '''
class Bar:
    def method(self):
        return 1
'''
    # Class name differs but gets normalized, docstring stripped
    assert ast_fingerprint(code_with) == ast_fingerprint(code_without)


def test_empty_code():
    fp = ast_fingerprint("")
    assert fp is not None  # Empty module is valid


def test_returns_hex_string():
    fp = ast_fingerprint("x = 1")
    assert fp is not None
    assert all(c in "0123456789abcdef" for c in fp)
    assert len(fp) == 64  # SHA-256 hex digest


def test_import_normalization():
    """Import aliases should be normalized, module names preserved."""
    code1 = "import math as m\nx = m.sqrt(4)"
    code2 = "import math as mymath\ny = mymath.sqrt(4)"
    assert ast_fingerprint(code1) == ast_fingerprint(code2)
