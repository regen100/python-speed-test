python -m timeit -s "import test" "test.python()"
python -m timeit -s "import test" "test.numpy()"
python -m timeit -s "import test" "test.weave()"
python -m timeit -s "import test" "test.cython()"
python -m timeit -s "import test" "test.cythonomp()"
