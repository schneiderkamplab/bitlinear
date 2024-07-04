import click
import signal
import time

# logging
base = time.time()
ERROR = 0
PROGRESS = 1
INFO = 2
DETAIL = 3
DEBUG = 4
_verbosity = INFO
def log(verbosity, *args, **kwargs):
    global _verbosity
    if verbosity <= _verbosity:
        click.echo(f"[{time.time()-base:.2f}] {' '.join(str(arg) for arg in args)}", **kwargs)
def get_verbosity():
    return _verbosity
def set_verbosity(verbosity):
    global _verbosity
    _verbosity = verbosity

# chunking
def chunker(seq, size):
    return (seq[idx:idx+size] for idx in range(0,len(seq),size))

# signal handling
exit_flag = False
def graceful_exit():
    global exit_flag
    exit_flag = True
def signal_handler(sig, frame):
    graceful_exit()
    log(ERROR, "Ctrl-C pressed: exiting gracefully")
def install_signal_handler():
    signal.signal(signal.SIGINT, signal_handler)
def should_exit():
    return exit_flag

# file handling
def load_bitlinear(file_name):
    data = []
    n = None
    with open(file_name, "rt") as f:
        for line in f:
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                _, bitlinear, n, m = line.split()
                assert bitlinear == "bitlinear"
                n, m = int(n), int(m)
                continue
            assert n is not None
            row = [int(x) for x in line.split()]
            assert len(row) == n
            data.append(row)
    assert len(data) == m
    return data
def save_bitlinear(file_name, data, comments=None):
    assert data and all(len(x) == len(data[0]) for x in data)
    m, n = len(data), len(data[0])
    with open(file_name, "wt") as f:
        if comments is not None:
            for comment in comments.split("\n"):
                f.write(f"c {comment}\n")
        f.write(f"p bitlinear {n} {m}\n")
        f.write('\n'.join(' '.join(str(y) for y in x) for x in data))
def load_slp(file_name):
    program = []
    outputs = []
    num_inputs = None
    outputs = None
    with open(file_name, "rt") as f:
        for line in f:
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                _, slp, num_inputs, len_program, num_outputs = line.split()
                assert slp == "slp"
                num_inputs, len_program, num_outputs = int(num_inputs), int(len_program), int(num_outputs)
                continue
            if num_outputs is not None:
                if len(program) < len_program:
                    x, y, z = [int(x) for x in line.split()]
                    program.append(((x, y), z))
                elif outputs is None:
                    outputs = [int(x) for x in line.split()]
                else:
                    assert False
    assert len(program) == len_program
    assert len(outputs) == num_outputs
    return num_inputs, program, outputs
def save_slp(file_name, num_inputs, program, outputs, comments=None):
    with open(file_name, "wt") as f:
        if comments is not None:
            for comment in comments.split("\n"):
                f.write(f"c {comment}\n")
        f.write(f"p slp {num_inputs} {len(program)} {len(outputs)}\n")
        for (x, y), z in program:
            f.write(f"{x} {y} {z}\n")
        f.write(" ".join(str(x) for x in outputs)+"\n")