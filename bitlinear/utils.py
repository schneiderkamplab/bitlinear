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
