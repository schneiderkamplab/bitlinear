import click
import random
import torch

from ..utils import (
    DETAIL,
    INFO,
    chunker,
    install_signal_handler,
    load_bitlinear,
    get_verbosity,
    log,
    set_verbosity,
    save_slp,
)

def preprocess(a):
    return {str(k+1): ((k+1)*v).tolist() for k, v in enumerate(torch.tensor(a).transpose(1, 0))}

def prioritize(a, heuristic):
    if heuristic == "left-to-right":
        return prioritize_left_to_right(a)
    if heuristic == "random":
        return prioritize_random(a)
    if heuristic == "greedy":
        return prioritize_greedy(a)
    if heuristic == "compute":
        return prioritize_compute(a)

def prioritize_left_to_right(a):
    return [list(a.keys())[:2]]

def prioritize_random(a):
    keys = list(a.keys())
    random.shuffle(keys)
    return list(chunker(keys, 2))

def prioritize_greedy(a):
    def histogram(v):
        h = {}
        for x in v:
            h[x] = h.get(x, 0) + 1
        return h
    keys = list(a.keys())
    keys.sort(key=lambda x: (-len(x), max(histogram(a[x]).values())), reverse=True)
    return [keys[:2]]

def prioritize_compute(a):
    log(INFO, "Computing priorities")
    keys = list(a.keys())
    pairs = []
    for k1 in keys:
        for k2 in keys:
            if k1 == k2:
                continue
            vals = {((x, y) if x < y else (-y, -x)) for x, y in zip(a[k1], a[k2])}
            pairs.append((len(k1), len(k2), len(vals), k1, k2))
    pairs.sort()
    prios = [(x, y) for _, _, _, x, y in pairs[:1]]
    log(INFO, "Priorities computed", prios)
    return prios

def merge(l, r, last_used, program, use_cache, drop_zeros):
    log(DETAIL, "Merge", l, "and", r)
    n = []
    for x, y in zip(l, r):
        if drop_zeros and x == 0:
            n.append(y)
            continue
        if drop_zeros and y == 0:
            n.append(x)
            continue
        if use_cache:
            z = program.get((x, y), None)
            if z:
                n.append(z)
                continue
            z = program.get((-x, -y), None)
            if z:
                n.append(-z)
                continue
        last_used[0] += 1
        if x < 0 and y < 0:
            if use_cache:
                program[(-x, -y)] = last_used[0]
            else:
                program.append(((-x, -y), last_used[0]))
            n.append(-last_used[0])
        else:
            if use_cache:
                program[(x, y)] = last_used[0]
            else:
                program.append(((x, y), last_used[0]))
            n.append(last_used[0])
    log(DETAIL, "Result:", n)
    return n

def normalize(a, last_used, program, use_cache):
    log(DETAIL, "Normalize", a)
    n = []
    for x in a:
        if x >= 0:
            n.append(x)
            continue
        if use_cache:
            y = program.get((0, x), None)
            if y:
                n.append(y)
                continue
        last_used[0] += 1
        if use_cache:
            program[(0, x)] = last_used[0]
        else:
            program.append(((0, x), last_used[0]))
        n.append(last_used[0])
    log(DETAIL, "Result:" ,n)
    return n

@click.group()
def _optimize():
    pass
@_optimize.command()
@click.argument("matrices", type=click.Path(exists=True), nargs=-1)
@click.option("--implementation", type=click.Choice(["elimination", "naive"]), default="elimination")
@click.option("--heuristic", type=click.Choice(["left-to-right", "random", "greedy", "compute"]), default="left-to-right")
@click.option("--drop-zeros/--no-drop-zeros", default=True)
@click.option("--verbosity", default=get_verbosity(), help=f"Verbosity of the output (default: {get_verbosity()})")
def optimize(matrices, implementation, heuristic, drop_zeros, verbosity):
    #install_signal_handler()
    do_optimize(matrices, implementation, heuristic, drop_zeros, verbosity)

def do_optimize(
        matrices,
        implementation="elimination",
        heuristic="left-to-right",
        drop_zeros=True,
        verbosity=get_verbosity(),
    ):
    set_verbosity(verbosity)
    for matrix_name in matrices:
        log(INFO, "Optimizing matrix", matrix_name)
        a = load_bitlinear(matrix_name)
        orig_a = a
        m = len(a)
        log(DETAIL, "After loading:", a)
        a = preprocess(a)
        log(DETAIL, "After preprocessing:", a)
        n = len(a)
        last_used = [n]
        if implementation in ("elimination", "naive"):
            if implementation == "naive":
                heuristic = "left-to-right"
                program = []
                use_cache = False
            else:
                program = {} 
                use_cache = True
            while len(a) > 1:
                if len(a) % 100 == 0:
                    log(DETAIL, "Current length:", len(a))
                log(DETAIL, "Current state:", a)
                log(DETAIL, "Last used:", last_used)
                priorities = prioritize(a, heuristic)
                log(DETAIL, "Priorities:", priorities)
                for l, r in priorities:
                    a[f".{l}{r}"] = merge(a[l] ,a[r], last_used, program, use_cache, drop_zeros)
                    del a[l], a[r]
                log(DETAIL, "After merging:", a)
            a = normalize(next(iter(a.values())), last_used, program, use_cache)
            log(DETAIL, "After normalization:", a)
        log(DETAIL, "Program:", program)
        log(DETAIL, "Optimized matrix:", a)
        log(INFO, f"Length of program for {n}x{m}:", len(program), (n-1)*m, len(program)*100/(n-1)/m, (n-1)*m/len(program))
        program_name = f"{matrix_name.rsplit('.', maxsplit=1)[0]}.program"
        log(INFO, f"Saving to {program_name}")
        save_slp(
            program_name,
            n,
            program.items() if isinstance(program, dict) else program,
            a,
            comments=f"Optimized matrix {n}x{m} using heuristic {heuristic}:\n"+'\n'.join(' '.join(str(x) for x in row) for row in orig_a),
        )
