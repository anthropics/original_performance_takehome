import argparse
import heapq
import importlib
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from problem import SLOT_LIMITS


def build_dependencies(ops):
    last_write = {}
    last_read = {}
    deps_list = []
    users = [[] for _ in ops]

    for i, op in enumerate(ops):
        deps = {}
        for src in op.srcs:
            if src in last_write:
                dep = last_write[src]
                deps[dep] = max(deps.get(dep, 0), 1)
        for dest in op.dests:
            if dest in last_write:
                dep = last_write[dest]
                deps[dep] = max(deps.get(dep, 0), 1)
            if dest in last_read:
                dep = last_read[dest]
                deps[dep] = max(deps.get(dep, 0), 0)
        for dep, latency in deps.items():
            users[dep].append((i, latency))
        deps_list.append(list(deps.items()))
        for src in op.srcs:
            last_read[src] = i
        for dest in op.dests:
            last_write[dest] = i

    return deps_list, users


def critical_path(deps_list):
    n_ops = len(deps_list)
    best_len = [0] * n_ops
    pred = [-1] * n_ops

    for i in range(n_ops):
        max_len = 0
        max_pred = -1
        for dep, latency in deps_list[i]:
            cand = best_len[dep] + latency
            if cand > max_len:
                max_len = cand
                max_pred = dep
        best_len[i] = max_len
        pred[i] = max_pred

    end = max(range(n_ops), key=lambda i: best_len[i], default=None)
    if end is None:
        return 0, []

    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = pred[cur]
    path.reverse()
    return best_len[end] + 1, path


def simulate_schedule(ops, users, start_offsets):
    n_ops = len(ops)
    deps_count = [0] * n_ops
    earliest = [0] * n_ops
    if start_offsets:
        for op_id, offset in start_offsets.items():
            if 0 <= op_id < n_ops:
                earliest[op_id] = max(earliest[op_id], offset)

    for op_id, deps in enumerate(users):
        for user, _ in deps:
            deps_count[user] += 1

    ready = []
    ready_time = [None] * n_ops
    for i, count in enumerate(deps_count):
        if count == 0:
            heapq.heappush(ready, (earliest[i], i))
            ready_time[i] = earliest[i]

    scheduled_cycle = [-1] * n_ops
    scheduled = 0
    cycle = 0
    engine_limits = {k: v for k, v in SLOT_LIMITS.items() if k != "debug"}
    users_count = [len(u) for u in users]
    engine_priority = {"flow": 0, "load": 1, "store": 2, "valu": 3, "alu": 4}

    while scheduled < n_ops:
        if not ready:
            cycle += 1
            continue

        if ready[0][0] > cycle:
            cycle = ready[0][0]

        slots_used = {k: 0 for k in engine_limits}
        available = []
        while ready and ready[0][0] <= cycle:
            available.append(heapq.heappop(ready)[1])

        def pick_op_index():
            best_idx = None
            best_score = None
            for idx, op_id in enumerate(available):
                op = ops[op_id]
                if slots_used[op.engine] >= engine_limits[op.engine]:
                    continue
                score = (engine_priority.get(op.engine, 99), -users_count[op_id])
                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = idx
            return best_idx

        while available:
            pick_idx = pick_op_index()
            if pick_idx is None:
                break
            op_id = available.pop(pick_idx)
            op = ops[op_id]
            scheduled_cycle[op_id] = cycle
            slots_used[op.engine] += 1
            scheduled += 1
            for user, latency in users[op_id]:
                deps_count[user] -= 1
                earliest[user] = max(earliest[user], cycle + latency)
                if deps_count[user] == 0:
                    ready_time[user] = earliest[user]
                    if earliest[user] <= cycle:
                        available.append(user)
                    else:
                        heapq.heappush(ready, (earliest[user], user))

        for op_id in available:
            heapq.heappush(ready, (earliest[op_id], op_id))

        cycle += 1

    return scheduled_cycle, ready_time


def op_label(op):
    if not op.slot:
        return op.engine
    op_name = op.slot[0]
    return f"{op.engine}:{op_name}"


def inspect_ops(module_name, forest_height, rounds, batch_size, batch_index, include_init, top_n):
    module = importlib.import_module(module_name)
    kb = module.KernelBuilder()
    kb.collect_ops = True
    n_nodes = 2 ** (forest_height + 1) - 1
    kb.build_kernel(forest_height, n_nodes, batch_size, rounds)

    if not hasattr(kb, "debug_ops"):
        raise SystemExit("KernelBuilder did not capture ops. Ensure the optimized path ran.")

    debug_ops = kb.debug_ops
    batch_ops = debug_ops["batches"]
    if not batch_ops:
        raise SystemExit("No batch ops captured.")

    if batch_index < 0 or batch_index >= len(batch_ops):
        raise SystemExit(f"batch-index {batch_index} out of range (0-{len(batch_ops)-1})")

    ops = batch_ops[batch_index]["ops"]
    start_offsets = batch_ops[batch_index].get("start_offsets", {})
    if include_init:
        ops = debug_ops["init"] + ops

    deps_list, users = build_dependencies(ops)
    critical_len, path = critical_path(deps_list)
    scheduled_cycle, ready_time = simulate_schedule(ops, users, start_offsets)

    waits = []
    for op_id, sched in enumerate(scheduled_cycle):
        if sched < 0 or ready_time[op_id] is None:
            continue
        waits.append((sched - ready_time[op_id], op_id))

    waits.sort(reverse=True)

    print(f"Total ops: {len(ops)}")
    print(f"Dependency edges: {sum(len(d) for d in deps_list)}")
    print(f"Critical path length (cycles): {critical_len}")
    if scheduled_cycle:
        print(f"Scheduled length (cycles): {max(scheduled_cycle) + 1}")
    print()

    print(f"Top {min(top_n, len(waits))} waits (bundling candidates):")
    for wait, op_id in waits[:top_n]:
        op = ops[op_id]
        print(f"  wait={wait:4d} op={op_label(op)} slot={op.slot}")

    print()
    print(f"Critical path (first {min(top_n, len(path))} ops):")
    for op_id in path[:top_n]:
        op = ops[op_id]
        print(f"  op={op_label(op)} slot={op.slot}")


def main():
    parser = argparse.ArgumentParser(description="Inspect kernel op dependency graph.")
    parser.add_argument("--kernel", default="perf_takehome", help="Kernel module to inspect")
    parser.add_argument("--forest-height", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--include-init", action="store_true", help="Include init ops")
    parser.add_argument("--top", type=int, default=20, help="Rows to print")

    args = parser.parse_args()
    inspect_ops(
        args.kernel,
        args.forest_height,
        args.rounds,
        args.batch_size,
        args.batch_index,
        args.include_init,
        args.top,
    )


if __name__ == "__main__":
    main()
