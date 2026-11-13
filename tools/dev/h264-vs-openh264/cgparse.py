"""Parse a callgrind file (Ir only) into self costs and call edges."""
import re, collections, subprocess, bisect

def parse(path, symfile=None):
    names = {}
    selfc = collections.Counter()
    edges = collections.defaultdict(lambda: [0, 0])  # (caller, callee) -> [calls, inclusive]
    cur = None; cfn = None; pending_calls = None
    sym = None
    if symfile:
        addrs, labels = [], []
        for line in open(symfile):
            p = line.split()
            if len(p) == 3:
                try: addrs.append(int(p[0], 16)); labels.append(p[2])
                except ValueError: pass
        sym = (addrs, labels)
    def resolve(n):
        m = re.match(r'0x([0-9a-f]+)$', n)
        if m and sym:
            a = int(m.group(1), 16); i = bisect.bisect_right(sym[0], a) - 1
            if i >= 0: return sym[1][i]
        return n
    for line in open(path):
        line = line.rstrip('\n')
        m = re.match(r'(c?fn)=\((\d+)\)(?: (.*))?$', line)
        if m:
            kind, i, name = m.groups()
            if name is not None: names[i] = resolve(name)
            if kind == 'fn': cur = i
            else: cfn = i
            continue
        if line.startswith('calls='):
            pending_calls = int(line.split('=')[1].split()[0]); continue
        if line and (line[0].isdigit() or line[0] in '+-*'):
            parts = line.split()
            if len(parts) >= 2:
                cost = int(parts[1])
                if pending_calls is not None:
                    e = edges[(cur, cfn)]; e[0] += pending_calls; e[1] += cost
                    pending_calls = None
                else:
                    selfc[cur] += cost
    return names, selfc, edges

def parse_files(path):
    """Self cost per (function, source file of the inlined line)."""
    names, files = {}, {}
    selfc = collections.Counter()
    cur = None; fl = None; cur_file = None; pending = False
    for line in open(path):
        line = line.rstrip('\n')
        m = re.match(r'(fl|fi|fe)=\((\d+)\)(?: (.*))?$', line)
        if m:
            kind, i, name = m.groups()
            if name is not None: files[i] = name
            if kind == 'fl': fl = i; cur_file = i
            else: cur_file = i
            continue
        m = re.match(r'(c?fn)=\((\d+)\)(?: (.*))?$', line)
        if m:
            kind, i, name = m.groups()
            if name is not None: names[i] = name
            if kind == 'fn': cur = i; cur_file = fl
            continue
        if line.startswith('calls='):
            pending = True; continue
        if line.startswith(('cfl=', 'cfi=', 'cob=', 'ob=')):
            continue
        if line and (line[0].isdigit() or line[0] in '+-*'):
            p = line.split()
            if len(p) >= 2:
                if pending: pending = False
                else: selfc[(cur, cur_file)] += int(p[1])
    return names, files, selfc
