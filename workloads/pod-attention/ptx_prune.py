#!/usr/bin/env python3
"""Conservative POD PTX dead-function removal; never edits kernel/function bodies.

NVCC --keep-device-functions preserves the typed ABI but also emits tens of
thousands of unused template helpers. Keep every entry, external declaration,
module data declaration and the complete named-reference closure. An indirect
call in retained code is deliberately unsupported, rather than guessed safe.
No GPU work, compiler invocation or content identifiers are used here.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

COMMENTS_OR_STRINGS = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', re.S)
START = re.compile(r'(?m)^[ \t]*(?:(?:\.visible|\.weak|\.extern)\s+)*\.(func|entry)\b')
IDENT = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*')
CALL = re.compile(r'\bcall(?:\.uni)?\s+(?:\([^;]*?\)\s*,\s*)?([%A-Za-z_$][A-Za-z0-9_$]*)\s*,', re.S)


def direct_calls(body, names):
    """Reject unknown/indirect calls in the supported fixed NVCC PTX form."""
    bare_registers = set()
    for declaration in re.finditer(r'\.reg\b([^;]*);', body):
        bare_registers.update(m.group(1) for m in re.finditer(
            r'(?<![%A-Za-z0-9_$.])([A-Za-z_$][A-Za-z0-9_$]*)', declaration.group(1)))
    calls = list(CALL.finditer(body))
    if len(calls) != len(re.findall(r'\bcall(?:\.uni)?\b', body)):
        raise ValueError('unrecognized call syntax in retained code')
    for call in calls:
        target = call.group(1)
        if target.startswith('%') or target in bare_registers:
            raise ValueError('retained indirect call requires a proven target set')
        if target not in names:
            raise ValueError('retained call has no parsed declaration: ' + target)
    return calls


def masks(source):
    """Same-offset lexical masks; keep strings for conservative symbol references."""
    def spaces(match, keep_strings):
        text = match.group()
        if keep_strings and text.startswith('"'):
            return text
        return re.sub(r'[^\n]', ' ', text)
    return (COMMENTS_OR_STRINGS.sub(lambda m: spaces(m, False), source),
            COMMENTS_OR_STRINGS.sub(lambda m: spaces(m, True), source))


def balanced_end(text, begin, opening, closing, limit):
    if text[begin] != opening:
        raise ValueError("invalid PTX delimiter")
    depth = 0
    for match in re.finditer('[' + re.escape(opening + closing) + ']', text[begin:limit]):
        depth += 1 if match.group() == opening else -1
        if depth == 0:
            return begin + match.end()
    raise ValueError("unterminated PTX signature/body")


def parse_functions(source):
    syntax, references = masks(source)
    starts = list(START.finditer(syntax))
    records = []
    for index, match in enumerate(starts):
        limit = starts[index + 1].start() if index + 1 < len(starts) else len(syntax)
        pos = match.end()
        while pos < limit and syntax[pos].isspace():
            pos += 1
        if pos < limit and syntax[pos] == '(':
            pos = balanced_end(syntax, pos, '(', ')', limit)
            while pos < limit and syntax[pos].isspace():
                pos += 1
        name = IDENT.match(syntax, pos)
        if not name:
            raise ValueError("unrecognized PTX function name")
        tail = syntax[name.end():limit]
        delimiter = re.search(r'[;{]', tail)
        if not delimiter:
            raise ValueError("missing PTX function declaration/body delimiter")
        begin = name.end() + delimiter.start()
        definition = syntax[begin] == '{'
        end = balanced_end(syntax, begin, '{', '}', limit) if definition else begin + 1
        records.append(dict(name=name.group(), kind=match.group(1), start=match.start(),
                            end=end, body_start=begin if definition else None,
                            external='.extern' in syntax[match.start():match.end()]))
    # Every .func/.entry must be accounted for at module scope; do not silently
    # ignore an unfamiliar linkage spelling or nested declaration syntax.
    leftovers, previous = [], 0
    for record in records:
        leftovers.append(syntax[previous:record['start']])
        previous = record['end']
    leftovers.append(syntax[previous:])
    if re.search(r'\.(?:func|entry)\b', ''.join(leftovers)):
        raise ValueError("unrecognized PTX function directive outside parsed records")
    if not records or not any(r['kind'] == 'entry' for r in records):
        raise ValueError("PTX has no real entry points")
    return records, syntax, references


def prune(source):
    records, syntax, references = parse_functions(source)
    by_name = defaultdict(list)
    for record in records:
        by_name[record['name']].append(record)
    names = set(by_name)
    module_parts, previous = [], 0
    for record in records:
        module_parts.append(references[previous:record['start']])
        previous = record['end']
    module_parts.append(references[previous:])
    entries = {r['name'] for r in records if r['kind'] == 'entry'}
    retained = entries | {r['name'] for r in records if r['external']}
    retained |= set(IDENT.findall(''.join(module_parts))) & names
    pending = list(retained)
    while pending:
        name = pending.pop()
        for record in by_name[name]:
            begin, end = record['start'], record['end']
            body = syntax[begin:end]
            direct_calls(body, names)
            # More conservative than a call graph: captures mov/function-address
            # operands, .calltargets, aliases and data initializers as well.
            needed = (set(IDENT.findall(references[begin:end])) & names) - retained
            retained.update(needed)
            pending.extend(needed)
    output, previous = [], 0
    for record in records:
        output.append(source[previous:record['start']])
        if record['name'] in retained:
            output.append(source[record['start']:record['end']])
        previous = record['end']
    output.append(source[previous:])
    output = ''.join(output)
    inventory = dict(source_bytes=len(source.encode()), retained_bytes=len(output.encode()),
                     retained_response_json_bytes=len(json.dumps({'output_ptx': output, 'modified': True},
                                                                  separators=(',', ':')).encode()),
                     entry_names=sorted(entries), retained_function_names=sorted(retained),
                     removed_function_names=sorted(names - retained),
                     source_records=len(records), retained_records=sum(r['name'] in retained for r in records),
                     method='all entries + extern + all module/named-reference reachability; indirect calls rejected')
    return output, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--source-output', type=Path)
    args = parser.parse_args()
    if len({bool(args.output), bool(args.inventory), bool(args.source_output)}) != 1:
        parser.error('--source-output, --output and --inventory must be supplied together')
    if any(path and path.exists() for path in (args.output, args.inventory, args.source_output)):
        parser.error('refusing to replace an existing output/inventory')
    source = sys.stdin.read()
    versions = list(re.finditer(r'(?m)^\.version\s+', source))
    if len(versions) != 1:
        parser.error('expected exactly one extracted official PTX module')
    source = source[versions[0].start():]  # Remove cuobjdump's non-PTX banner only.
    output, inventory = prune(source)
    if args.output:
        with args.source_output.open('x') as out:
            out.write(source)
        with args.output.open('x') as out:
            out.write(output)
        with args.inventory.open('x') as out:
            json.dump(inventory, out, indent=2)
            out.write('\n')
    print(json.dumps({key: (len(value) if isinstance(value, list) else value)
                      for key, value in inventory.items()}, indent=2))


if __name__ == '__main__':
    main()
