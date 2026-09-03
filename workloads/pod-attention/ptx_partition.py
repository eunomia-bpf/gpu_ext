"""Prove whether a POD PTX module has independent launchable entry points.

Only the existing fixed NVCC PTX form is supported. Module-shared storage is
CTA-local. Every other module data object must be unreferenced by retained
code; function/entry address identity and aliases are deliberately rejected.
"""
import re
from collections import Counter

from ptx_prune import IDENT, direct_calls, parse_functions

DATA = re.compile(r'(?m)^[ \t]*(?:(?:\.visible|\.weak|\.extern)\s+)*\.'
                  r'(global|const|shared|tex|surf)\b([^;]*);')


def independent_entries(source):
    records, syntax, references = parse_functions(source)
    outside, previous = [], 0
    outside_refs = []
    for record in records:
        outside.append(syntax[previous:record['start']])
        outside_refs.append(references[previous:record['start']])
        previous = record['end']
    outside.append(syntax[previous:])
    outside_refs.append(references[previous:])
    module = ''.join(outside)
    names = {r['name'] for r in records}
    entries = {r['name'] for r in records if r['kind'] == 'entry'}
    defined = {r['name'] for r in records if r['body_start'] is not None}
    if (names - entries) - defined:
        raise ValueError('external helper without a body has unproven state/identity semantics')
    if set(IDENT.findall(''.join(outside_refs))) & names:
        raise ValueError('module-scope function address/alias prevents independent modules')
    data = {}
    declarations = list(DATA.finditer(module))
    if len(declarations) != len(re.findall(r'\.(?:global|const|shared|tex|surf)\b', module)):
        raise ValueError('unrecognized module data declaration')
    for declaration in declarations:
        prefix = declaration.group(2).split('=', 1)[0].split('[', 1)[0]
        if ',' in prefix:
            raise ValueError('unsupported multi-object declaration')
        identifiers = IDENT.findall(prefix)
        if not identifiers:
            raise ValueError('unrecognized module data object name')
        data[identifiers[-1]] = declaration.group(1)
    referenced_data = set()
    for record in records:
        if record['body_start'] is None:
            continue
        begin, end = record['body_start'], record['end']
        body_syntax, body_refs = syntax[begin:end], references[begin:end]
        identifiers = set(IDENT.findall(body_refs))
        referenced_data |= identifiers & data.keys()
        forbidden = {name for name in identifiers & data.keys() if data[name] != 'shared'}
        if forbidden:
            raise ValueError('retained code references module-lifetime data; cannot duplicate it')
        # Remove only direct-call target tokens, not arguments or other uses of
        # that same function's address. The remaining references must be empty.
        chunks, previous = [], 0
        for call in direct_calls(body_syntax, names):
            target = call.group(1)
            if target.startswith('%') or target not in names or target in entries:
                raise ValueError('non-helper direct call prevents independent modules')
            chunks.append(body_refs[previous:call.start(1)])
            previous = call.end(1)
        chunks.append(body_refs[previous:])
        if set(IDENT.findall(''.join(chunks))) & names:
            raise ValueError('retained function/entry address identity prevents independent modules')
    return records, dict(entry_names=sorted(entries),
        data_objects=[{'name': name, 'space': space, 'referenced': name in referenced_data}
                      for name, space in sorted(data.items())],
        data_counts=dict(Counter(data.values())),
        referenced_data_counts=dict(Counter(data[name] for name in referenced_data)),
        module_lifetime_data_references=0, function_address_references=0,
        helper_definitions=len(names - entries), unknown_external_helpers=0)


def partition_ptx(source):
    """Two complete, independent entry groups; helpers/data remain unmodified."""
    records, proof = independent_entries(source)
    weights = Counter()
    for record in records:
        if record['kind'] == 'entry':
            weights[record['name']] += record['end'] - record['start']
    if len(weights) < 2:
        raise ValueError('need at least two entries for independent transport packets')
    groups, sizes = [set(), set()], [0, 0]
    # Code-size packing only, decided before execution; never timing-driven.
    for name in sorted(weights, key=lambda name: (-weights[name], name)):
        group = 0 if sizes[0] <= sizes[1] else 1
        groups[group].add(name)
        sizes[group] += weights[name]
    packets = []
    for group in groups:
        output, previous = [], 0
        for record in records:
            output.append(source[previous:record['start']])
            if record['kind'] != 'entry' or record['name'] in group:
                output.append(source[record['start']:record['end']])
            previous = record['end']
        output.append(source[previous:])
        packets.append(''.join(output))
    if groups[0] & groups[1] or groups[0] | groups[1] != set(proof['entry_names']):
        raise ValueError('entry packet inventory lost or duplicated an official entry')
    proof['packet_entry_names'] = [sorted(group) for group in groups]
    proof['packet_bytes'] = [len(packet.encode()) for packet in packets]
    proof['all_entry_bodies_unchanged'] = True
    return packets, proof
