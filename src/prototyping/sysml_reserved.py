"""Reserved words of the KerML and SysML v2 textual notations.

Verified against the parser: each word is rejected as a declared name in at
least one of the positions the pipeline renders (attribute, action, item,
port, part). `nonunique` and `ordered` parse as attribute names but not as
definition names; the set keeps them because callers do not track which
position a name will reach. Every module that turns an externally chosen
name into SysML text checks this set rather than keeping its own list.
"""
from __future__ import annotations

SYSML_RESERVED_WORDS: frozenset[str] = frozenset("""
about abstract accept action actor after alias all allocate allocation
analysis and as assert assign assume at attribute bind binding by calc case
comment concern connect connection constraint crosses decide def default
defined dependency derived do doc else end entry enum event exhibit exit
expose false filter first flow for fork frame from hastype if implies import
in include individual inout interface istype item join language library
locale loop merge message meta metadata new nonunique not null objective
occurrence of or ordered out package parallel part perform port private
protected public redefines ref references rendering rep require requirement
return satisfy send snapshot specializes stakeholder standard state subject
subsets succession terminate then timeslice to transition true until use
variant variation verification verify via view viewpoint when while xor
""".split())
