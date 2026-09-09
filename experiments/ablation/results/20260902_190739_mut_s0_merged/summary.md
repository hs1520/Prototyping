# Mutation study `20260902_190739_mut_s0_merged`

- source: experiments/ablation/results/20260902_133704_s3_prefix_matched/runs/NO-DSE_seed0  ·  commit `merged: 2026`
- sets: {'CONTROL': [], 'SYNTAX': [{'op': 'DOC-QUOTE', 'line': 276}, {'op': 'DOC-QUOTE', 'line': 277}, {'op': 'DOC-QUOTE', 'line': 278}, {'op': 'READONLY', 'line': 93, 'target': 'airframeObstacleSeparation : LengthValue = 0.0 [m];'}, {'op': 'READONLY', 'line': 97, 'target': 'hoverThrottleMargin : DimensionOneValue = 0.0 [percent];'}, {'op': 'C-NEGATION', 'line': 463, 'target': 'onboardSensorReportsFailure'}, {'op': 'GUARD-ATTR-MISSING', 'line': 626, 'target': 'propulsionCriticalFailure'}, {'op': 'TYPE-TYPO', 'line': 308, 'target': 'ObstacleSeparationPort -> ObstacleSeparationPor'}], 'CONNECT': [{'op': 'DROP-CONNECT', 'line': 820, 'target': 'connect perceptionSystem.obstacleSeparation to flightController.obstacleSeparation;'}, {'op': 'DROP-CONNECT', 'line': 823, 'target': 'connect powerSystem.flightDuration to flightController.flightDuration;'}, {'op': 'PORT-DIRECTION', 'line': 309, 'target': 'hoverThrottleMargin'}], 'BEHAVIOUR': [{'op': 'DROP-TRANSITION', 'line': 408, 'target': 'transition toAvoiding', 'machine': 'AvoidanceBehavior'}, {'op': 'DROP-ENTRY-ACTION', 'line': 419, 'target': 'navigateRevisedWaypointSequence', 'machine': 'WaypointModificationBehavior'}], 'ALL': [{'op': 'DOC-QUOTE', 'line': 276}, {'op': 'DOC-QUOTE', 'line': 277}, {'op': 'DOC-QUOTE', 'line': 278}, {'op': 'READONLY', 'line': 93, 'target': 'airframeObstacleSeparation : LengthValue = 0.0 [m];'}, {'op': 'READONLY', 'line': 97, 'target': 'hoverThrottleMargin : DimensionOneValue = 0.0 [percent];'}, {'op': 'C-NEGATION', 'line': 463, 'target': 'onboardSensorReportsFailure'}, {'op': 'GUARD-ATTR-MISSING', 'line': 626, 'target': 'propulsionCriticalFailure'}, {'op': 'TYPE-TYPO', 'line': 308, 'target': 'ObstacleSeparationPort -> ObstacleSeparationPor'}, {'op': 'DROP-CONNECT', 'line': 819, 'target': 'connect perceptionSystem.obstacleSeparation to flightController.obstacleSeparation;'}, {'op': 'DROP-CONNECT', 'line': 822, 'target': 'connect powerSystem.flightDuration to flightController.flightDuration;'}, {'op': 'PORT-DIRECTION', 'line': 309, 'target': 'hoverThrottleMargin'}, {'op': 'DROP-TRANSITION', 'line': 408, 'target': 'transition toAvoiding', 'machine': 'AvoidanceBehavior'}, {'op': 'DROP-ENTRY-ACTION', 'line': 419, 'target': 'navigateRevisedWaypointSequence', 'machine': 'WaypointModificationBehavior'}]}

| arm | set | ok | qualified | closure | 7/7 | reach | syntax err | iters | calls | tokens | residual defects |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FULL | CONTROL | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | — |
| NO-REFINE | CONTROL | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | — |
| NO-SURGICAL | CONTROL | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | — |
| NO-DETFIX | CONTROL | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | — |
| NO-REPAIR | CONTROL | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | — |
| FULL | SYNTAX | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | DOC-QUOTE 0/3, READONLY 0/2, C-NEGATION 0/1, GUARD-ATTR-MISSING 0/1, TYPE-TYPO 0/1 |
| NO-REFINE | SYNTAX | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | DOC-QUOTE 0/3, READONLY 0/2, C-NEGATION 0/1, GUARD-ATTR-MISSING 0/1, TYPE-TYPO 0/1 |
| NO-SURGICAL | SYNTAX | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | DOC-QUOTE 0/3, READONLY 0/2, C-NEGATION 0/1, GUARD-ATTR-MISSING 0/1, TYPE-TYPO 0/1 |
| NO-DETFIX | SYNTAX | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 4 | 15291 | DOC-QUOTE 0/3, READONLY 0/2, C-NEGATION 0/1, GUARD-ATTR-MISSING 0/1, TYPE-TYPO 0/1 |
| NO-REPAIR | SYNTAX | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | DOC-QUOTE 0/3, READONLY 0/2, C-NEGATION 0/1, GUARD-ATTR-MISSING 0/1, TYPE-TYPO 0/1 |
| FULL | CONNECT | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 2 | 1 | 10633 | DROP-CONNECT 0/2, PORT-DIRECTION 0/1 |
| NO-REFINE | CONNECT | ✓ | NOT_QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 2 | 112015 | DROP-CONNECT 0/2, PORT-DIRECTION 1/1 |
| NO-SURGICAL | CONNECT | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 2 | 1 | 18097 | DROP-CONNECT 0/2, PORT-DIRECTION 0/1 |
| NO-DETFIX | CONNECT | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 2 | 1 | 10633 | DROP-CONNECT 0/2, PORT-DIRECTION 0/1 |
| NO-REPAIR | CONNECT | ✓ | NOT_QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 2 | 112416 | DROP-CONNECT 0/2, PORT-DIRECTION 1/1 |
| FULL | BEHAVIOUR | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 2 | 1 | 17075 | DROP-TRANSITION 0/1, DROP-ENTRY-ACTION 0/1 |
| NO-REFINE | BEHAVIOUR | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | DROP-TRANSITION 0/1, DROP-ENTRY-ACTION 0/1 |
| NO-SURGICAL | BEHAVIOUR | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 2 | 1 | 17108 | DROP-TRANSITION 0/1, DROP-ENTRY-ACTION 0/1 |
| NO-DETFIX | BEHAVIOUR | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 2 | 1 | 17075 | DROP-TRANSITION 0/1, DROP-ENTRY-ACTION 0/1 |
| NO-REPAIR | BEHAVIOUR | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 0 | 0 | DROP-TRANSITION 0/1, DROP-ENTRY-ACTION 0/1 |
| FULL | ALL | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 2 | 1 | 10594 | DOC-QUOTE 0/3, READONLY 0/2, C-NEGATION 0/1, GUARD-ATTR-MISSING 0/1, TYPE-TYPO 0/1, DROP-CONNECT 0/2, PORT-DIRECTION 0/1, DROP-TRANSITION 0/1, DROP-ENTRY-ACTION 0/1 |
| NO-REFINE | ALL | ✓ | NOT_QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 2 | 112408 | DOC-QUOTE 0/3, READONLY 0/2, C-NEGATION 0/1, GUARD-ATTR-MISSING 0/1, TYPE-TYPO 0/1, DROP-CONNECT 0/2, PORT-DIRECTION 1/1, DROP-TRANSITION 0/1, DROP-ENTRY-ACTION 0/1 |
| NO-SURGICAL | ALL | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 2 | 1 | 17772 | DOC-QUOTE 0/3, READONLY 0/2, C-NEGATION 0/1, GUARD-ATTR-MISSING 0/1, TYPE-TYPO 0/1, DROP-CONNECT 0/2, PORT-DIRECTION 0/1, DROP-TRANSITION 0/1, DROP-ENTRY-ACTION 0/1 |
| NO-DETFIX | ALL | ✓ | QUALIFIED | CLOSED | 7 | 0.92 | 0 | 2 | 6 | 28054 | DOC-QUOTE 0/3, READONLY 0/2, C-NEGATION 0/1, GUARD-ATTR-MISSING 0/1, TYPE-TYPO 0/1, DROP-CONNECT 0/2, PORT-DIRECTION 0/1, DROP-TRANSITION 0/1, DROP-ENTRY-ACTION 0/1 |
| NO-REPAIR | ALL | ✓ | NOT_QUALIFIED | CLOSED | 7 | 0.92 | 0 | 1 | 2 | 116431 | DOC-QUOTE 0/3, READONLY 0/2, C-NEGATION 0/1, GUARD-ATTR-MISSING 0/1, TYPE-TYPO 0/1, DROP-CONNECT 0/2, PORT-DIRECTION 1/1, DROP-TRANSITION 0/1, DROP-ENTRY-ACTION 0/1 |
