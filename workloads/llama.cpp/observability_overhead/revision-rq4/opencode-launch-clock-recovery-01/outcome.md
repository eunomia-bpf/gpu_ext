# Review disposition

The corrected OpenCode review agrees with the bounded conclusion:

- the one-second NVBit minimum span is a legitimate measurement-resolution
  repair, but only a new run can show whether the unchanged drift gate passes;
- preflight-575-07 does not establish actual NVBit clock drift above the limit;
- the gpubpf arm remains invalid because 197/220 samples are uncertain, and
  the old log cannot identify how many cross zero versus a bin boundary;
- CUPTI's normalized timestamp contract is insufficient to replace raw
  `%globaltimer` calibration;
- the documented RM PTIMER correlation path is the strongest principled next
  implementation, provided its internal CPU bracket uncertainty is exposed or
  otherwise conservatively bounded.

The review did not authorize changing a threshold, bin, raw result, or paper
claim. No GPU execution was performed in this repair audit.

