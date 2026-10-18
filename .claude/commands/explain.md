---
description: Explain a kernel the way I need to present it (handout questions, registers, layout, numbers)
argument-hint: <kernel function name>   e.g. conv3x3_s1_v1
---
Read CLAUDE.md, the owner's brief in docs/briefs/, and the source of "$ARGUMENTS". Explain it for a 5-minute presentation, using the section headings of docs/STRIDE1_GUIDE.md: independent operations, dependent instructions, functional units, SIMD, peak and what limits it, the register tile and why that size, register allocation and reuse, memory layout and working set per call, and parallelization. Use real shapes from docs/LAYERS.md and real numbers from results/ when they exist; mark anything not yet measured as "to measure on ece007". End with three questions the professor is likely to ask and short answers.
