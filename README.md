# Keep Agent Benchmarks Honest About Work

This repository contains the release materials for the position paper
**Keep Agent Benchmarks Honest About Work**.

The paper argues that knowledge-work agent benchmarks should state what
work claim their scores can support. It proposes a three-step reporting
approach: define the work activity under evaluation, specify the tested
setting, and score the appropriate work product.

Paper link:

An arXiv link will be added after posting.

Open the companion HTML page:

[`index.html`](index.html)

The HTML page gives an interactive view of the O*NET-derived
work-activity taxonomy and the ESCO comparison used in the paper. It is a
companion artifact for reading the position paper.

## Paper Summary

Knowledge-work benchmarks often report performance on prompts, tasks, or
environments, then their scores are used to support broader claims about
workplace-facing capability. The paper argues that this inference is
fragile unless the benchmark report identifies the work activity, the
tested setting, and the scored work product.

The paper uses three benchmark case analyses to demonstrate the approach:
GDPval, OfficeQA Pro, and APEX-SWE. These cases show how different
scoring designs support different claims. Occupational deliverables,
grounded final answers, and executable software-state changes expose
different parts of work and leave different gaps.

## Work-Activity Reference

To help benchmark reports name the work being evaluated, the paper
derives an 18-activity reference set from
[O*NET](https://www.onetcenter.org/database.html) 30.2 task statements in
Job Zones 3-5 and knowledge-work occupations. After screening, the
reporting corpus contains 12,464 task statements. A stricter atlas subset
contains 8,372 task statements.

[ESCO](https://esco.ec.europa.eu/en) is used as an external legibility
check. The ESCO check assigns 5,826 scoped skill and competence items to
the same 18 labels or to a none-of-the-above category after the O*NET
taxonomy is fixed.

## Companion HTML

The companion HTML page places the main taxonomy table and interactive
atlas first, followed by result interpretation, O*NET and ESCO comparison
results, implications, limitations, and method details.

The atlas supports 3D and 2D views. Hovering over a point shows the source
job, original task, normalized task label, work activity,
work-activity cluster, and label color.

## Citation

Please cite the arXiv paper after posting.
