# Benchmark instances

The evaluation instances used in the paper. Each file is one benchmark: a list of
instances, each holding the idea(s) to judge and the ground-truth label.

```
human-only/                     D_h = validated novel ICLR papers
                                D_l = validated lower-novelty ICLR papers
  pairwise.yaml                   which of two ideas is more novel
  pairwise_plan.yaml              same pairs, ideas converted to plan form
  pointwise.yaml                  is this idea novel (binary)
  pointwise_plan.yaml             same ideas, plan form

human-plus-generated/           D_h unchanged; D_l = ideas from a plain LLM ideator (sonnet-4-5)
  pairwise.yaml                   
  pairwise_plan.yaml
  pointwise.yaml
  pointwise_plan.yaml

  backbone-gpt-5.1/             D_l regenerated with a different ideation backbone
  backbone-gpt-5.4/             
  backbone-opus-4-5/
    pairwise.yaml
    pointwise.yaml
```

`_plan` variants hold the same ideas rewritten into plan form (context, purpose,
mechanism, evaluation) and are the idea-format ablation.