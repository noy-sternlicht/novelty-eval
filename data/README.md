# Benchmark instances

The evaluation instances used in the paper. Each file is one benchmark: a list of
instances, each holding the idea(s) to judge and the ground-truth label.

```
human-only/                     D_+ = validated novel ICLR papers
                                D_- = validated lower-novelty ICLR papers
  pairwise.yaml                   which of two ideas is more novel
  pairwise_plan.yaml              same pairs, ideas converted to plan form
  pointwise.yaml                  is this idea novel (binary)
  pointwise_plan.yaml             same ideas, plan form

human-plus-generated/           D_+ unchanged; D_- = ideas from a plain LLM ideator (sonnet-4-5)
  pairwise.yaml                   
  pairwise_plan.yaml
  pointwise.yaml
  pointwise_plan.yaml

  backbone-gpt-5.1/             D_- regenerated with a different ideation backbone
  backbone-gpt-5.4/             
  backbone-opus-4-5/
    pairwise.yaml
    pointwise.yaml
```

`_plan` variants hold the same ideas rewritten into a two-field plan (purpose,
mechanism) and are the idea-format ablation.

These files are also released on the Hub as
[noystl/novelty-judge-bench](https://huggingface.co/datasets/noystl/novelty-judge-bench)