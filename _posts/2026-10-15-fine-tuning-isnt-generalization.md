---
layout: post
title: "Fine-Tuning Isn't Generalization: Teaching Qwen3-4B PyTorch Knowledge with QLoRA and PEFT"
categories:
  - Machine Learning
  - Large Language Models
  - Fine-Tuning
  - post
comments: true
tags:
  - Machine learning
  - LLM
  - Fine-tuning
  - PyTorch
  - Evaluation
excerpt_separator:  <!--more-->
sidebar_link: true
last_modified_at: 2026-10-15T00:00:00+05:30
---

Fine-tuning a language model on a technical document looks deceptively simple. Turn a document into question-answer pairs, train a small adapter, ask a similar question, and inspect the answer. If the model reproduces the answer it saw during training, it is tempting to say that the model learned the document.

This post is about why that conclusion is too strong.
<!--more-->

Asked which PyTorch version it knew, the base model said it had no PyTorch-specific training data and claimed PyTorch 2.1.0 was latest in June 2024. The training-data statement is a model self-report, not verified corpus metadata.

That version claim was wrong: [PyTorch 2.3](https://pytorch.org/blog/pytorch2-3/) was announced in April 2024, and [2.3.1](https://dev-discuss.pytorch.org/t/pytorch-2-3-1-general-availability/2144) became generally available in June. Qwen3 was announced in April 2025, before the [PyTorch 2.14 release blog](https://pytorch.org/blog/pytorch-2-14-release-blog/) in September 2026. This chronology motivates testing newer material; it does not show what PyTorch knowledge the model acquired during pretraining.

I ran a small, local experiment using QLoRA on Qwen3-4B. The [PyTorch 2.14 release blog](https://pytorch.org/blog/pytorch-2-14-release-blog/) was the only source used to construct training targets and ground expected evaluation answers. The model's pretrained knowledge remained present and could contribute to its responses; the source constraint did not erase or disable it. The goal was not to build a production assistant or to compare every possible knowledge-injection method. It was narrower: can a parameter-efficient fine-tune preserve technical facts, defaults, version boundaries, and source limitations when a question is asked in a new form?

The short answer is: it can reproduce training answers, and it can sometimes transfer pieces of a fact, but that was not the same thing as reliable generalization. The most useful lesson was not a winning adapter weight. It was learning to separate recall, qualification, and evidence discipline instead of treating a fluent answer as a pass.

## Why run this experiment?

Behavioral fine-tuning targets consistent format, vocabulary, tone, or specialized task style. In RAG, prompts, examples, and structured-output instructions often suffice; fine-tuning may help when those controls are unreliable or examples must teach specialized behavior. Factual injection is different: fine-tuning tries to shape parametric recall, while RAG supplies evidence at inference time.

An adapter can be compact and work offline, but that does not guarantee factual accuracy, preserved qualifiers, or restraint. A model can name the right feature while reversing its default or inventing an API.

Retrieval-augmented generation is relevant context here, but it was not the subject of this experiment. Ovadia et al. compare unsupervised fine-tuning with RAG across several knowledge-intensive tasks and report that RAG outperformed their fine-tuning condition in their tested settings. They also report that exposure to multiple variations of a fact could alleviate some fine-tuning difficulties. These findings motivate the question but do not predict the result of this supervised QLoRA pilot. [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs](https://aclanthology.org/2024.emnlp-main.15/) contrasts modifying weights with supplying retrieved evidence at inference time.

Instruction-tuning research provided a more direct motivation. Ghosh et al. found that, in their tested models and datasets, LoRA mainly changed response style while full-parameter tuning degraded knowledge; this does not establish that LoRA cannot inject knowledge. For our experiment, reproducing a training answer is not enough: it shows fit, not transfer to new question forms or preservation of qualifiers. Arm A reproduced all eight canonical answers but transferred inconsistently; Arm B got some complementary questions right, without a reliable generalization advantage. [A Closer Look at the Limitations of Instruction Tuning](https://proceedings.mlr.press/v235/ghosh24a.html) reports the study. In tests with Gemini 1.5, Zhao et al. found that knowledge transfer varied by data format and fact type, and that multi-step reasoning remained difficult; these results are specific to their datasets and models. [From Style to Facts](https://papers.nips.cc/paper_files/paper/2025/hash/90043ebd68500f9efe84fedf860a64f3-Abstract-Conference.html) frames knowledge injection as conditional, not guaranteed.

That framing shaped the experiment. I wanted to see whether a small adapter trained on carefully sourced facts would preserve the qualifications that make technical answers useful.

## The experiment in one picture

![Experiment flow from the PyTorch release blog through claim-ledger dataset creation, Base and Arm A/Arm B training, three adapter combinations, and frozen evaluation with manual scoring.]({{site.url}}/assets/image/experiment-flow.svg)

The release blog was the sole source for authoring factual targets and evaluation reference criteria. The base model's pretrained knowledge remained active, and the system prompt's source instruction could guide but not erase that knowledge. The experiment therefore measures how fine-tuning changed answers from a prompted pretrained model; it does not measure learning from a blank or source-isolated model. The source constraint is still useful for scoring: when the blog does not establish a requested latency, driver requirement, API, or performance number, the reference criteria mark that limitation rather than supplying an outside fact.

The base model was `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`. The training machine was a Windows laptop with an NVIDIA GeForce GTX 1060 Max-Q and 6 GB of VRAM. The local runtime used PyTorch 2.8.0 with CUDA 12.6; PyTorch 2.14 was the subject of the source material, not the installed training framework. This distinction matters when reading the results.

## Dataset preparation: several iterations before the pilot

The dataset did not begin as the final pilot. It evolved because early measurements showed how little a training loss or a response prefix can tell us about factual transfer.

The first version, v0.1, had 12 training rows: nine supported factual claim clusters and three behavior examples. The original frozen evaluation had 14 questions: ten supported factual claims and four questions whose requested details were not in the source. A very small initial budget of six optimizer steps was obviously not enough to support a strong conclusion, but it established the initial comparison format.

Version v0.2 expanded to 21 rows, mostly by repeating the same nine claim clusters in different forms. It completed nine optimizer steps with a reported training loss around 2.4. The important lesson was structural: more rows are not automatically more independent facts. Rephrasing can be useful for robustness, but it does not create new knowledge coverage by itself.

Version v0.3 improved the evaluation protocol. It separated held-out facts from alternate wordings of trained facts, compositional questions, and mixed supported/unsupported questions. It also included a prompt ablation for the required answer prefix. That ablation mattered because both the base model and adapter could achieve perfect prefix compliance when the system prompt explicitly asked for it. Prefix compliance was behavior under prompting, not evidence of factual learning.

Version v0.4 was the broader data pass: 114 rows, including 98 supported factual rows across 90 distinct supported claim clusters and 16 source-boundary behavior rows. All targets used the same prefix. The 3-epoch run completed 45 optimizer steps, reported training loss 1.929, and took roughly 16 minutes. A subsequent 12-question training diagnostic still did not reliably reproduce the intended answers. The run had completed; that did not establish useful knowledge transfer.

The completion-only training-mask audit was reassuring but limited. It found that all 114 prompt token sequences were exact prefixes of their corresponding prompt-plus-answer sequences, no rows were truncated at the 1024-token limit, and every row retained answer tokens for loss computation. That ruled out one obvious implementation explanation. It did not guarantee that the targets were easy to learn, that all target phrasing was equally useful, or that a lower loss would survive a changed question formulation.

### What facts did the pilot cover?

The final controlled comparison narrowed the scope to eight claim clusters from the PyTorch 2.14 release material. Six covered specific technical changes; two tested whether the model could recognize the limits of the available evidence.

| Claim cluster | Technical topic | What we wanted the model to learn or recognize |
| --- | --- | --- |
| `IND-01` | Inductor `simple_overlap` | How collectives overlap with independent computation, and that the feature is enabled by default in PyTorch 2.14. |
| `DIST-01` | The `nccl2` distributed backend | Its preview status, its role as a drop-in replacement for the existing NCCL c10d backend, and the stated transition to the default in PyTorch 2.15+. |
| `CTRL-01` | Declarative dynamic shapes | How `@dynamic_spec` provides a shared declaration across `torch.compile`, `torch.export`, and `make_fx`. |
| `CUDA-01` | NVGEMM | Which operations—`mm`, `addmm`, and `scaled_mm`—can receive NVGEMM candidates, and how those candidates compete with Triton and ATen during autotuning. |
| `MPS-07` | Cholesky on Apple Silicon | The faster panel-factorization algorithm using a `matmul2d`-based trailing update. |
| `XPU-02` | Low-precision XPU operations | MXFP8 and MXFP4 support for `scaled_mm`, along with the API's experimental status. |
| `BEH-UNSUPPORTED-01` | Unsupported latency claim | Whether the model would invent an exact H100 latency for `torch.linalg.polar` when the available evidence did not establish one. |
| `BEH-UNSUPPORTED-14` | Pinned host-memory visualization | Whether the model could handle a question about rendering pinned host-memory data without asserting capabilities unsupported by the evidence provided. |

The first six clusters test factual recall and preservation of technical qualifiers: defaults, version boundaries, supported operations, and API stability. The last two test whether the model recognizes when the evidence does not support a requested detail, rather than inventing a plausible answer.

This distinction matters: a model can remember a feature name but get its default wrong, or answer the main question correctly and append unsupported details. The pilot was designed to expose these different failure modes rather than reduce every response to a single right-or-wrong score.

## QLoRA configuration

The training script loaded the base model in four-bit NF4, enabled double quantization, used FP16 compute, prepared the model for k-bit training, and attached LoRA adapters to seven projection modules. This is an excerpt from the actual configuration in `train_lora.py`:

```python
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True
)
model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ),
)
```

The training data used completion-only loss: the system prompt and user question were masked with `-100`, and only assistant answer tokens contributed to the causal-language-model objective. Per-device batch size was one, the learning rate was `1e-4`, the maximum sequence length was 1024, the optimizer was `paged_adamw_8bit`, and greedy evaluation used `enable_thinking=False` with at most 220 generated tokens.

These details made the experiment feasible on 6 GB of VRAM. They do not make it a controlled study of optimal rank, learning rate, data scale, seed sensitivity, or training duration.

## Canonical versus complementary question forms

The Pilot v1 comparison tried to hold optimizer steps constant while changing the diversity of question forms.

Arm A contained eight canonical examples, one per claim cluster, trained for 30 epochs. Arm B contained 24 complementary forms, three per cluster, trained for 10 epochs. With batch size one and gradient accumulation one, both arms had 240 optimizer steps. Assuming each epoch processed every row, each claim cluster was presented 30 times in both arms: Arm A repeated one question form 30 times, while Arm B presented three forms 10 times each. The difference was how exposure was distributed across forms, not necessarily the total number of example presentations per claim. Equal presentations and equal optimizer steps do not establish equal supervised-token exposure because target lengths and token counts can differ. The saved Pilot v1 artifacts do not contain verified tokenizer-level answer-token totals by row or claim, so I cannot quantify that difference.

Here is a simplified, faithful illustration from the actual `simple_overlap` examples. The prose is shortened for readability; the full rows also carry IDs and source-evidence metadata.

Canonical form, abbreviated to the question and answer fields:

```json
{"question":"What does Inductor's `simple_overlap` reordering do, and what is its default?","answer":"PyTorch v2.14: `simple_overlap` interleaves collectives with independent computation so communication can overlap with compute. It is enabled by default."}
```

Complementary form, also abbreviated to those fields:

```json
{"question":"Correct this claim: `simple_overlap` only changes communication ordering and does not create communication/compute overlap.","answer":"The claim is false. `simple_overlap` interleaves collectives with independent computation specifically so communication can overlap with compute, and it is enabled by default in PyTorch 2.14."}
```

The complementary set also used version-boundary questions, negative questions, and source-boundary questions. The intent was reasonable: a technical fact is often needed in several forms, and a model that can only answer one template is not very useful.

However, Pilot v1 has a major format confound. Arm A targets started with `PyTorch v2.14:` and Arm B targets did not use that same prefix consistently. The system prompt required the prefix, but the target mismatch made prefix behavior an unfair proxy for question-form diversity. It was therefore recorded independently from the factual scores.

## Training fit is not unseen-formulation transfer

An eight-example overfit check made the distinction vivid. With the 30-epoch adapter, the exact training questions reportedly reproduced all eight target answers. The corresponding five-epoch diagnostic also reduced mean teacher-forced target-token loss from 4.188 for the base model to 1.031 for the adapter, with all eight target answers receiving lower loss.

Those are meaningful observations about the probability assigned to the training targets. They are not evidence that the adapter learned an abstract rule or can apply a source-grounded fact under a different prompt.

When the 30-epoch adapter was asked eight alternative formulations of the same claims, a preliminary manual review found two clear passes, two partial answers with unsupported additions, and four failures. It reversed the `simple_overlap` default, moved the `nccl2` transition from 2.15 onward to “2.14+,” denied the shared `@dynamic_spec` capability, and invented implementation detail around MPS and XPU. The two cleanest passes were source-boundary behavior: saying that the release blog did not report an exact H100 latency for `torch.linalg.polar`, and that `memory_viz` did not yet render captured host data.

This is why “the training answer came back” is a misleading success metric. It measures fit to an exact answer distribution, not robust factual transfer.

## Freezing the evaluation and scoring more than correctness

For the adapter-merging study, I created an eight-question development set and a 24-question final set before generating merged-model outputs. The final set contains three prompts for each of eight claim clusters, including direct questions, false-premise corrections, and source-boundary questions. Its SHA-256 digest was recorded before inference.

The final prompts deliberately included traps that a well-grounded assistant should refuse. For example, the release blog does not provide an H100 millisecond latency for `torch.linalg.polar`; it does not state a particular Intel driver requirement for MXFP4; and it does not list Python or ATen APIs whose semantics `simple_overlap` changes. A correct response to those questions is not an impressive hidden fact. It is an explicit limitation.

Each response was evaluated manually on four independent 0–2 scales:

1. Core factual accuracy.
2. Qualifier preservation, such as defaults, API-stability labels, or version boundaries.
3. Unsupported-detail avoidance.
4. Relevance and completeness.

Prefix compliance was recorded separately. This separation is important. A response can identify the right feature and still be untrustworthy because it invents a driver version, a hardware requirement, or an API. Conversely, a cautious base model can avoid unsupported detail while failing to answer a source-supported question.

### The `simple_overlap` case study

`simple_overlap` was the clearest compact example. The source says that it interleaves collectives with independent computation so communication can overlap with compute, and that it is enabled by default in PyTorch 2.14. The blog does not say that it is a stack/cat data-transfer feature, and it does not enumerate specific modified ATen APIs.

The models often remembered a fragment of that statement. Some said the mechanism correctly but reversed the default. Some rejected the stack/cat premise but replaced it with invented APIs. A response that contains “collectives” and “overlap” can sound convincing while being wrong in exactly the portion an engineer needs for a configuration decision. That is why the scoring treated mechanism, default, and unsupported additions as distinct dimensions.

## Adapter merging

Both source adapters used the same Qwen3-4B revision, LoRA rank 8, alpha 16, dropout 0.05, target modules, and adapter tensor layout. Three exploratory combinations were built:

| Candidate | Arm A weight | Arm B weight |
|---|---:|---:|
| Balanced | 0.50 | 0.50 |
| Canonical-heavy | 0.75 | 0.25 |
| Complementary-heavy | 0.25 | 0.75 |

The implementation used PEFT's `add_weighted_adapter` with `combination_type="cat"`:

```python
model.add_weighted_adapter(
    ["arm_a", "arm_b"],
    [args.weight_a, args.weight_b],
    adapter_name="merged",
    combination_type="cat",
)
model.set_adapter("merged")
```

This is an exact excerpt from the local merge utility, with the command-line variables retained. In the installed PEFT version, concatenation builds a weighted effective update by concatenating the appropriately weighted LoRA factors; the resulting adapter rank is 16. That was preferable here to `linear`, which PEFT documents as a rough approximation. PEFT also offers SVD-based merging, and its merge utilities include methods such as TIES and DARE. Those methods answer different questions about compression, conflict resolution, sparsity, or task-vector composition. They were not evaluated here. This was a small compatibility-tested `cat` composition study, not a survey or a claim that concatenation is universally best.

The three candidates were validated for compatible configuration, expected rank, finite tensors, and successful serialization. The final evaluation file was hashed before outputs were generated. The current repository contains complete eight-row development outputs and metadata for Base, Arm A, Arm B, and all three merge candidates. An early merge report says the first Base development attempt terminated before writing a row, but that report is stale relative to the current nonempty `base_dev.jsonl` and its metadata. I found no saved development score sheet or model-selection log that would let a reader reproduce a score-based selection of 50/50. In the recorded workflow, the balanced candidate advanced after an informal review of the development responses; that choice should be treated as qualitative, not as a documented numerical winner.

The final split was frozen before the primary final outputs, as shown by its recorded hash and the merge manifest. The reconciled four-condition scoring report is timestamped 03:40 on October 13, 2026. The metadata for the 75/25 and 25/75 final runs is later, at approximately 03:52 and 04:22. Those additional runs therefore followed inspection and scoring of the primary final comparison. They are post-inspection exploratory outputs and cannot fairly be used to select an optimal weighting.

## What did the six conditions show?

All six conditions have complete 24-response final JSONL outputs: Base, Arm A, Arm B, and the 50/50, 75/25, and 25/75 combinations. The original reconciled manual scoring report covers the primary four-condition comparison: Base, Arm A, Arm B, and the balanced merge. Its scores are item-level manual judgments on four separate 0–2 dimensions; the report does not describe blinded or independent double scoring. The 75/25 and 25/75 outputs were generated after the primary final results had been inspected. A later six-condition review provides supplementary manual scores for these two candidates; they are reported separately below as post-hoc exploratory results.

| Condition | Core fact mean /2 | Qualifier mean /2 | Unsupported-detail avoidance /2 | Relevance /2 | Prefix |
|---|---:|---:|---:|---:|---:|
| Base | 1.04 | 0.83 | 1.50 | 1.38 | 22/24 |
| Arm A | 0.79 | 0.79 | 0.58 | 1.25 | 23/24 |
| Arm B | 1.25 | 1.17 | 0.92 | 1.33 | 0/24 |
| 50/50 merge | 1.33 | 1.21 | 0.92 | 1.50 | 0/24 |

The 50/50 merge had the highest descriptive means for core factual accuracy and relevance in the reviewed four-condition sample. It did not improve evidence discipline over Arm B, and it was materially less cautious than Base. Its zero prefix score is also not evidence that merging damaged factual transfer: it is confounded by Arm B's missing prefix supervision.

### Supplementary merge-weighting experiments

The 75/25 and 25/75 generations were produced after the primary final results had been inspected. Their scores are therefore post-hoc exploratory evidence, not confirmatory evidence and not a valid basis for selecting an optimal weighting. They come from a later manual scoring pass, are based on the same 24-question frozen set, and should be interpreted with the scoring package's single-evaluator and small-sample limitations.

| Metric | 75/25 (canonical-heavy) | 25/75 (complementary-heavy) |
|---|---:|---:|
| Core factual accuracy /2 | 1.125 | 1.333 |
| Qualifier preservation /2 | 1.000 | 1.083 |
| Unsupported-detail avoidance /2 | 1.042 | 1.250 |
| Relevance and completeness /2 | 1.500 | 1.500 |
| Prefix compliance* | 14/24 | 0/24 |

*Prefix compliance is reported separately, not included in the four means. It is confounded by inconsistent prefix supervision between Arm A and Arm B.

The complementary-heavy merge tied the balanced merge on mean core factual accuracy (1.333/2) and relevance (1.500/2), and scored higher on avoiding unsupported additions (1.250 versus 0.917). The balanced merge scored higher on qualifier preservation (1.208 versus 1.083). Thus, the balanced merge remains promising in the primary comparison, but it did not consistently preserve qualifiers or prevent unsupported additions. These small, manually scored results do not establish a statistically significant winner or an optimal weighting.

The additional raw outputs help illustrate the trade-offs in the aggregate scores. In the 75/25 responses, some answers correctly named the `simple_overlap` mechanism and acknowledged unsupported latency or driver details, while others invented APIs and version boundaries or made unsupported performance and hardware claims. In the 25/75 responses, some source-boundary answers were appropriately cautious, but the model reversed the `simple_overlap` default and invented details about NCCL, MPS, XPU, and host-memory APIs. These examples do not establish which weighting is better: both runs followed inspection of the primary final results, and a new untouched evaluation would be needed for a confirmatory comparison.

The most defensible reading is modest: the balanced merge was a promising recall-and-relevance candidate in this small sample, but it was not a clean factual winner. Better recall and better evidence discipline did not move together.

## Limitations and what I would change

This was a single-model, single-seed pilot on a local GPU. The final evaluation covered 24 prompts across eight claim clusters, scored manually by one evaluator rather than through a blind, multi-rater benchmark. The source audit covered one release article and was not independently verified.

The training comparison also had limitations. Both arms used 240 optimizer steps and, assuming every row was processed each epoch, 30 presentations per claim cluster. However, answer-token exposure was not matched, and prefix supervision differed between arms. The alternative-formulation check and two additional merge-weight sweeps were exploratory, not confirmatory evidence of generalization or a superior merge.

A follow-up experiment would address four things:

1. **Verify the claims:** Maintain a source-verified claim ledger recording the evidence, required qualifiers, and nearby unsupported details for each claim.
2. **Balance training exposure:** Match answer-token exposure, prefix supervision, and question-form diversity per claim—not just optimizer steps and example presentations.
3. **Separate selection from evaluation:** Freeze development and confirmatory sets before training, and define the scoring rubric and selection rules before evaluating the final set.
4. **Include retrieval:** Treat retrieval as a serious baseline for evidence-grounded technical questions.

The broader lesson is that fine-tuned answers should be evaluated as **evidence-bearing technical communication**, not just strings that resemble references. A model may learn a response pattern, a keyword association, part of a fact, or an answer template. These are different outcomes. If reliability depends on exact defaults, API names, version boundaries, and appropriate abstention, each must be evaluated explicitly.

## References

1. Oded Ovadia, Menachem Brief, Moshik Mishaeli, and Oren Elisha. [Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs](https://aclanthology.org/2024.emnlp-main.15/). EMNLP, 2024.
2. Sreyan Ghosh et al. [A Closer Look at the Limitations of Instruction Tuning](https://proceedings.mlr.press/v235/ghosh24a.html). ICML, 2024.
3. Eric Zhao, Pranjal Awasthi, and Nika Haghtalab. [From Style to Facts: Mapping the Boundaries of Knowledge Injection with Finetuning](https://papers.nips.cc/paper_files/paper/2025/hash/90043ebd68500f9efe84fedf860a64f3-Abstract-Conference.html). NeurIPS, 2025.
4. [PyTorch 2.14 Release Blog](https://pytorch.org/blog/pytorch-2-14-release-blog/).
