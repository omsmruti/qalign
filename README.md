# Sample, Don't Search: Rethinking Test-Time Alignment for Language Models

Gonçalo Faria, Noah A. Smith

**Paper**: tbd

**TL;DR:** QAlign is a new test-time alignment approach that improves language model performance by using Markov chain Monte Carlo methods.

### Abstract:
Increasing test-time computation has emerged as a promising direction for improving language model performance, particularly in scenarios where model finetuning is impractical or impossible due to computational constraints or private model weights. However, existing test-time search methods using a reward model (RM) often degrade in quality as compute scales, due to the over-optimization of what are inherently imperfect reward proxies. We introduce QAlign, a new test-time alignment approach. As we scale test-time compute, QAlign converges to sampling from the optimal aligned distribution for each individual prompt. By adopting recent advances in Markov chain Monte Carlo for text generation, our method enables better-aligned outputs without modifying the underlying model or even requiring logit access. We demonstrate the effectiveness of QAlign on mathematical reasoning benchmarks (GSM8K and GSM-Symbolic) using a task-specific RM, showing consistent improvements over existing test-time compute methods like best-of-n and majority voting. Furthermore, when applied with more realistic RMs trained on the Tulu 3 preference dataset, QAlign outperforms direct preference optimization (DPO), best-of-n, majority voting, and weighted majority voting on a diverse range of datasets (GSM8K, MATH500, IFEval, MMLU-Redux, and TruthfulQA). A practical solution to aligning language models at test time using additional computation without degradation, our approach expands the limits of the capability that can be obtained from off-the-shelf language models without further training.
<!-- toc -->


-----
## <div align="center">Documentation</div>

TBD

We heavily relly on deepspin/quest-decoding and  goncalofaria/expkit.

-----
## <div align="center">Quick Start Examples</div>

Replicating the work: 

* create configs for general experiments and task
```bash
scripts/create_all_general_experiments.sh
```
```bash
scripts/create_all_task_experiments.sh
```
* run those locally
```bash
scripts/run_local_experiments.sh
```
* run those remote
```bash
scripts/run_remote_experiments.sh
```
* compare responses against gt answers
```bash
scripts/run_eval_experiment.sh
```
* evaluate reward model for ancestral predictions ( remote by default )
```bash
scripts/run_rm_eval.sh
```
* run WMV BON MV final prediction methods
```bash
scripts/run_pred.sh
```



-----

## <div align="center">Contact</div>

For bugs and feature requests please visit [GitHub Issues](https://github.com/goncalorafaria/qalign/issues). For business inquiries or
professional support requests please send an [e-mail](mailto:goncalofaria.research@gmail.com).

-----

## <div align="center">Citation</div>

````
@misc{faria2024sample,
      title={Sample, Don't Search: Rethinking Test-Time Alignment for Language Models},
      author={Gon{\c{c}}alo R. A. Faria and Noah Smith},
      year={2024},
      eprint={2024.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      note={Website: \url{https://www.questdecoding.com/qalign}}
    }
````

