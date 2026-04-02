# Docs Index

`docs/` 现在按用途分为 4 类，建议优先从这里进入。

## 目录结构

- `experiments/`
  - 项目实验记录、阶段总结、Diff-MoE / DeepSpeed 复现与 benchmark 文档
- `setup/`
  - 环境部署、AutoDL 初始化与迁移说明
- `guides/`
  - FasterTransformer 各模型/模块使用指南
- `reference/`
  - 补充问答、Artifact Evaluation 等辅助资料

保留的专题目录：

- `images/`
  - 文档配图与流程图资源
- `models/`
  - 预置模型说明文档
- `paper_notes/`
  - 论文阅读笔记

## 推荐阅读顺序

如果你是继续推进当前 `Diff-MoE + AutoDL + switch-base-128` 工作，推荐按下面顺序阅读：

1. [`experiments/autodl_switch_base_closeout_zh.md`](./experiments/autodl_switch_base_closeout_zh.md)
2. [`experiments/autodl_switch_base_experiment_note_zh.md`](./experiments/autodl_switch_base_experiment_note_zh.md)
3. [`setup/autodl_4090d_deploy_zh.md`](./setup/autodl_4090d_deploy_zh.md)
4. [`experiments/autodl_switch_base_zh.md`](./experiments/autodl_switch_base_zh.md)

## 分类索引

### experiments

- [`autodl_switch_base_closeout_zh.md`](./experiments/autodl_switch_base_closeout_zh.md)
- [`autodl_switch_base_experiment_note_zh.md`](./experiments/autodl_switch_base_experiment_note_zh.md)
- [`autodl_switch_base_zh.md`](./experiments/autodl_switch_base_zh.md)
- [`deepspeed_benchmark_guide.md`](./experiments/deepspeed_benchmark_guide.md)
- [`deepspeed_repro_zh.md`](./experiments/deepspeed_repro_zh.md)

### setup

- [`autodl_4090d_deploy_zh.md`](./setup/autodl_4090d_deploy_zh.md)

### guides

- [`bart_guide.md`](./guides/bart_guide.md)
- [`bert_guide.md`](./guides/bert_guide.md)
- [`deberta_guide.md`](./guides/deberta_guide.md)
- [`decoder_guide.md`](./guides/decoder_guide.md)
- [`gpt_guide.md`](./guides/gpt_guide.md)
- [`gptj_guide.md`](./guides/gptj_guide.md)
- [`gptneox_guide.md`](./guides/gptneox_guide.md)
- [`longformer_guide.md`](./guides/longformer_guide.md)
- [`swin_guide.md`](./guides/swin_guide.md)
- [`t5_guide.md`](./guides/t5_guide.md)
- [`vit_guide.md`](./guides/vit_guide.md)
- [`xlnet_guide.md`](./guides/xlnet_guide.md)

### reference

- [`ae.md`](./reference/ae.md)
- [`QAList.md`](./reference/QAList.md)
- [`paper_notes/diff_moe_paper_notes_zh.md`](./paper_notes/diff_moe_paper_notes_zh.md)

## 维护约定

- 新增 AutoDL / 实验推进文档时，优先放到 `experiments/` 或 `setup/`。
- 通用模型指南统一放到 `guides/`。
- `docs/` 根目录尽量只保留目录索引和专题文件夹，避免再次堆平。
