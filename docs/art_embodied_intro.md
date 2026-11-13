# wandb/ART-Embodied 紹介

[wandb/ART-Embodied](https://github.com/wandb/ART-Embodied) は robot policy の rollout と学習更新を trajectory-aware に扱う embodied RL 基盤です。

- LeRobot が policy、processor、action 実行を所有
- simulator / robot runtime が reset と transition を所有
- ART-Embodied が trajectory group、relative advantage、optimizer update を管理
- W&B が metrics、simulation video、evaluation Table、model / training-state Artifact を記録
- Weave が update / group / trajectory の trace を記録

observability の設定例:

```yaml
observability:
  delivery_failure_policy: fail_run
  wandb:
    enabled: true
    project: art-embodied
    mode: online
    log_model_artifacts: true
    log_evaluation_table: true
  weave:
    enabled: true
    project: art-embodied
    trace_trajectories: true
  videos_per_update: 2
  videos_per_evaluation: 4
```

今回の短時間講座では実行せず、「2つの Lab で手動実装した metric / checkpoint / video の記録を、trajectory group と trace を含む embodied RL 全体へ拡張したもの」として5〜10分紹介します。

embodied API は active development です。実行時は公式 README と `art-embodied doctor` で、検証済みの policy / simulator / dependency profile を確認してください。LeRobot、LIBERO、PI の環境は分離します。

