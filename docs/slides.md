<!--
W&B Physical AI ワークショップ スライドデック下書き (Marp 形式)

位置づけ:
- 講師 (W&B 社員) は W&B エキスパート、Physical AI 非当業者
- 聴衆には Physical AI 実務者が含まれる（講師より詳しい可能性）
- 戦略: Physical AI の専門解説で競わない。W&B 固有の付加価値
  （インテグレーション網羅性 / マルチメディア UX / Weave 観測 / ART Serverless RL）に集中

用途: 顧客向け Robotics ワークショップ & NVIDIA 共催セミナー共用
想定時間: 90–120 分
このファイルは ChatGPT / Claude に渡して Marp でレンダリングさせる想定。
-->

---
marp: true
theme: default
paginate: true
size: 16:9
header: "W&B Physical AI Workshop"
footer: "Weights & Biases Japan"
---

# Physical AI を **W&B で加速する**
## インテグレーション・マルチメディア・Agentic RL

**Weights & Biases Japan**

<!--
核心メッセージ:「あなたが使っているフレームワークが何であれ、W&B が
その上に載る」。Physical AI を教えに来たのではなく、W&B が Physical AI
開発にどう刺さるかを見せに来た、という立ち位置を最初に明確にする。
-->

---

# 今日のゴール

- Physical AI スタックの**どこに W&B が刺さるか**の地図を提供
- W&B の**マルチメディア機能**（動画・3D・画像）をライブ体験
- Weave による **rollout 観測 + マルチモーダル Eval** を見る
- OpenPipe ART + W&B **Serverless RL** で Agentic RL の現在地を示す

<!--
「Physical AI を教える」とは言わない。聴衆がそこは詳しい可能性がある。
フォーカスは W&B 側の差別化機能。
-->

---

# 本日のアジェンダ

1. 会社紹介と講師自己紹介 (5min)
2. **W&B 製品全体像** — Physical AI 視点 (15min)
3. **Physical AI × W&B 連携マップ** — 主要 FW 全カバー (15min)
4. **マルチメディア可視化** — Models と Weave のライブデモ (30min)
5. **Agentic RL** — OpenPipe ART × W&B Serverless RL (25min)

---

<!-- _class: lead -->
# 1. 会社紹介 & 自己紹介

---

# Weights & Biases

- 2017 年創業、SF 本社、**日本法人あり**
- ML 開発者プラットフォーム：
  - **Models** — 実験追跡 / Artifact / Registry (MLOps)
  - **Weave** — LLM / Agent 観測 + Eval (LLMOps)
  - **Inference** — Managed LLM serving
- 利用企業：OpenAI, Meta, NVIDIA, Toyota Research, NVIDIA, Stability AI...
- 国内：SONY, トヨタ, メルカリ, LINE...

---

# 講師自己紹介

- 氏名：[TODO]
- ロール：Weights & Biases Japan, [TODO]
- **W&B の中の人として**、Physical AI 領域でのお客様事例を日々見ている
- 今日は「自分が Physical AI 研究者」としてではなく、「W&B が Physical AI 開発に**何を足せるか**」を共有しに来ました

<!--
この正直な立ち位置は聴衆にも好印象。研究発表ではなくソリューション共有。
-->

---

<!-- _class: lead -->
# 2. W&B 製品全体像
## — Physical AI 視点で

---

# なぜ Physical AI で W&B が効くか

Physical AI の開発サイクル特有の痛点：

- 実験数が多い（sim seed / reward weight / net size / ...）
- **loss では品質が分からない**（rollout 動画見ないと判断不能）
- sim2real のギャップ → デバッグは**動画 + 3D** が必須
- Policy の良し悪し比較に**並列動画プレビュー**が欲しい
- RL と LLM critic を組み合わせた**Agentic** ワークフロー増加中

→ これ全部、W&B の**既存機能**でカバーされている

---

# W&B 3 本柱

```
┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐
│   W&B Models      │  │   W&B Weave       │  │  W&B Inference  │
│                   │  │                   │  │                 │
│ - Experiments     │  │ - Traces          │  │ - Managed LLM   │
│ - Artifacts       │  │ - Evaluation      │  │ - OpenAI 互換    │
│ - Tables (動画)   │  │ - Multimodal      │  │ - Weave 連動     │
│ - Registry        │  │ - Scorers         │  │                 │
└─────────┬─────────┘  └─────────┬─────────┘  └────────┬────────┘
          │                      │                     │
          └──── set_wandb_run_context() で相互リンク ───┘
```

**この相互リンクが Physical AI の "訓練 → eval → 観測 → デプロイ" を一本線で追える**

---

# Models：実験追跡からモデル管理まで

| 機能 | Physical AI での役割 |
|---|---|
| Experiments | 訓練 loss / reward / eval success rate |
| Artifacts | **checkpoint / dataset / rollout 動画** をバージョン管理 |
| Tables | **動画カラム付き**で policy 比較 |
| Registry | `candidate → staging → production` の昇格ゲート |

```python
run = wandb.init(project="physical-ai")
run.log({"loss": 0.3, "eval_success": 0.62})
run.log({"rollout": wandb.Video("ep.mp4", fps=10)})
run.log_artifact(wandb.Artifact("pusht-diffusion", type="model"))
```

---

# Weave：Agent / Policy 観測の de facto

- `@weave.op()` を付けた関数は**すべての入出力が自動 trace**
- **マルチモーダル**：動画 / 画像 / 音声 / 3D を trace に埋め込める
- `weave.Evaluation` で**スコアラーを並列実行**
- W&B Models と相互リンク（同じ run で追跡可能）

```python
@weave.op()
def replay_rollout(episode, video_bytes):
    return {"success": episode["success"],
            "video": Content.from_bytes(video_bytes, extension=".mp4")}
```

---

# Inference：W&B 提供の LLM ホスト

- Llama 3 / Qwen 2.5 / DeepSeek V3 などを OpenAI 互換 API で
- **Weave と自動連携** → 呼び出しごとに trace が残る
- Physical AI 文脈では **Critic LLM / アノテーター** として組み込みやすい

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.inference.wandb.ai/v1",
                api_key=os.environ["WANDB_API_KEY"])
```

→ **OpenPipe ART の Serverless RL はここに checkpoint を自動デプロイ**（後述）

---

<!-- _class: lead -->
# 3. Physical AI × W&B 連携マップ
## どの FW に乗ってもプラグイン可

---

# Physical AI スタックの全体像

```
┌────────────────────────────────────────┐
│ Agentic / Critic   : OpenPipe ART      │ ← W&B Serverless RL
├────────────────────────────────────────┤
│ Policy (Model)     : Diffusion / ACT / │ ← W&B Models
│                      π0 / GR00T / VLA  │
├────────────────────────────────────────┤
│ 学習フレームワーク  : LeRobot, Isaac Lab│ ← W&B ネイティブ連携
│                      OpenPi, Cosmos-RL │
├────────────────────────────────────────┤
│ シミュレータ       : Isaac Sim, MuJoCo │
│                      Genesis, PushT    │
├────────────────────────────────────────┤
│ データ            : OXE, DROID, HF     │ ← W&B Artifacts
└────────────────────────────────────────┘
```

**学習フレームワーク層に W&B が"既に"刺さっている**のが本日のポイント

---

# W&B 連携総覧（主要フレームワーク）

| Framework | 深さ | ログ内容 | 有効化 |
|---|---|---|---|
| **LeRobot** (HF) | ★★★ | loss / eval 動画 / Artifact | `--wandb.enable=true` |
| **Isaac Lab** (NV) | ★★★ | scalars / **checkpoint & 動画 Artifact** | `--logger wandb --video` |
| **GR00T N1** (NV) | ★★ | fine-tune scalars | `--use-wandb` |
| **OpenVLA** | ★★ | Action Token Acc / LIBERO success | `--use_wandb True` |
| **Cosmos-RL** (NV) | ★★ | config / scalar | `init_wandb(cfg)` |
| **OpenPi** (PI 公式) | ★ | loss のみ (実務は LeRobot 経由推奨) | `wandb_enabled=True` |
| **OpenPipe ART** | ★★★★ | **trajectory + checkpoint → Inference 自動デプロイ** | `ServerlessBackend(api_key)` |

<!--
★の意味だけ説明: 数値だけか、動画・Artifact まで、それ以上か。
内容を読み上げるより「全フレームワークでフラグ1つ」を強調する。
-->

---

# 同じ UX、どの FW でも

**LeRobot：**
```bash
lerobot-train --policy.type=diffusion \
  --dataset.repo_id=lerobot/pusht \
  --wandb.enable=true --wandb.project=physical-ai
```

**Isaac Lab（RSL-RL）：**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Reach-Franka-v0 --num_envs=4096 \
  --logger wandb --log_project_name=physical-ai --video
```

→ **どちらも同じ W&B project に集まり、Tables / Registry で横断管理**

---

# デモ：LeRobot 訓練 → eval ログの流れ

[ライブ W&B UI ツアー — スクショ差し込み]

1. 訓練 run：loss / eval success rate カーブ
2. `eval/rollouts` Table：**20 エピソード動画を並列プレビュー**
3. Artifact の lineage：dataset → checkpoint → eval
4. Registry：candidate / production の昇格

<!--
このワークショップのレポ (wandb-physicalai-workshop) の実ログを
そのまま見せる。50k 訓練の success_rate=15% と該当動画を出す。
-->

---

# Physical AI 側のフレームワーク一覧（軽く）

| カテゴリ | 代表 |
|---|---|
| シミュレータ | Isaac Sim, MuJoCo MJX, Genesis, Habitat |
| VLA 基盤モデル | π0 / π0-FAST, OpenVLA, **GR00T N1** (NV), SmolVLA, Gemini Robotics |
| 世界モデル | **NVIDIA Cosmos**, Genie 3, 1X World Model |
| RL lib | RSL-RL, Stable-Baselines3, CleanRL |
| データ | Open X-Embodiment, DROID, HF Hub |

**どれを使っても W&B に繋がる**、が今日持ち帰って欲しいメッセージ。

<!--
聴衆のほうが詳しい可能性があるので深堀しない。
「名前だけは把握しておいてね」程度で通過。
-->

---

<!-- _class: lead -->
# 4. マルチメディア可視化
## loss だけでは robot は語れない

---

# Physical AI 開発のリアルな痛点

- **成功率 60% のポリシー同士**でも、失敗の仕方が全然違う
- sim2real ギャップは**動画でしか見つからない**
- カメラ・深度・点群など**観測自体がマルチモーダル**
- 「失敗 rollout 20 本だけ並べて見たい」という現場ニーズ

→ **W&B は最初からマルチメディアファースト**

---

# W&B Models：対応メディア一覧

| クラス | 用途 | 例 |
|---|---|---|
| `wandb.Image` | カメラ / 注意マップ / 失敗フレーム | 学習時の observation 確認 |
| `wandb.Video` | rollout / eval エピソード | PushT / Franka / ヒューマノイド |
| `wandb.Audio` | マイク観測 / UI 音声 | ALOHA のフィードバック音 |
| `wandb.Object3D` | point cloud / mesh / URDF | LiDAR / 深度カメラ |
| `wandb.Molecule` | (専門用途) | — |

```python
run.log({"rollout": wandb.Video("ep.mp4", fps=10)})
run.log({"pointcloud": wandb.Object3D(np.array(points))})
run.log({"mic": wandb.Audio("feedback.wav", sample_rate=16000)})
```

---

# デモ：W&B Tables で動画比較

```python
table = wandb.Table(columns=["episode", "success", "coverage", "video"])
for rec in records:
    table.add_data(rec.id, rec.success, rec.coverage,
                   wandb.Video(rec.video_path, format="mp4"))
run.log({"eval/rollouts": table})
```

→ UI 上で **20 エピソードを 1 画面で比較、動画プレビュー付き**、success でフィルタ可能

<!--
ここでフィルタで失敗だけ抽出、coverage で並び替え、動画ホバーで再生、
の流れをライブデモ。この画面が刺さる顧客は多い。
-->

---

# デモ：Object3D で 3D 観測

```python
run.log({
    "scene": wandb.Object3D({
        "type": "lidar/beta",
        "points": np.array([[x, y, z, r, g, b], ...]),
    })
})
```

- LiDAR / 深度 / URDF / mesh 対応
- **Sim2Real デバッグ**で "シム観測 vs 実機観測" を並列表示

<!--
Isaac Sim で得たデプス点群を一度ログして UI ツアーするだけで強烈。
-->

---

# Weave：マルチモーダル Trace

- `@weave.op()` で関数の入出力を自動追跡
- `Content.from_bytes(video_bytes, extension=".mp4")` で**動画をトレースに直接埋込**
- 画像 / 音声 / ファイル何でも扱える

```python
@weave.op()
def replay_rollout(episode, video_bytes):
    return {
        "success": episode["success"],
        "coverage": episode["max_coverage"],
        "video": Content.from_bytes(video_bytes, extension=".mp4"),
    }
```

---

# デモ：Weave Evaluation

```python
evaluation = weave.Evaluation(
    name="pusht-rollouts",
    dataset=episodes,
    scorers=[success_scorer, coverage_scorer,
             efficiency_scorer, failure_mode_scorer],
)
asyncio.run(evaluation.evaluate(model))
```

- Scorers は**並列実行**
- UI で**横並び比較テーブル + 個別 trace ドリルダウン**
- Models run と相互リンク（`set_wandb_run_context`）

<!--
scorer を 1 つ追加して再実行、UI が差分を自動表示する流れを見せる。
-->

---

<!-- _class: lead -->
# 5. Agentic RL
## OpenPipe ART × W&B Serverless RL

---

# なぜ Agentic 層が必要か

- Diffusion Policy / π0 は**反射的**：観測 → 行動を即出す
- 失敗からの**回復戦略**や**タスク分解**は苦手
- **LLM / VLM を supervisor / critic として挟む**と賢くなる
- 例：「失敗したら retry？ recovery_stroke？ abort？」を**学習で決めさせる**

---

# OpenPipe ART 概要

- OpenPipe 社の Agent Reinforcement Trainer
- 内部：vLLM（推論）+ Unsloth（LoRA）+ TRL + GRPO
- **Trajectory = OpenAI chat messages + scalar reward**
- ユースケース：tool-calling agent / ゲーム AI / 本ワークショップの critic

```python
traj = art.Trajectory(
    messages_and_choices=[
        {"role": "system", "content": "You are a robot supervisor."},
        {"role": "user", "content": "Rollout failed: no_contact."},
        choice,  # assistant 出力
    ],
    reward=1.0,  # 回復に成功したら 1.0
)
```

---

# 本ワークショップの ART 使い方

`art_critic.py` : 失敗 rollout → LLM → 回復アクション提案

```
[rollout summary] → LLM → {retry, recovery_stroke, abort}
                               │
                     実行→再rollout reward
                               │
                  GRPO / LoRA 更新
```

- Dry-run で GPU なし動作確認可
- 全 trajectory が Weave trace に
- LoRA checkpoint は Registry で管理

---

# VLA 対応拡張：VLM-Critic パターン

**画像を見て**回復判断させる：

```
[画像 + rollout統計] → Qwen2-VL / Gemma3 → {retry, recovery, abort}
                                                 │
                                   reward = 再rollout success
                                                 │
                                      GRPO / LoRA 更新
```

- ART の trajectory は OpenAI ImageContentBlock 対応 → **マルチモーダル入力そのまま**
- H100 1 枚で GRPO 可
- **π0-FAST（離散 action token 出力）まで行くと VLA 本体の RL fine-tune への入口**

---

# W&B Serverless RL が強い理由

```python
from art.serverless.backend import ServerlessBackend
backend = ServerlessBackend(api_key=os.environ["WANDB_API_KEY"])
model.register(backend)
```

- Trajectory / reward が**自動で W&B + Weave に記録**
- **Checkpoint が即 W&B Inference にデプロイ**される
  → 次ステップの rollout でそのまま推論できる
- 40% 低コスト / 2000 並列推論 / マネージドインフラ

**これが Physical AI 連携で今一番尖っている W&B 機能**

---

<!-- _class: lead -->
# まとめ

---

# 今日のテイクアウェイ

1. **W&B は Physical AI スタックの既存層にそのまま刺さる**
   - LeRobot / Isaac Lab / GR00T / OpenVLA / Cosmos-RL どれでもフラグ 1 つ
2. **マルチメディア**（動画 / 3D / 画像 / 音声）は最初から対応
3. **Weave** で rollout と critic を全部 trace
4. **Serverless RL** で Agentic ワークフローまで managed
5. 本ワークショップの全コード：[wandb-physicalai-workshop](https://github.com/wandb/physical-ai-workshop)

---

# 次の一歩（お客様向け）

- **既に LeRobot / Isaac Lab 使っている**チーム
  - フラグ 1 つで W&B 連携。今日のリポを叩き台に
- **VLA (π0 / GR00T) 触っている**チーム
  - LeRobot 経由で W&B の動画 / Artifact / Registry が即効く
- **Agentic / Critic 層**に興味あるチーム
  - ART + Serverless RL で managed GRPO

**ご相談：japan@wandb.com**

---

<!-- _class: lead -->
# Appendix

---

# 環境構築（抜粋）

```bash
# LeRobot + PushT
pip install "lerobot[pusht]==0.4.4" gym-pusht

# Isaac Lab (NGC container)
docker run -it --rm --gpus all nvcr.io/nvidia/isaac-lab:2.0.0 bash

# W&B + Weave + ART
pip install wandb weave openpipe-art
```

---

# 再現コマンド

```bash
# 1-5 を一気通貫
bash run_pipeline.sh 50000

# eval 以降だけ再実行
bash run_pipeline_eval_only.sh 50000
```

各ステップ：
- `scripts/01_train_diffusion_pusht.py`
- `scripts/03_evaluate.py`
- `scripts/05_promote_to_registry.py`
- `scripts/06_weave_evaluation.py`
- `scripts/07_art_critic.py`
- `scripts/08_isaac_lab_train.py`

---

# W&B 連携の深さ比較（詳細版）

| FW | scalar | config | checkpoint Art. | 動画 Art. | eval 動画 | Trajectory |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| LeRobot | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| Isaac Lab | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| GR00T | ✅ | ✅ | — | — | — | — |
| OpenVLA | ✅ | ✅ | — | — | ✅ | — |
| Cosmos-RL | ✅ | ✅ | — | — | — | — |
| OpenPi | ✅ | — | — | — | — | — |
| ART | ✅ | ✅ | **✅→Inference** | — | — | ✅ |

---

# 実ログスクショ (TODO)

最終化時に差し込み：

- W&B Models: loss curve / eval_success_rate / Artifact lineage
- W&B Tables: rollout 20 本の動画プレビュー
- Weave: trace tree / Evaluation 結果
- Registry: policy 昇格フロー
- Isaac Lab: 4096 並列環境の reward curve + mp4
