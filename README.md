# EEG 学习状态识别系统
> 将 5 通道 EEG 信号转换为学习状态预测的端到端项目。项目覆盖信号处理、模型训练、可靠性评估和在线部署，最终以 FastAPI + Docker 形式提供推理服务。

## 项目概述

真实 EEG 数据通常同时存在三个问题：类别分布不均衡、相邻窗口相互相关，以及离线训练和线上推理容易使用不同的预处理流程。单纯追求一个更高的准确率，不能说明模型能够稳定交付。

围绕这个问题，我把项目推进成了一条完整链路：统一原始信号处理与推理输入，使用轻量 EEGNet 建立主模型，再加入 FBCSP + ExtraTrees 辅助分支捕捉互补特征；通过 OOF（Out-of-Fold）和二级交叉拟合评估融合策略，最后将模型、校准参数和完整性校验信息打包为唯一生产清单。

**项目亮点**

- **从实验到服务**：不是只提交一个训练脚本，而是交付可复现的训练流水线、模型制品、API 和 Docker 配置。
- **评估口径清晰**：二级评估中的融合策略、温度和类别倍率只在开发行拟合，避免把调参结果误报成测试成绩。
- **工程风险可控**：启动时校验模型 SHA-256，接口具备输入校验、API Key、并发限制、请求日志和 Prometheus 指标。

## 当前结果

生产清单对应的是 **5 折样本级二级 OOF 评估**，不是跨受试者测试。结果属于 EEGNet 与 FBCSP 辅助分支的融合方案，不将融合成绩归因于单一模型。

| 指标 | 结果 |
| --- | ---: |
| 准确率 | **82.10%** |
| 准确率 95% Bootstrap 区间 | **80.76% - 83.39%** |
| 平衡准确率 | 69.07% |
| Macro-F1 | 72.73% |
| 融合权重 | EEGNet 22.5% / FBCSP 77.5% |

数据规模为 3,196 个样本、3 个类别，每条样本包含 5 通道、1,000 个采样点（250 Hz，即 4 秒窗口）。由于当前数据没有 `subject_id` 或 `session_id`，结果只能说明窗口级识别能力，不能直接宣称对新受试者的泛化效果。企业验收前应补充分组信息，并运行 group-level 交叉验证。

## 核心工作

### 1. 统一信号处理，保证训练和推理一致

- 对原始 EEG 执行 4-40 Hz 零相位带通滤波和 50 Hz 陷波滤波。
- 每个训练折只用训练数据计算归一化均值和标准差，验证、测试和线上推理只读取已保存的统计量。
- 将滤波参数、输入形状、类别顺序和归一化信息随模型一起写入预处理制品，避免部署后出现隐性分布偏移。

### 2. 用互补模型处理类别不均衡和信号差异

- EEGNet 主干使用时间卷积、空间深度卷积、可分离卷积和 SE 通道注意力，在保持轻量的同时提取时空特征。
- FBCSP + ExtraTrees 作为显式辅助分支，从多个频带提取 CSP 特征，补充传统频域/空间判别信息。
- 在未见样本概率上学习融合权重、温度参数和类别倍率，并保留各分支的独立结果，便于定位问题和解释模型行为。

### 3. 用 OOF 和二级交叉拟合约束评估偏差

- 外层五折交叉验证为每个样本生成未见过该样本的预测概率。
- 二级融合再次划分开发行与元验证折，只在开发行拟合策略，再对元验证折评估。
- 同时报告 Accuracy、Balanced Accuracy、Macro-F1 和 Bootstrap 置信区间，不只看总体准确率。

### 4. 将模型包装成可上线的推理服务

- FastAPI 接收原始 `[N, 5, 1000]` EEG，自动执行滤波、归一化、模型融合和概率校准。
- 提供 `/live`、`/health`、`/ready`、`/metadata`、`/predict` 和 `/metrics` 接口。
- 支持 API Key、批大小和并发上限、请求 ID、结构化日志，以及模型制品 SHA-256 完整性校验。
- 使用 `artifacts/production/manifest.json` 作为唯一发布入口，避免线上直接引用某个折模型。

## 技术栈

`Python 3.11` · `TensorFlow/Keras` · `SciPy` · `scikit-learn` · `FastAPI` · `Docker` · `pytest`

## 快速开始

### 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 启动现有生产模型

仓库已包含 `artifacts/production/manifest.json` 及其引用的模型制品，可直接启动服务：

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

检查服务状态：

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/metadata
```

### 重新训练并生成生产清单

准备好以下数据文件后运行完整流水线：

```text
data/X_features.npy    # 原始 EEG，形状 [N, 5, 1000]
data/y_labels.npy      # 与样本一一对应的标签
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_production_pipeline.ps1
```

流水线会依次完成预处理、EEGNet 五折训练、OOF 集成、FBCSP 辅助分支训练和二级概率融合。最终部署入口固定为：

```text
artifacts/production/manifest.json
```

### Docker 部署

```powershell
docker compose up --build
```

生产环境可通过 `.env` 配置 `EEG_API_KEY`、`EEG_MAX_BATCH_SIZE`、`EEG_MAX_CONCURRENT_INFERENCES` 和 `EEG_VERIFY_ARTIFACTS`。

## API 示例

`POST /predict` 接收一个或多个 EEG 窗口：

```json
{
  "samples": [
    [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0]
    ]
  ]
}
```

示例仅展示 JSON 层级；实际请求必须使用 `[N, 5, 1000]`。服务返回预测类别、原始概率、校准概率、置信度和请求延迟。设置 `EEG_API_KEY` 后，请在请求头中携带 `X-API-Key`。

## 目录结构

```text
EEG_Project/
├── api.py                         # FastAPI 推理服务
├── inference.py                   # 单模型、集成和融合推理
├── preprocess.py                  # EEG 滤波与数据校验
├── train.py                       # EEGNet 交叉验证训练
├── tflite_quantize.py             # EEGNet 边缘端导出
├── arl_eegmodels/                 # EEGNet 网络定义
├── eeg_project/                   # 训练、特征、校准、指标和清单模块
├── configs/                       # EEGNet 与 FBCSP 配置
├── scripts/                       # 生产训练、融合和发布脚本
├── artifacts/production/          # 可发布模型及唯一 manifest
├── docs/OPERATIONS.md             # 运维与验收说明
└── test_pipeline.py               # 自动化测试
```

## 验证与发布边界

运行自动化测试：

```powershell
python -m pytest -q
```

测试覆盖配置校验、滤波一致性、训练统计量隔离、OOF 覆盖、融合策略、模型输入校验和 API 健康检查。

发布时遵循以下原则：

- 默认只从 `artifacts/production/manifest.json` 加载模型。
- `EEG_VERIFY_ARTIFACTS=true` 时，任一模型或预处理文件哈希不匹配都会阻止服务启动。
- 样本级 OOF 结果用于研发比较；客户验收必须提供受试者或会话分组，并报告 group-level 指标。
- 任何跨受试者准确率或更高业务目标，都需要在锁定的独立测试协议上重新验证。

更详细的运维命令和验收配置见 [`docs/OPERATIONS.md`](docs/OPERATIONS.md)。
