# EEGNet EEG 学习状态识别系统

这是一个面向实时部署的 EEG 三分类项目。系统从 5 通道、4 秒窗口的 EEG 信号中识别 3 类学习状态，核心神经网络基于 EEGNet，并配套严格的数据预处理、交叉验证、概率校准、模型完整性校验和 FastAPI 服务。

## 项目概览：STAR

### Situation：问题背景

原始数据形状为 `[N, 5, 1000]`，采样率为 250 Hz，共 3,196 个样本、3 个类别。类别分布不均衡，少数类更难识别；同时，EEG 相邻窗口具有相关性，随机切分很容易把同一段采集数据同时放进训练集和测试集，造成指标虚高。

### Task：交付目标

- 保持 EEGNet 作为核心神经网络，完成端到端训练和推理。
- 将训练、验证、校准、推理和部署使用的预处理严格对齐。
- 提供可复现的 OOF 评估，避免把调参结果误报为测试成绩。
- 提供企业服务需要的模型清单、SHA-256 完整性校验、限流、日志和 Docker 配置。

### Action：实现方案

1. **信号处理**：4-40 Hz 零相位带通和 50 Hz 陷波；每个训练折单独计算归一化统计量，推理时只读取对应折的统计量。
2. **EEGNet 主干**：使用长时间卷积核、深度可分离卷积、SE 通道注意力、紧凑分类头和类别校准，保留轻量网络特性。
3. **辅助信息融合**：使用 FBCSP + ExtraTrees 作为显式辅助分支，和 EEGNet 概率在二级 OOF 验证中融合。它被明确标记为辅助模型，不会被称作纯 EEGNet。
4. **防泄漏评估**：外层 OOF 产生未见样本概率，二级策略只在开发行拟合融合权重、温度和类别倍率，再在元验证折上评估。
5. **企业部署**：默认清单为 `artifacts/production/manifest.json`；FastAPI 接收原始 EEG，自动执行滤波、归一化、融合和预测；启动时校验模型和清单哈希。

### Result：当前结果

当前生产清单在样本级五折二级 OOF 评估中的结果：

| 指标 | 结果 |
| --- | ---: |
| 准确率 | **82.10%** |
| 准确率 95% bootstrap 区间 | 80.76%-83.39% |
| 平衡准确率 | 69.07% |
| Macro-F1 | 72.73% |
| EEGNet 概率权重 | 22.5% |
| 辅助分支概率权重 | 77.5% |

这些是样本级 OOF 结果，不是跨受试者结果。当前数据没有提供受试者或会话 ID，因此不能据此宣称对新受试者达到 90%。纯 EEGNet 目前也没有经过严格验证的 80% 以上结果；如需企业验收，必须补充 `subject_id`/`session_id` 后运行 group-level 验证。

## 快速开始

### 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 使用现有生产模型启动服务

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000
```

检查服务：

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/metadata
```

### 从数据重新构建生产模型

以下命令会依次训练 EEGNet、生成 EEGNet 集成清单、训练辅助分支并生成最终融合清单：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_production_pipeline.ps1
```

最终部署入口固定为：

```text
artifacts/production/manifest.json
```

### Docker 部署

```powershell
docker compose up --build
```

可通过 `.env` 配置 `EEG_API_KEY`、`EEG_MAX_BATCH_SIZE`、`EEG_MAX_CONCURRENT_INFERENCES` 和 `EEG_VERIFY_ARTIFACTS`。

## API

请求 `POST /predict` 的格式为：

```json
{
  "samples": [[[0.0, 0.0]]]
}
```

实际请求必须是 `[N, 5, 1000]`。服务返回类别、原始概率、校准概率和置信度。生产环境建议设置 `EEG_API_KEY`，并使用请求 ID 关联日志。

## 目录结构

```text
EEG_Project/
├── api.py                         # FastAPI 服务
├── inference.py                   # EEGNet、辅助模型和融合推理
├── train.py                       # EEGNet 五折训练
├── preprocess.py                  # EEG 预处理
├── tflite_quantize.py             # EEGNet 边缘导出
├── test_pipeline.py               # 自动化测试
├── arl_eegmodels/                 # EEGNet 网络定义
├── eeg_project/                   # 配置、训练、校准、指标和特征模块
├── configs/
│   ├── eegnet.yaml                # EEGNet 主配置
│   └── fbcsp_auxiliary.yaml       # 辅助分支配置
├── scripts/
│   ├── run_production_pipeline.ps1
│   ├── build_mixed_ensemble.py
│   ├── build_eegnet_fbcsp_hybrid.py
│   └── train_fbcsp.py
├── artifacts/production/          # 唯一发布模型目录
└── docs/                          # 运维说明和项目材料
```

## 数据与验收要求

生产验收应为 group-level 交叉验证，而不是随机样本切分。建议数据文件额外提供：

```text
data/subject_ids.npy
data/session_ids.npy
```

然后在配置中指定：

```yaml
groups: data/subject_ids.npy
split_mode: group
repeats: 3
selection_metric: macro_f1
```

只有当多个独立 group-level 测试中准确率稳定达到目标，并且少数类召回率、Macro-F1 和置信区间同时满足要求时，才应将模型用于客户验收。窗口级准确率达到目标，不代表跨受试者泛化达到目标。

## 测试

```powershell
python -m pytest -q
```

当前代码包含配置校验、滤波一致性、归一化隔离、OOF 融合、API 健康检查和推理预处理测试。

## 发布原则

- 只从 `artifacts/production/manifest.json` 加载默认模型。
- `EEG_VERIFY_ARTIFACTS=true` 时，模型哈希不匹配会阻止服务启动。
- 不把开发集调参结果写成测试结果。
- 不在没有 group ID 的情况下宣称跨受试者准确率。
