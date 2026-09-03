# 生产运维说明

## 发布入口

生产服务只读取：

```text
artifacts/production/manifest.json
```

该清单包含五折 EEGNet 集成和 FBCSP 辅助分支的引用、权重、校准参数及哈希校验信息。不要直接指定某个折模型作为线上默认模型。

## 构建流程

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_production_pipeline.ps1
```

流程依次完成预处理、EEGNet 五折训练、OOF 集成清单、辅助分支训练和最终概率融合。构建前应确认 `data/X_features.npy`、`data/y_labels.npy` 与配置中的样本数一致。

## 验收协议

样本级 OOF 结果仅用于研发比较。客户验收必须提供与每条样本对齐的受试者或会话分组：

```yaml
groups: data/subject_ids.npy
split_mode: group
repeats: 3
selection_metric: macro_f1
bootstrap_iterations: 2000
```

没有 group ID 时，不得将相邻窗口随机分到训练集和测试集后宣称跨受试者泛化。验收同时报告准确率、平衡准确率、Macro-F1、少数类召回率和置信区间。

## 服务控制

- `/live`：进程存活检查。
- `/health`、`/ready`：模型是否完成加载。
- `/metadata`：输入形状、类别和当前推理方法。
- `/predict`：接收原始 `[N, 5, 1000]` EEG。
- `/metrics`：Prometheus 文本格式的请求和延迟指标。

生产环境建议设置：

```text
EEG_API_KEY=<secret>
EEG_VERIFY_ARTIFACTS=true
EEG_MAX_CONCURRENT_INFERENCES=1
EEG_JSON_LOGS=true
```

服务启动时会校验模型和清单 SHA-256；文件被替换或损坏时拒绝加载。日志只记录请求元数据和延迟，不记录原始 EEG。

## 结果边界

当前清单的样本级二级 OOF 准确率为 82.10%，平衡准确率为 69.07%，Macro-F1 为 72.73%。这不是跨受试者指标，也不代表纯 EEGNet 达到同样准确率。任何 90% 目标都必须在锁定的 group-level 测试集上重新验证。
