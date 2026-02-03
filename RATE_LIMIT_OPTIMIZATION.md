# 限流优化方案说明

## 修改概述

采用**方案5（组合方案）**对代码进行了全面优化，以解决API限流问题。

## 修改内容

### 1. ChatGPT类优化 ([src/llms/chatgpt.py](src/llms/chatgpt.py))

#### 1.1 新增功能
- **请求前延迟**：每次请求前自动等待 0.5-0.8 秒（可配置）
- **智能错误识别**：区分限流、超时、其他错误
- **指数退避策略**：根据错误类型使用不同的等待时间
- **随机化延迟**：避免多线程同时重试

#### 1.2 具体改进

**新增参数：**
```python
--request_delay 0.5  # 每次请求前的延迟（秒），默认0.5秒
```

**请求前延迟（第78-81行）：**
```python
if self.request_delay > 0:
    delay = self.request_delay + random.uniform(0, 0.3)  # 0.5-0.8秒
    time.sleep(delay)
```

**智能重试策略（第102-130行）：**
- **限流错误**：60秒 × 2^重试次数 + 随机0-10秒
  - 第1次重试：60秒
  - 第2次重试：120秒
  - 第3次重试：240秒
  - 第4次重试：480秒
  - 第5次重试：960秒

- **超时错误**：30秒 + 随机0-10秒

- **其他错误**：30秒 + 随机0-5秒

**改进的日志输出：**
- 显示重试次数：`Request failed (attempt 1/6)`
- 截断长消息：只显示前200字符
- 明确错误类型：`Rate limit detected` / `Timeout detected`
- 显示等待时间：`Waiting 60.5 seconds before retry...`

### 2. Shell脚本优化

#### 2.1 graph_inductive_reasoning.sh
```bash
N_THREAD=3              # 从10降低到3
REQUEST_DELAY=0.5       # 新增：请求延迟
```

#### 2.2 graph_inductive_reasoning_batch.sh
```bash
N_THREAD=3              # 从10降低到3
REQUEST_DELAY=0.5       # 新增：请求延迟
K_WAIT_TIME=20          # 新增：k值之间等待20秒
```

**k值之间增加等待：**
```bash
if [ $k != ${K_VALUES[-1]} ]; then
    echo "Waiting ${K_WAIT_TIME} seconds before next k value..."
    sleep ${K_WAIT_TIME}
fi
```

## 优化效果对比

### 修改前
- 并发数：10
- 请求间隔：无
- 重试等待：固定30秒
- 多线程问题：10个线程同时重试，加剧限流

### 修改后
- 并发数：3（降低70%）
- 请求间隔：0.5-0.8秒（随机）
- 重试等待：智能指数退避（60秒起）
- 多线程优化：随机延迟避免同时重试
- k值间隔：20秒缓冲

## 预期改进

1. **限流错误减少 80%+**
2. **成功率提升至 95%+**
3. **运行时间增加约 2-3倍**（但成功率大幅提升）

## 使用方法

### 基础使用（使用默认配置）
```bash
bash scripts/graph_inductive_reasoning.sh
```

### 自定义配置
```bash
# 修改脚本中的参数
N_THREAD=2              # 更保守：降低到2
REQUEST_DELAY=1.0       # 更保守：增加到1秒
K_WAIT_TIME=30          # 更保守：增加到30秒
```

### Python直接调用
```bash
python workflow/predict_final_answer.py \
    --data_path ./offline_assets/datasets \
    --d industry-kg \
    --split train \
    --model_name gpt-4o \
    --reasoning_path results/GenPaths/... \
    --add_path True \
    -n 3 \
    --request_delay 0.5
```

## 参数调优建议

### 如果仍然限流严重
```bash
N_THREAD=1              # 降低到1
REQUEST_DELAY=1.5       # 增加到1.5秒
K_WAIT_TIME=60          # 增加到60秒
```

### 如果限流改善，想加快速度
```bash
N_THREAD=5              # 提高到5
REQUEST_DELAY=0.3       # 降低到0.3秒
K_WAIT_TIME=10          # 降低到10秒
```

### 如果使用 gpt-4o-mini（限流通常更宽松）
```bash
MODEL_NAME=gpt-4o-mini
N_THREAD=5              # 可以提高到5
REQUEST_DELAY=0.3       # 可以降低到0.3秒
```

## 监控建议

运行时注意观察：
1. **限流错误频率**：如果频繁出现 "Rate limit detected"，需要降低并发或增加延迟
2. **重试次数**：如果经常重试3次以上，说明配置仍需优化
3. **成功率**：目标是 95% 以上的请求成功

## 回滚方法

如果需要恢复原始配置：
```bash
N_THREAD=10
REQUEST_DELAY=0
# 删除 K_WAIT_TIME 相关代码
# 在 chatgpt.py 中移除请求前延迟和指数退避逻辑
```

## 注意事项

1. **运行时间会变长**：由于降低了并发数和增加了延迟，总运行时间会增加
2. **保留了原有重试机制**：仍然最多重试5次
3. **兼容性**：所有修改向后兼容，不影响其他功能
4. **日志更详细**：可以更好地追踪限流问题

## 技术细节

### 指数退避算法
```
限流错误等待时间 = 60 × 2^重试次数 + random(0, 10)
```

这确保了：
- 第1次快速重试（60秒）
- 后续重试间隔指数增长
- 给服务器充足的恢复时间

### 随机化策略
所有延迟都加入了随机因子，避免：
- 多个线程同步重试
- 请求模式过于规律被识别
- 雪崩效应（所有线程同时失败又同时重试）
