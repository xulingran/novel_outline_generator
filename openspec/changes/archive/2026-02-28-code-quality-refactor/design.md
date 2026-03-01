# 代码质量重构设计

## 1. Config 模块重构

### 当前问题
模块导入时执行配置加载，产生全局状态副作用。

### 设计方案
```
延迟初始化模式
═══════════════════════════════════════════════════════════════

修改前:                          修改后:
┌─────────────────┐             ┌─────────────────┐
│ TXT_FILE =      │             │ _txt_file = None│
│   get_xxx()     │             │                 │
│ (导入时执行)     │             │ def get_txt_file│
└─────────────────┘             │   if _txt_file  │
         │                      │      is None:   │
         ▼                      │     _txt_file = │
  配置立即固定                  │       _load()   │
                                │   return _txt_file
                                └─────────────────┘
                                         │
            ┌────────────────────────────┘
            ▼
  每次调用都检查，首次加载
  保持向后兼容：常量引用转为函数调用
```

### 实现细节
- 添加模块级 `_xxx = None` 缓存变量
- 修改 getter 函数使用缓存
- 保持原有全局常量名，转为属性访问或延迟计算

---

## 2. LLM Service HTTP 连接池重构

### 当前问题
OpenAIService 使用类级静态变量管理 HTTP 客户端，多实例可能产生竞争。

### 设计方案
```
连接池管理器提取
═══════════════════════════════════════════════════════════════

修改前:                          修改后:
┌─────────────────────┐        ┌─────────────────────┐
│ OpenAIService       │        │ HTTPConnectionPool  │
│ ├─ _http_client     │        │ ├─ _clients: dict   │
│ ├─ _proxy_clients   │        │ ├─ _main_client     │
│ ├─ _cleanup_reg     │        │ ├─ get_client()     │
│ ├─ get_http_client()│        │ └─ close_all()      │
│ └─ close_http...()  │        └──────────┬──────────┘
└─────────────────────┘                   │
                                          │ 依赖注入
                                          ▼
                              ┌─────────────────────┐
                              │ OpenAIService       │
                              │ ├─ _conn_pool       │
                              │ └─ _init_client()   │
                              │   pool.get_client() │
                              └─────────────────────┘
```

### 实现细节
- 创建 `HTTPConnectionPool` 类管理所有客户端
- OpenAIService 通过构造函数接收 pool 实例
- 默认 pool 为全局单例，保持简单使用
- 生命周期管理由 pool 负责

---

## 3. Novel Processing Service 执行管道重构

### 当前问题
`process_novel()` 函数约 130 行，协调过多步骤。

### 设计方案
```
执行管道模式
═══════════════════════════════════════════════════════════════

┌───────────────────────────────────────────────────────────────┐
│                    NovelProcessingService                     │
│                         process_novel                         │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                  ProcessingPipeline                           │
├───────────────────────────────────────────────────────────────┤
│  execute()                                                    │
│    ├── _execute_phase("loading", _load)                      │
│    ├── _execute_phase("processing", _process)                │
│    ├── _execute_phase("merging", _merge)                     │
│    └── _execute_phase("saving", _save)                       │
└───────────────────────────────────────────────────────────────┘

每个 phase:
  - 统一错误处理
  - 自动进度报告
  - 状态转换
```

### 实现细节
- 提取 `ProcessingPipeline` 内部类或独立类
- 每个阶段为独立方法
- 统一阶段执行包装器处理进度和错误

---

## 4. 统一错误处理和重试装饰器

### 当前问题
`_process_chunks` 和 `_process_single_chunk` 重复实现重试逻辑。

### 设计方案
```
@with_retry 装饰器
═══════════════════════════════════════════════════════════════

修改前 (内联重试):              修改后 (装饰器):
async def _process(...):        @with_retry(
  for attempt in range(...):      max_attempts=3,
    try:                          backoff_base=2.0,
      ...                         exceptions=(APIError, ...)
    except ...:                 )
      if attempt < ...:         async def _process(...):
        sleep(...)                  ... # 纯业务逻辑
      else:
        raise
```

### 实现细节
- 创建 `with_retry` 装饰器函数
- 支持参数：max_attempts, backoff_base, exceptions
- 保持与原逻辑一致：指数退避、异常类型过滤
- 可选：支持 jitter 防止惊群

---

## 5. Web API 重复代码提取

### 当前问题
`_run_job` 和 `run_queue_task` 中 Token 统计日志和进度更新逻辑重复。

### 设计方案
```
提取共享函数
═══════════════════════════════════════════════════════════════

原代码: _run_job()              提取为:
  if info.get("token_usage"):   def _log_token_usage(
    prompt = ...                  job_or_task, token_usage
    completion = ...            ):
    total = ...                     # 统一日志格式
    job.log(...)                    job.log(f"Token统计...")

                                def _update_progress_from_info(
                                  job_or_task, info
                                ):
                                  # 统一进度更新

两个函数合并使用:
  _log_token_usage(job, result.get("token_usage"))
  _update_progress_from_info(job, progress_info)
```

### 实现细节
- 提取 `_log_token_usage(job_or_task, token_usage)`
- 提取 `_update_job_from_info(job_or_task, info)`
- 支持 Job 和 QueueTask 两种类型

---

## 6. Utils 导入清理

### 当前问题
行号 17-22 被标记为双重导入，需确认具体问题。

### 检查策略
1. 检查同一模块是否被导入两次
2. 检查是否有循环导入风险
3. 清理未使用的导入

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `config.py` | 修改 | 延迟初始化配置常量 |
| `services/connection_pool.py` | 新增 | HTTP 连接池管理器 |
| `services/llm_service.py` | 修改 | 使用连接池管理器 |
| `services/novel_processing_service.py` | 修改 | 执行管道重构 |
| `decorators.py` | 新增 | @with_retry 装饰器 |
| `web_api.py` | 修改 | 提取重复函数 |
| `utils.py` | 修改 | 清理导入 |
