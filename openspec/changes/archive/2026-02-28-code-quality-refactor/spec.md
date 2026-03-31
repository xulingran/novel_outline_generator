# 代码质量重构规格说明

## 1. Config 延迟初始化规格

### 1.1 要求
- 所有模块级配置常量改为延迟初始化
- 首次访问时从环境变量加载
- 保持向后兼容的常量访问方式

### 1.2 接口规范
```python
# 模块级缓存变量（私有）
_txt_file: str | None = None
_output_dir: str | None = None
...

# getter 函数（现有，修改为延迟加载）
def get_txt_file() -> str:
    global _txt_file
    if _txt_file is None:
        _txt_file = os.getenv("TXT_FILE", "novel.txt")
    return _txt_file

# 向后兼容：保持常量可用
# 方式A: 使用 property 风格的类
class _ConfigWrapper:
    @property
    def TXT_FILE(self) -> str:
        return get_txt_file()
    ...

# 或方式B: 模块级 __getattr__
def __getattr__(name: str) -> Any:
    if name == "TXT_FILE":
        return get_txt_file()
    ...
```

### 1.3 行为验证
- `from config import TXT_FILE` 工作正常
- 修改环境变量后重新导入获取新值
- 同一进程内多次访问返回相同值（缓存）

---

## 2. HTTP Connection Pool 规格

### 2.1 类接口
```python
class HTTPConnectionPool:
    """HTTP 客户端连接池管理器"""

    def __init__(self):
        self._main_client: httpx.AsyncClient | None = None
        self._proxy_clients: dict[str, httpx.AsyncClient] = {}
        self._cleanup_registered: bool = False

    def get_client(
        self,
        proxy_url: str | None = None,
        limits: httpx.Limits | None = None,
        timeout: float = 60.0
    ) -> httpx.AsyncClient:
        """获取 HTTP 客户端"""

    async def close_all(self) -> None:
        """关闭所有客户端连接"""

    def register_cleanup(self, cleanup_func: Callable) -> None:
        """注册程序退出清理函数（仅一次）"""
```

### 2.2 OpenAIService 修改
```python
class OpenAIService(LLMService):
    def __init__(
        self,
        api_config: APIConfig,
        processing_config: ProcessingConfig,
        connection_pool: HTTPConnectionPool | None = None
    ):
        self._pool = connection_pool or _default_pool
        ...

    def _init_client(self) -> None:
        proxy_url = ...
        http_client = self._pool.get_client(proxy_url)
        self.client = AsyncOpenAI(..., http_client=http_client)
```

### 2.3 全局默认池
```python
# llm_service.py 模块级
_default_pool: HTTPConnectionPool | None = None

def get_default_connection_pool() -> HTTPConnectionPool:
    global _default_pool
    if _default_pool is None:
        _default_pool = HTTPConnectionPool()
    return _default_pool
```

---

## 3. Processing Pipeline 规格

### 3.1 管道类接口
```python
class ProcessingPipeline:
    """小说处理执行管道"""

    def __init__(self, service: NovelProcessingService):
        self.service = service
        self.state = service.processing_state

    async def execute(
        self,
        file_path: str,
        output_dir: str | None,
        resume: bool
    ) -> dict[str, Any]:
        """执行完整处理流程"""

    async def _execute_phase(
        self,
        phase_name: str,
        phase_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """执行单个阶段，统一处理进度和错误"""
```

### 3.2 阶段列表
1. **loading** - 加载和验证文件
2. **splitting** - 文本分块
3. **resuming** - 处理进度恢复
4. **processing** - 处理文本块
5. **merging** - 合并大纲
6. **saving** - 保存结果

### 3.3 process_novel 简化后
```python
async def process_novel(...):
    # 初始化
    self._reset_token_stats()
    self._init_state(file_path)

    # 执行管道
    pipeline = ProcessingPipeline(self)
    return await pipeline.execute(file_path, output_dir, resume)
```

---

## 4. @with_retry 装饰器规格

### 4.1 接口定义
```python
def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable:
    """重试装饰器

    Args:
        max_attempts: 最大尝试次数
        backoff_base: 退避基数（秒）
        exceptions: 需要重试的异常类型
        on_retry: 重试回调函数 (exception, attempt)
    """
```

### 4.2 使用示例
```python
@with_retry(
    max_attempts=3,
    backoff_base=2.0,
    exceptions=(APIError, ProcessingError),
    on_retry=lambda e, a: logger.warning(f"Retry {a}: {e}")
)
async def _process_single_chunk(self, chunk: TextChunk, ...) -> dict[str, Any]:
    # 纯业务逻辑，无重试代码
    ...
```

### 4.3 重试逻辑规格
- 第 n 次重试等待时间：`backoff_base * (n-1)` 秒
- 只捕获指定异常类型，其他异常立即抛出
- 最后一次失败后抛出原异常
- 支持同步和异步函数

---

## 5. Web API 提取函数规格

### 5.1 _log_token_usage
```python
def _log_token_usage(
    job_or_task: Job | QueueTask,
    token_usage: dict[str, int],
    context: str = "处理完成"
) -> None:
    """记录 Token 使用统计

    Args:
        job_or_task: Job 或 QueueTask 对象
        token_usage: {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        context: 日志上下文描述
    """
```

### 5.2 _update_progress_from_info
```python
def _update_progress_from_info(
    job_or_task: Job | QueueTask,
    info: dict[str, Any]
) -> None:
    """从进度信息更新任务状态

    Args:
        job_or_task: Job 或 QueueTask 对象
        info: 进度信息字典
    """
```

### 5.3 统一处理函数
```python
def _handle_processing_result(
    job_or_task: Job | QueueTask,
    result: dict[str, Any]
) -> None:
    """处理处理完成后的通用逻辑"""
    # Token 日志
    # 状态更新
    # 文件清理
```

---

## 6. Utils 导入清理规格

### 6.1 检查项
- [ ] 同一模块是否导入多次
- [ ] 是否有未使用的导入
- [ ] 是否存在循环导入风险

### 6.2 修复方式
- 合并重复导入
- 删除未使用的导入
- 延迟导入（如需要避免循环）
