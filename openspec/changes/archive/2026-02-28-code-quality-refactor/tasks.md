# 代码质量重构任务列表

## 任务总览

| ID | 任务 | 优先级 | 预估工时 | 依赖 |
|----|------|--------|----------|------|
| T1 | 创建 HTTPConnectionPool 类 | P0 | 2h | - |
| T2 | 修改 OpenAIService 使用连接池 | P0 | 1h | T1 |
| T3 | Config 延迟初始化重构 | P0 | 2h | - |
| T4 | 创建 @with_retry 装饰器 | P1 | 1.5h | - |
| T5 | 重构 _process_single_chunk 使用装饰器 | P1 | 1h | T4 |
| T6 | 创建 ProcessingPipeline 类 | P1 | 2h | - |
| T7 | 重构 process_novel 使用管道 | P1 | 1.5h | T6 |
| T8 | Web API 提取重复函数 | P2 | 1h | - |
| T9 | Utils 导入清理 | P2 | 0.5h | - |
| T10 | 运行测试和代码检查 | P0 | 1h | All |

---

## 详细任务

### T1: 创建 HTTPConnectionPool 类
**文件**: `services/connection_pool.py` (新建)

**验收标准**:
- [x] 类实现 `get_client(proxy_url)` 方法
- [x] 类实现 `close_all()` 方法
- [x] 支持代理客户端管理
- [x] 支持清理函数注册（仅一次）

**实现要点**:
```python
class HTTPConnectionPool:
    _instance: ClassVar[HTTPConnectionPool | None] = None

    def __new__(cls) -> HTTPConnectionPool:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._main_client: httpx.AsyncClient | None = None
        self._proxy_clients: dict[str, httpx.AsyncClient] = {}
        self._cleanup_registered = False
        self._initialized = True
```

---

### T2: 修改 OpenAIService 使用连接池
**文件**: `services/llm_service.py`

**修改内容**:
- [x] 移除类级静态变量 `_http_client`, `_proxy_clients`, `_cleanup_registered`
- [x] 添加 `_pool: HTTPConnectionPool` 实例变量
- [x] 修改 `_init_client()` 使用 `self._pool.get_client()`
- [x] 修改 `close_http_clients()` 委托给 pool

**向后兼容**:
- 默认使用全局连接池实例
- 允许通过构造函数注入自定义 pool

---

### T3: Config 延迟初始化重构
**文件**: `config.py`

**修改内容**:
- [x] 添加模块级缓存变量（`_config_cache`）
- [x] 使用 `_LazyConfig` 类实现延迟加载
- [x] 使用 `__getattr__` 保持向后兼容
- [x] 保持所有常量名可用

**实现要点**:
```python
# 缓存变量
_txt_file: str | None = None
_output_dir: str | None = None
...

# 修改后的 getter
def get_txt_file() -> str:
    global _txt_file
    if _txt_file is None:
        _txt_file = os.getenv("TXT_FILE", "novel.txt")
    return _txt_file

# 向后兼容
def __getattr__(name: str) -> Any:
    mapping = {
        "TXT_FILE": get_txt_file,
        "OUTPUT_DIR": get_output_dir,
        ...
    }
    if name in mapping:
        return mapping[name]()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

---

### T4: 创建 @with_retry 装饰器
**文件**: `decorators.py` (新建)

**验收标准**:
- [x] 支持异步函数
- [x] 支持同步函数
- [x] 指数退避实现
- [x] 可配置异常类型

**实现要点**:
```python
import functools
import asyncio
from typing import TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")

def with_retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts:
                        await asyncio.sleep(backoff_base * (attempt - 1))
            raise last_error
        return async_wrapper  # type: ignore
    return decorator
```

---

### T5: 重构 _process_single_chunk 使用装饰器
**文件**: `services/novel_processing_service.py`

**修改内容**:
- [x] 导入 `with_retry` 装饰器
- [x] 移除内联重试逻辑
- [x] 添加装饰器到 `_try_process_chunk`

**修改后代码结构**:
```python
@with_retry(
    max_attempts=3,
    backoff_base=2.0,
    exceptions=(APIError, ProcessingError),
)
async def _process_single_chunk(self, chunk: TextChunk, ...) -> dict[str, Any]:
    # 取消检查
    if self.cancel_event.is_set():
        raise asyncio.CancelledError()

    async with sem:
        # 业务逻辑（无重试代码）
        prompt = chunk_prompt(chunk.content, chunk.id)
        llm_response = await self.llm_service.call(prompt, chunk_id)
        ...
```

---

### T6: 创建 ProcessingPipeline 类
**文件**: `services/processing_pipeline.py` (新建)

**验收标准**:
- [x] 类接收 NovelProcessingService 作为参数
- [x] 实现 execute() 主流程
- [x] 实现 _execute_phase() 包装器
- [x] 保持与原 process_novel 行为一致

**实现要点**:
```python
class ProcessingPipeline:
    def __init__(self, service: NovelProcessingService):
        self.service = service
        self.state = service.processing_state
        self.config = service.processing_config

    async def execute(self, file_path: str, output_dir: str | None, resume: bool) -> dict[str, Any]:
        # 1. 加载
        text, encoding = await self._execute_phase("loading", self._load_file, file_path)

        # 2. 分块
        chunks = await self._execute_phase("splitting", self._split_text, text)

        # 3. 恢复/处理
        outlines = await self._handle_processing(chunks, resume, encoding)

        # 4. 合并
        final_outline = await self._execute_phase("merging", self._merge, outlines)

        # 5. 保存
        await self._execute_phase("saving", self._save, outlines, final_outline, file_path, output_dir)

        return self._build_result(outlines, final_outline, chunks)

    async def _execute_phase(self, phase_name: str, phase_func: Callable, *args, **kwargs):
        self.state.current_phase = phase_name
        self.service._emit_progress()
        try:
            return await phase_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Phase {phase_name} failed: {e}")
            raise
```

---

### T7: 重构 process_novel 使用管道
**文件**: `services/novel_processing_service.py`

**修改内容**:
- [x] 导入 ProcessingPipeline
- [x] 简化 process_novel 为初始化 + 管道执行
- [x] 将原逻辑迁移为 pipeline 的方法

**修改后代码结构**:
```python
async def process_novel(self, file_path: str, output_dir: str | None = None, resume: bool = True) -> dict[str, Any]:
    # 初始化
    self._reset_token_stats()
    self.processing_state = ProcessingState(file_path=file_path, total_chunks=0)
    self.processing_state.current_phase = "loading"
    self._emit_progress()

    try:
        # 执行管道
        from .processing_pipeline import ProcessingPipeline
        pipeline = ProcessingPipeline(self)
        result = await pipeline.execute(file_path, output_dir, resume)

        # 完成
        self.processing_state.complete()
        self._emit_progress()
        return result

    except asyncio.CancelledError:
        logger.info("Novel processing cancelled")
        raise
    except Exception as e:
        if self.processing_state:
            self.processing_state.fail(str(e))
            self._emit_progress()
        raise ProcessingError(f"处理小说失败: {str(e)}") from e
```

---

### T8: Web API 提取重复函数
**文件**: `web_api.py`

**修改内容**:
- [x] 提取 `_log_token_usage(job_or_task, token_usage)`
- [x] 复用 `_update_progress_from_info(job_or_task, info)`
- [x] 修改 `_run_job` 使用提取的函数
- [x] 修改 `run_queue_task` 使用提取的函数

---

### T9: Utils 导入清理
**文件**: `utils.py`

**检查项**:
- [x] 检查是否存在重复导入（代码审查误报，实际不存在）
- [x] 运行 Ruff 检查导入问题

---

### T10: 运行测试和代码检查
**命令**:
```bash
# 1. Ruff 检查
.venv/Scripts/python -m ruff check . --fix

# 2. Black 格式化
.venv/Scripts/python -m black .

# 3. Mypy 类型检查
.venv/Scripts/python -m mypy .

# 4. Pytest 测试
.venv/Scripts/python -m pytest tests/ -v
```

**验收标准**:
- [x] 所有测试通过（543 个测试）
- [x] Ruff 无错误
- [x] Black 无格式化问题
- [x] Mypy 仅有类型警告（非阻塞）
