from .GAConvGRU import Model as GAConvGRU

# 兼容旧代码中 `from models import SOFTS` 的导入
SOFTS = GAConvGRU

__all__ = ["GAConvGRU", "SOFTS"]

