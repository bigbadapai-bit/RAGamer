"""界面：服务端渲染的页面（ADR-0005）。

对外只有 `create_router`：给它一个组合根，它交回一个挂好页面的路由器。模板在同目录的
`templates/`，跟着包走，装成 wheel 也在。
"""

from __future__ import annotations

from ragamer.web.pages import create_router

__all__ = ["create_router"]
