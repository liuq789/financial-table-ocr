"""网页服务：上传图片/PDF → json + excel。

主干代码，跟 pipeline 同级放在 `src/` 下，入口：

    PYTHONPATH=src python -m web.server --port 8010

三个模块：`server`（HTTP + 任务编排）/ `upload`（流式 multipart）/ `gate`（准入控制）。
`from gate import ...` 这种写法能用，是因为 server.py 把 `WEB_DIR` 也塞进了 sys.path，
所以 `-m web.server`、`import web.server`、`python src/web/server.py` 三种跑法都成立。
"""
