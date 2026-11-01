# `scripts/download_model.py` 详解

## 这个文件干什么

把 HuggingFace 上的模型权重下到本地 `models/` 目录。

它解决的**不是**「怎么下载」，而是「**在 huggingface.co 不通的机器上怎么下载**」。

---

## 1. 为什么要写这个脚本

国内直连 `huggingface.co` 经常超时。三条路：

| 方案 | 做法 | 本项目 |
|---|---|---|
| 镜像 | `HF_ENDPOINT=https://hf-mirror.com` | 支持（环境变量） |
| 代理 | `HTTPS_PROXY=http://127.0.0.1:10081` | 支持（环境变量） |
| 手动 curl 单个文件 | 只下 `model.safetensors` | 不推荐（容易漏 tokenizer） |

`huggingface_hub` 会自动读这两个环境变量，所以脚本本身**不需要写任何网络代码**——它的价值是**打印当前用的是哪条路**，并做完整性统计。

实测（这台机器上）：

```
hf-mirror.com 直连     : ~0.07 MB/s   → 942 MB 要 3 小时
本地代理 huggingface.co: ~4.4 MB/s    → 942 MB 约 3.5 分钟
```

差了 60 倍。**先测速再决定走哪条路**，比盲目重试省时间。

---

## 2. `snapshot_download` 的参数

```python
path = snapshot_download(
    repo_id=args.model,
    local_dir=str(target),
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    ignore_patterns=["*.gguf", "original/*"],
)
```

* **`repo_id`**：`"Qwen/Qwen2.5-0.5B-Instruct"` 这种 `org/name` 形式。
* **`local_dir`**：存到哪。注意它**不是** HF 的缓存目录（`~/.cache/huggingface`），而是普通目录，里面直接是可用的文件。这样可以直接传路径给 `from_pretrained("models/...")`。
* **`allow_patterns`**：**白名单**。只要权重（`*.safetensors`）、配置（`*.json`）、tokenizer 词表（`*.txt`）。
  * 不写的话会把 `README.md`、图、`.gitattributes`、甚至 GGUF 版本全下下来。
* **`ignore_patterns`**：**黑名单**，在 allow 之后生效。
  * `*.gguf` 是给 llama.cpp 用的量化版本，可能比 safetensors 还大，用不上。
  * `original/*` 是老式 bin 格式，和新格式重复。

> allow + ignore 同时用是常见组合：allow 先圈定范围，ignore 再剔除例外。实际上这里只用 allow 就够了，ignore 属于"保险"。

---

## 3. 打印「当前用什么路」

```python
print(f"endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
print(f"proxy   : {os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') or '(none)'}")
```

这两行是整个脚本最有用的部分。因为**下载慢的时候，90% 的问题都是"环境变量没设上"**：

* PowerShell 设的 `$env:HTTPS_PROXY` 只在当前会话有效；新开一个终端就没了。
* 有的代理软件设的是系统代理（注册表），但 Python 的 `requests` 读的是环境变量——两者不一定同步。

打印出来一眼就能确认。

> 注意本项目实测遇到过的具体情况：注册表里有 `ProxyServer = 127.0.0.1:10081`（系统代理开着），但环境变量是空的。`urllib.request.getproxies()` 会读注册表所以显示有代理，而 `requests` 在某些情况下走的是另一套逻辑——这就是为什么 hf-mirror 报 `ProxyError`，而加上 `HTTPS_PROXY` 之后走 huggingface.co 反而正常。

---

## 4. 完整性统计

```python
total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
print(f"downloaded {args.model} -> {path} ({total / 2**30:.2f} GiB)")
```

* `rglob("*")` 递归遍历所有文件。
* `if f.is_file()` 排除目录（`stat().st_size` 对目录返回的是元数据大小，会算错）。
* `/ 2**30` 转 GiB（`2**30 = 1073741824`）。注意和 `2**10`（KiB）、`2**20`（MiB）的层级。

这个数字能**快速发现下载不完整**：Qwen2.5-0.5B 的 safetensors 应该是 942 MB，如果只有 200 MB 就是断了。

---

## 5. `if __name__ == "__main__": main()`

标准写法。作用：这个文件**既能直接运行**（`python scripts/download_model.py`），**也能被 import**（`import download_model; download_model.main`）而不自动执行。

如果少了这个守卫，任何 `import` 都会触发下载——这是 Python 项目最常见的意外副作用。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `argparse.ArgumentParser(description=__doc__)` | 用模块 docstring 当帮助文本 |
| `os.environ.get(KEY, default)` | 读环境变量（不存在给默认值） |
| `Path.mkdir(parents=True, exist_ok=True)` | 建目录，父目录一起建，已存在不报错 |
| `Path.rglob("*")` | 递归遍历 |
| `Path.stat().st_size` | 文件字节数 |
| `sum(generator)` | 生成器求和，不建中间 list |
| `2**30` | GiB 换算 |

---

## 设计取舍

**为什么不做断点续传逻辑？**

`huggingface_hub` 内部已经做了：它用 `.incomplete` 后缀 + `.metadata` 记录已下载的范围，重跑就是续传。所以脚本不需要自己实现——**用库的能力，而不是重造**。

（如果真的卡在某个文件，最直接的办法是 `curl -C - -L -o 目标文件 URL` 手动续传单个大文件。本项目实测就是这么做的：`hf-mirror` 虽通但只有 0.07 MB/s，于是用 `curl -x http://127.0.0.1:10081` 直连 `huggingface.co` 走了 4.4 MB/s。所以下载完要做的事就是核对 `model.safetensors` 的大小。）

**为什么默认 `local_dir` 是相对路径？**

相对路径基于**当前工作目录**。约定从仓库根目录运行，所以落在 `models/`。这样 `EngineConfig.model` 用同一个相对路径就能找到。缺点是在别的目录跑会落到别处——所以 README 里所有命令都假设你在仓库根。

---

## 常见误解

* 「`local_dir` 会软链接到全局缓存」——默认不会（会真的复制/硬链接）。想要省空间可以设 `HF_HUB_ENABLE_HF_TRANSFER` 或改用缓存路径。
* 「下载慢就是网速慢」——多数情况是选了慢的端点。
* 「下完 `model.safetensors` 就够了」——还必须有 `config.json`、`tokenizer.json`、`tokenizer_config.json`，否则 `from_pretrained` 会失败。

---

## 动手练习

1. 先 `Remove-Item Env:HTTPS_PROXY` 跑一次，再设上跑一次，对比速度（观察日志里的 `proxy:` 行）。
2. 把 `allow_patterns` 改成 `["*.safetensors"]` 再跑，然后尝试加载模型，看缺少配置文件时 `from_pretrained` 报什么错。
3. 用 `curl -C - -L -o model.safetensors <url>` 手动续传，验证 `-C -` 的断点续传效果。
