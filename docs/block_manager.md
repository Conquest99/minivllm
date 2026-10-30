# `minivllm/block_manager.py` 详解 ★核心

## 这个文件干什么

**物理页池的分配器**。它管三件事，全部共用同一个池子：

1. **按需分配页**（不预留）
2. **引用计数 + 写时复制**（多请求共享同几页）
3. **前缀缓存**（相同前缀复用已算好的 KV）

它是纯 Python 数据结构，**不碰任何张量**。这个分层很关键：所有「谁该拥有哪几页」的决策都在 CPU 上做，做完之后才把「复制这几页」翻译成一次显存 memcpy。

---

## 1. 数据结构总览

```python
self.blocks = [PhysicalBlock(i) for i in range(num_blocks)]   # 所有物理页
self.free_blocks: Deque[PhysicalBlock] = deque(self.blocks)   # 空闲队列
self._in_free = bytearray(b"\x01") * num_blocks               # 每页是否在空闲状态
self._num_free = num_blocks                                   # 空闲页计数
self.hash_to_block: dict[bytes, int] = {}                     # 内容哈希 -> 物理块 id
```

四个结构之间的关系：

```
blocks[7]  ←──────────────┐
  block_id=7              │
  ref_count=2             │  物理页本体（内容不在 Python 里，在显存张量里）
  block_hash=b'\x9a..'    │
  tokens=(1,2,3,...16)    │
                          │
free_blocks ── deque([0, 3, 5, ...])   谁现在能被分配
_in_free    ── bytearray                同上，但它是"真相"，deque 可能有陈旧项
hash_to_block ── {hash: 7, ...}        前缀缓存的索引
```

### `PhysicalBlock` 为什么用 `__slots__`

```python
class PhysicalBlock:
    __slots__ = ("block_id", "ref_count", "block_hash", "tokens")
```

正常情况下每个 Python 对象都带一个 `__dict__`（约 100+ 字节）。`__slots__` 去掉它，只保留声明的 4 个属性。

池子可能有几万页（1024 页起步，大模型上百万），内存和创建速度都有意义。另外还有个小好处：**拼错属性名会立即 `AttributeError`**，而不是悄悄创建一个新属性。

---

## 2. 空闲链表为什么需要 `_in_free`

这是本文件第一个精妙点。

```python
self.free_blocks: Deque[PhysicalBlock] = deque(self.blocks)
self._in_free = bytearray(b"\x01") * num_blocks
```

看起来 `free_blocks` 就够了。但当**前缀缓存命中**时：

```
某个块 ref_count 已经降到 0 → 它被 append 进 free_blocks，_in_free=1
另一条请求命中前缀 → _share(block_id) 直接复用它 → _in_free=0
但它还在 free_blocks 队列里！
```

如果只靠队列，下次 `popleft()` 会拿到一个正在被使用的块，把别人的 KV 覆盖掉——这是灾难性的误写。

所以 `_acquire_free_block` 里做了二次校验：

```python
def _acquire_free_block(self) -> PhysicalBlock:
    while self.free_blocks:
        blk = self.free_blocks.popleft()
        if not self._in_free[blk.block_id]:
            continue  # 陈旧项：这个块已经被 _share 拿走用了
        self._in_free[blk.block_id] = 0
        self._num_free -= 1
        self._evict(blk)
        blk.tokens = ()
        blk.ref_count = 1
        ...
        return blk
    raise OutOfBlocksError(f"out of KV blocks ({self.num_blocks} total)")
```

**这就是所谓的「惰性删除」**：不真的从队列里移除（O(n)），而是标记状态，取出时校验并跳过（摊还 O(1)）。和数据库里的 tombstone、CUDA 里的 free-list 一个思路。

> 为什么用 `bytearray` 而不是 `list[bool]`？因为 `list[bool]` 里每个元素是一个指向单例 `True/False` 的指针（8 字节），`bytearray` 是每元素 1 字节。几万页时差别不大，但这个习惯在写调度器时很有用。

### `OutOfBlocksError` 的意义

```python
raise OutOfBlocksError(f"out of KV blocks ({self.num_blocks} total)")
```

它**必须**是一个能被上层区分的异常类型，因为调度器要靠它触发抢占：

```python
# scheduler.py
try:
    cow_ops = self.block_manager.ensure_capacity(seq, ...)
except OutOfBlocksError:
    ... 抢占一条序列 ...
```

如果用 `RuntimeError` 或返回 `None`，调度器就得靠猜。**用异常表达「资源耗尽」是这里的正解**。

---

## 3. 引用计数

```python
def _share(self, block_id: int) -> None:
    blk = self.blocks[block_id]
    if blk.ref_count == 0:
        if not self._in_free[block_id]:
            raise AssertionError(f"block {block_id} is neither referenced nor free")
        self._in_free[block_id] = 0
        self._num_free -= 1
    blk.ref_count += 1
    self.peak_used_blocks = max(self.peak_used_blocks, self.num_used_blocks)
```

行为分两种：

| 情况 | 含义 | 做什么 |
|---|---|---|
| `ref_count > 0` | 已经在用（别人正持有） | 只加计数 |
| `ref_count == 0` | 前缀缓存里躺着、没人用 | 从空闲状态「复活」，加计数 |

`ref_count == 0` 这条分支就是**前缀缓存命中的实现**：块的内容还在显存里，但没有任何序列引用它。这时把它「借」给新序列，零拷贝。

`raise AssertionError` 那句是防御性检查：如果一个块既不是空闲状态又没被引用，说明计数逻辑已经坏了，早发现比晚发现好。

### `free()`：计数减一，减到 0 就回池子

```python
def free(self, seq: Sequence) -> None:
    for block_id in seq.block_table:
        blk = self.blocks[block_id]
        blk.ref_count -= 1
        if blk.ref_count < 0:
            raise AssertionError(f"double free of block {block_id}")
        if blk.ref_count == 0:
            self._in_free[block_id] = 1
            self.free_blocks.append(blk)
            self._num_free += 1
    seq.block_table.clear()
    seq.block_hashes.clear()
```

注意两点：

1. **引用计数到 0 不代表内容被清空**。它只是「可以再被分配」。如果它已经登记在 `hash_to_block` 里，内容就是一份有效的缓存，下次同前缀请求还能命中。这个「懒释放」是前缀缓存能工作的前提。
2. `seq.block_table.clear()` 用的是 `clear()` 而不是 `= []`。因为 `Scheduler`、测试、`fork` 都可能持有同一个 list 的引用；`clear()` 是原地修改，所有引用都看到变化。

> `blk.ref_count < 0` 的检查抓的是「同一个序列被 free 两次」。本项目的 `free_finished` 用 `running` 列表过滤，理论上不会发生；但这类断言的价值在于**一旦发生能在正确的位置炸**，而不是几步之后表现为「KV 内容莫名其妙错了」。

---

## 4. 写时复制 COW

### 场景

两条请求有相同前缀（比如同一个 system prompt），它们**共享**前缀那几页。现在其中一条要往后写新 token，正好落在最后一页的**空槽位**里。

如果直接写，另一条请求会看到被污染的数据。所以在写之前必须**先把这一页复制一份**。

### `ensure_capacity`

```python
def ensure_capacity(self, seq: Sequence, num_tokens: int) -> list[tuple[int, int]]:
    bs = self.block_size
    needed = (num_tokens + bs - 1) // bs
    cow_ops: list[tuple[int, int]] = []

    idx = seq.num_computed_tokens // bs
    if num_tokens > seq.num_computed_tokens and idx < len(seq.block_table):
        src = seq.block_table[idx]
        if self.blocks[src].ref_count > 1:
            dst = self._acquire_free_block()
            self.blocks[src].ref_count -= 1
            seq.block_table[idx] = dst.block_id
            seq.block_hashes[idx] = None
            cow_ops.append((src, dst.block_id))
            self.num_cow_ops += 1

    while len(seq.block_table) < needed:
        blk = self._acquire_free_block()
        seq.block_table.append(blk.block_id)
        seq.block_hashes.append(None)
    return cow_ops
```

逐行看：

**`idx = seq.num_computed_tokens // bs`**

这是「下一个要写的 token 落在逻辑第几块」。比如 `num_computed_tokens = 20`，`bs = 16` → `idx = 1`（第 2 块）。

**为什么判断 `num_tokens > num_computed_tokens`**

只有「真的要写新 token」时才需要考虑 COW。如果只是查询式地调 `ensure_capacity(seq, 20)`（不增长），不该白白复制一页。

**`idx < len(seq.block_table)`**

如果下一块还没分配（比如 `num_computed_tokens = 16`、`block_table` 只有 1 项，`idx = 1`），那就没有共享页的问题——新块本来就是独占的。

**`ref_count > 1`**

只有被 **2 个以上** 持有才需要复制。独占（=1）时直接写没问题。

**为什么不整页复制而是只复制这一页**

因为只有这一页可能被写坏。更前面的页早已写满，不会再被改（token 是只追加的）。这就是为什么 COW 在 KV cache 上特别划算：**每条分叉请求最多只需要复制 1 页**。

**返回值为什么是 `[(src, dst), ...]` 而不是直接在管理器里做复制**

因为管理器不知道显存长什么样。它只产出「结构决策」，由 `ModelRunner.apply_cow` 翻译成真正的 `copy_blocks`。**数据和元数据分离**，这也是可以纯 CPU 测试的原因。

### `fork`：一次复制都没有的「分叉」

```python
def fork(self, parent: Sequence, child: Sequence) -> None:
    child.block_table = list(parent.block_table)
    child.block_hashes = list(parent.block_hashes)
    child.num_computed_tokens = parent.num_computed_tokens
    for block_id in child.block_table:
        self._share(block_id)
```

对应 `n > 1` 并行采样（同一个 prompt 出多个答案）。此时子序列**完全共享**父序列的所有页，`ref_count` 各加一。

真正开始分叉发生在两者各自写下第一个 token 时：`ensure_capacity` 会看到 `ref_count == 2`，把分叉那一页复制一份。之后 `ref_count` 降回 1，再往后写就都是独占，不再复制。

**所以 N 个并行采样、生成长度 L 的总复制量 ≈ N 页，而不是 N×L 页。** 这是这段代码最值得讲的一句话。

---

## 5. 前缀缓存

### 5.1 哈希为什么是「链式」的

```python
def _block_hash(self, prev_hash: bytes, tokens: tuple[int, ...]) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(prev_hash)
    h.update(array("q", tokens).tobytes())
    return h.digest()
```

`hash_i = H(hash_{i-1}, tokens_i)`。

**为什么要带上上一块的哈希？** 因为只哈希自己那块内容的话，`[A,B,C]` 的第 1 块和 `[X,B,C]` 的第 1 块会得出同一个哈希——但它们的**前缀不同**，KV 不可互换（因为 K/V 里融入了位置编码和因果依赖）。

链式哈希让 `hash_i` 唯一编码「从第 0 块到第 i 块的完整内容」。**前缀相同 ⟺ 链式哈希相同**。

> 这就是 Merkle 链的思想（区块链、Git 的 commit hash 都是这个套路）。

`array("q", tokens).tobytes()` 把 Python `list[int]` 转成 8 字节有符号整数的紧凑字节串。为什么不用 `str(tokens)`？因为 `str` 更慢、更长，且数字转字符串的边界容易有歧义（`[1, 23]` vs `[12, 3]`）。

`digest_size=16` → 128 位。生日碰撞概率在 2^64 量级，实际使用完全够。vLLM 用更长的哈希，因为它的缓存跨进程/跨重启持久化。

### 5.2 `match_prefix`：从第 0 块开始逐块比，遇错即停

```python
def match_prefix(self, token_ids: list[int]) -> list[int]:
    if not self.enable_prefix_caching:
        return []
    bs = self.block_size
    matched: list[int] = []
    prev = _PREFIX_SALT
    for i in range(len(token_ids) // bs):
        chunk = tuple(token_ids[i * bs : (i + 1) * bs])
        prev = self._block_hash(prev, chunk)
        block_id = self.hash_to_block.get(prev)
        if block_id is None:
            break
        if self.blocks[block_id].tokens != chunk:
            break  # 哈希碰撞防御
        matched.append(block_id)
    return matched
```

三个关键点：

1. **`len(token_ids) // bs`**：只用**完整**的块。最后不满 16 个 token 的尾巴不参与——因为它的内容还会变，没有稳定的哈希。
2. **`break` 而不是 `continue`**：前缀一旦断开，后面就没意义了（第 5 块的 KV 依赖第 0-4 块）。这是 O(命中长度) 而不是 O(全长度) 的剪枝。
3. **`tokens != chunk` 的二次校验**：哈希碰撞时（或哈希被 `setdefault` 覆盖的极端情况）靠实际内容兜底。128 位哈希碰撞几乎不可能，但这个检查几乎零成本，属于「便宜的保险」。

`_PREFIX_SALT` 是个常量前缀（`b"minivllm/prefix/v1"`），作用是**域分隔**：避免和别处用同一份 blake2b 算出的哈希互相干扰，也便于将来格式升级时让旧缓存自动失效。

### 5.3 `register_computed_blocks`：把写满的页登记进缓存

```python
def register_computed_blocks(self, seq: Sequence) -> None:
    if not self.enable_prefix_caching:
        return
    bs = self.block_size
    prev = _PREFIX_SALT
    for i in range(seq.num_computed_tokens // bs):
        blk = self.blocks[seq.block_table[i]]
        if blk.block_hash is not None:
            prev = blk.block_hash
            continue
        chunk = tuple(seq.data[i * bs : (i + 1) * bs])
        if blk.tokens and blk.tokens != chunk:
            return
        prev = self._block_hash(prev, chunk)
        blk.block_hash = prev
        blk.tokens = chunk
        self.hash_to_block.setdefault(prev, blk.block_id)
        seq.block_hashes[i] = prev
```

* 只登记 `num_computed_tokens // bs` 个块，即**已经算完的完整块**。
* `if blk.block_hash is not None: prev = blk.block_hash; continue` —— 已经登记过的块跳过，但**必须把它的哈希接上链**，否则后面的哈希就断了。（这是最容易漏的一行。）
* `if blk.tokens and blk.tokens != chunk: return` —— 这一页装的是别人的内容（COW 之前共享过的残留），绝不能登记成「本序列的内容」。直接 `return` 而不是 `break`，因为后面所有块的哈希链都依赖这一块，全都没意义了。
* `setdefault` 而不是 `[prev] = ...`：**先到先得**。如果两个不同物理块内容碰巧相同（例如两条请求恰好有一样的前缀），保留第一个，第二个只是普通块。这样 `hash_to_block` 永远指向一个稳定的块。

**调用时机很关键**：在 `free_finished()` 里，**先** `register_computed_blocks` 再 `free`：

```python
# scheduler.py
self.block_manager.register_computed_blocks(seq)
self.block_manager.free(seq)
```

因为 `free()` 会 `seq.block_table.clear()`，之后再登记就没页表可读了。这个顺序反了会导致前缀缓存**完全失效**（而且不报错，只是慢）——我第一版就写反过一次。

### 5.4 驱逐：内容被覆盖时把索引删掉

```python
def _evict(self, blk: PhysicalBlock) -> None:
    if blk.block_hash is not None:
        if self.hash_to_block.get(blk.block_hash) == blk.block_id:
            del self.hash_to_block[blk.block_hash]
        blk.block_hash = None
```

在 `_acquire_free_block` 里调用。`get(...) == blk.block_id` 的判断很重要：如果这个哈希已经被另一个块"抢走"（`setdefault` 的场景），那删掉索引会误伤别人，所以只删指向自己的那条。

**策略是纯 FIFO**（`deque.popleft()`），没有 LRU。真实引擎会用 LRU 或多级队列。本项目选了最简单的，因为目标是讲清「机制」而不是「缓存策略调优」。

---

## 6. 分配的两个入口

### `allocate`：第一次分配（可能全部命中缓存）

```python
def allocate(self, seq: Sequence, num_tokens: int | None = None) -> int:
    if seq.block_table:
        raise AssertionError("sequence already owns a block table")
    bs = self.block_size
    num_tokens = len(seq.data) if num_tokens is None else num_tokens
    needed = (num_tokens + bs - 1) // bs

    matched = self.match_prefix(seq.data[:num_tokens])[:needed]
    for block_id in matched:
        self._share(block_id)
        seq.block_table.append(block_id)
        seq.block_hashes.append(self.blocks[block_id].block_hash)
    self.num_prefix_hit_blocks += len(matched)

    for _ in range(needed - len(matched)):
        blk = self._acquire_free_block()
        seq.block_table.append(blk.block_id)
        seq.block_hashes.append(None)
    return len(matched)
```

* **`num_tokens` 参数存在的原因**：chunked prefill 时只分配**第一块**的页，而不是整个 prompt 的页。这样超长 prompt 也不会瞬间吃掉大量显存。
* **`[:needed]` 的截断**：`match_prefix` 返回的命中块可能超过本次实际需要的块数（因为 `num_tokens` 只是第一个 chunk）。
* **返回值是命中块数**，用于统计和测试断言。

### `can_allocate`：只判断不动作

```python
def can_allocate(self, seq: Sequence, num_tokens: int | None = None) -> AllocStatus:
    ...
    needed = (num_tokens + bs - 1) // bs
    if self.enable_prefix_caching:
        needed -= len(self.match_prefix(seq.data[:num_tokens]))
    if needed <= self._num_free:
        return AllocStatus.OK
    return AllocStatus.LATER
```

**先问后做**（check-then-act）。这在单线程调度器里是安全的；多线程环境里就得合并成「尝试分配，失败返回 None」。

`AllocStatus` 有三个值（`OK` / `LATER` / `NEVER`），本项目只用前两个。`NEVER` 是为「请求长度超过 `max_model_len`，永远装不下」预留的——目前那个判断在 `Scheduler` 里直接做掉了。

---

## 7. 统计与对照物

```python
def naive_reservation_cost(self, seq: Sequence, max_model_len: int) -> int:
    """非分页引擎会为这条序列预留多少块。"""
    return (max_model_len + self.block_size - 1) // self.block_size
```

这行代码本身没人调用（只在测试里用），但它是整个项目的**对照组**：有了它，「分页省了多少」就不再是论文里的一句话，而是可以算出来、跑出来的数字。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `__slots__` | 省内存、防拼错 |
| `deque` | 两端 O(1) 插入/弹出 |
| `bytearray` | 紧凑布尔数组 |
| `hashlib.blake2b(digest_size=16)` | 快速短哈希 |
| `array("q", ...).tobytes()` | list[int] → 紧凑字节串 |
| `dict.setdefault(k, v)` | 不存在才插入（先到先得） |
| `Enum` | 分配结果三态 |
| 自定义异常 | 让上层能精确 catch |
| `raise AssertionError` | 内部不变量检查 |

---

## 设计取舍

**为什么不用 `torch.Tensor` 存页表？**

页表是**每个 step 都要改**的小数据结构（追加、复制、清空）。用它做 Python 操作比反复构造张量快得多，而且改完只要在 `_build_inputs` 时一次性转成张量。真实引擎（vLLM）也是这么分的：Python 侧管元数据，kernel 侧只读最终形状。

**为什么没有真正的「内存回收」？**

GPU 显存池是一次性 `torch.zeros` 出来的，块的「释放」只是把索引还回 free list。**从不 `del` 张量、从不重新分配**——这正是消除碎片的关键。如果每来一条请求都新建张量，碎片问题会以另一种形式回来。

**为什么前缀缓存用内容哈希而不是 `(seq_id, 前缀长度)`？**

因为不同请求之间要能互相命中。`seq_id` 是请求私有标识，哈希内容才能跨请求匹配。

---

## 常见误解

* 「`ref_count == 0` 意味着这块内存空了」——不，内容可能还是有效缓存。空不空是 `tokens`/`block_hash` 说了算。
* 「分页之后完全没有碎片」——有**内部碎片**（最后一页的尾巴），实测 4.7%。消失的是**外部碎片**。
* 「COW 会在 fork 时复制」——不会。fork 时零复制，只在**第一次写入共享页**时复制，而且只复制那一页。
* 「`free_blocks` 队列长度就是空闲页数」——不是，队列里可能有被 `_share` 复活的陈旧项。真相是 `_num_free`。
* 「前缀缓存命中就能跳过 prefill」——光共享页不够，调度器还要把 `num_computed_tokens` 推到缓存长度之后，见 [scheduler.md](scheduler.md)。

---

## 我踩过的两个坑

1. **`register_computed_blocks` 和 `free` 的顺序写反**。先 `free` 再登记，`block_table` 已经被清空，前缀缓存永远登记不进去。表现为「功能正常但一点加速都没有」。
2. **只共享页、不推进 `num_computed_tokens`**。前缀缓存明明命中了 200 个块，调度器还是把整个 prompt 重算一遍。修完之后 warm TTFT 才从 0.779s 降到 0.077s（10.2 倍）。

---

## 动手练习

1. 在 `_acquire_free_block` 里删掉 `if not self._in_free[blk.block_id]: continue`，跑 `tests/test_block_manager.py::test_prefix_cache_shares_blocks`，看它怎么挂（这个测试就是为这个不变量写的）。
2. 把 `_block_hash` 里的 `h.update(prev_hash)` 删掉，跑 `test_prefix_cache_shares_blocks` 和 `test_engine_e2e.py::test_prefix_caching_gives_the_same_tokens`，观察「不同前缀被误判为相同」。
3. 把 `ensure_capacity` 里 `ref_count > 1` 改成 `ref_count >= 1`，跑 `test_copy_on_write_after_fork`，看会发生多少次多余复制（`num_cow_ops`）。
4. 把 `ensure_capacity` 里 COW 整段删掉（直接共享页往下写），跑 `tests/test_engine_e2e.py`，验证 `test_prefix_caching_gives_the_same_tokens` 会失败——这就是「没有 COW 会静默产生错误结果」的证据。
