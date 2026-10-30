# `tests/test_block_manager.py` 详解

## 这个文件干什么

给分配器写 **9 条不变量（invariant）**。不碰 torch、不碰模型，毫秒级跑完。

它守的是三件最容易写错、且错了会静默出错的事：**引用计数、COW、前缀缓存**。

---

## 0. 两个构造辅助

```python
def make_seq(seq_id: int, token_ids: list[int]) -> Sequence:
    return Sequence(seq_id=seq_id, prompt_token_ids=token_ids, sampling_params=SamplingParams())

def tokens(seq_id: int, n: int) -> list[int]:
    return [seq_id * 1000 + i for i in range(n)]
```

`tokens(seq_id, n)` 生成「属于 seq_id 的 n 个唯一 token」。`seq_id * 1000` 错开区间，保证**不同序列的 token 不重叠**——否则前缀缓存会误命中，测试就没意义了。

---

## 1. `test_allocate_and_free_returns_blocks`

```python
bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_caching=False)
seq = make_seq(1, tokens(1, 10))
assert bm.can_allocate(seq) is AllocStatus.OK

bm.allocate(seq, 10)
assert len(seq.block_table) == 3
assert bm.num_free_blocks == 5

bm.free(seq)
assert bm.num_free_blocks == 8
assert seq.block_table == []
```

守的不变量：**分配 10 个 token（4/块）要 3 块；释放后计数完全还原。**

`is AllocStatus.OK` 用 `is` 比较枚举（枚举是单例，`is` 比 `==` 更严格也更快）。

---

## 2. `test_allocation_is_lazy_not_reserved`

```python
bm = BlockManager(num_blocks=64, block_size=4, enable_prefix_caching=False)
seq = make_seq(1, tokens(1, 10))
bm.allocate(seq, 10)
assert len(seq.block_table) == 3
assert bm.naive_reservation_cost(seq, max_model_len=256) == 64
assert bm.num_free_blocks == 61
```

**这条测试就是整个项目的核心命题**：

* 分页：10 个 token 只占 **3** 块
* 预留式：要占 `ceil(256/4) = 64` 块

差了 21 倍。这把「为什么需要 PagedAttention」从一句话变成了一个断言。

---

## 3. `test_out_of_blocks`

```python
bm = BlockManager(num_blocks=2, block_size=4, enable_prefix_caching=False)
seq = make_seq(1, tokens(1, 12))
assert bm.can_allocate(seq) is AllocStatus.LATER
with pytest.raises(OutOfBlocksError):
    bm.allocate(seq, 12)
```

守两件事：

1. `can_allocate` **先问后做**，问的时候返回 `LATER` 而不是抛异常。
2. `allocate` 真的装不下时抛 `OutOfBlocksError`——这个类型必须精确，因为调度器靠 `except OutOfBlocksError` 触发抢占。

> `pytest.raises` 是上下文管理器：块内的代码必须抛出指定异常，否则测试失败。

---

## 4. `test_growth_allocates_only_when_crossing_a_boundary`

```python
bm = BlockManager(num_blocks=8, block_size=8, enable_prefix_caching=False)
seq = make_seq(1, tokens(1, 4))
bm.allocate(seq, 4)
seq.num_computed_tokens = 4

for n in range(5, 9):        # 5..8 都还在第 0 页里
    assert bm.ensure_capacity(seq, n) == []
    assert len(seq.block_table) == 1

assert bm.ensure_capacity(seq, 9) == []    # 第 9 个 token 才要新页
assert len(seq.block_table) == 2
```

守的不变量：**只在跨页边界时分配新块**。

这里刻意用 `block_size=8`（而不是 4），让「4 个 token 占 1 页」这个前提成立，然后验证 5/6/7/8 都不触发新分配。

> 第一版这里写错了：我用了 `block_size=4`、初始 4 token，然后循环测 5/6/7，期待「不新分配」——但 5 已经跨页了。**测试写错比代码写错更常见**，改对之后断言才真正对应「跨边界才分配」这个语义。

`ensure_capacity` 返回 `[]` 表示「没有 COW 操作」。空列表是对的——独占页不需要复制。

---

## 5. `test_prefix_cache_shares_blocks`

```python
bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
first = make_seq(1, tokens(1, 8))     # 2 个满块
bm.allocate(first, 8)
first.num_computed_tokens = 8
bm.register_computed_blocks(first)

second = make_seq(2, first.prompt_token_ids[:6])   # 共享第一个满块
assert bm.match_prefix(second.data) == first.block_table[:1]

matched = bm.allocate(second, 6)
assert matched == 1
assert second.block_table[0] == first.block_table[0]
assert bm.blocks[first.block_table[0]].ref_count == 2

bm.free(first)
assert bm.blocks[second.block_table[0]].ref_count == 1
assert bm.num_free_blocks == 16 - 2     # 还活着，不能被回收
```

流程走了一遍完整生命周期：

1. 序列 1 算完 8 个 token → 登记 2 个满块进缓存
2. 序列 2 有相同的前 6 个 token → 只能命中 **1 个满块**（第 2 块只有 2 个 token，不满，不参与）
3. `matched == 1`：1 块来自缓存，1 块新分配
4. 两条序列共享同一物理块 → `ref_count == 2`
5. 释放序列 1 → `ref_count` 降回 1，**块仍在用**（`num_free_blocks` 只回收了序列 1 的另一个块）

最后那条断言最容易写错：`16 - 2` 而不是 `16 - 1`。因为序列 1 独占的第 2 块被回收了，共享的第 1 块没有。

---

## 6. `test_prefix_cache_disabled_ignores_matches`

```python
bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=False)
...
assert bm.match_prefix(first.data) == []
second = make_seq(2, first.prompt_token_ids[:6])
assert bm.allocate(second, 6) == 0
```

同样的场景，关掉缓存后 **命中数必须是 0**。

**为什么需要这条**：`bench_static_vs_continuous.py` 和 `bench_prefix_caching.py` 都靠「翻转 `enable_prefix_caching`」做 A/B 对比。如果关闭时仍然悄悄命中，所有对比数据都是假的。

---

## 7. `test_copy_on_write_after_fork`

```python
parent = make_seq(1, tokens(1, 6))       # 1 满块 + 1 半块
bm.allocate(parent, 6)
parent.num_computed_tokens = 6

child = make_seq(2, list(parent.prompt_token_ids))
bm.fork(parent, child)
assert child.block_table == parent.block_table
assert bm.blocks[parent.block_table[1]].ref_count == 2

cow = bm.ensure_capacity(child, 7)
assert len(cow) == 1
src, dst = cow[0]
assert src == parent.block_table[1]
assert dst == child.block_table[1] != parent.block_table[1]
assert bm.blocks[src].ref_count == 1

assert bm.ensure_capacity(parent, 7) == []
```

这是全文件最精细的一条，逐步看：

1. `fork` 之后两者**完全共享**（页表相同，`ref_count == 2`）
2. 子序列要写第 7 个 token → 落在**半满的第 2 块**（`idx = 6 // 4 = 1`）→ 必须 COW
3. `cow` 返回 1 个操作 `(src, dst)`
4. 复制后 `src` 的 `ref_count` 从 2 降到 1（只剩 parent）
5. **parent 再写时不需要再复制**——因为它现在是独占的

第 5 点是精髓：**分叉的总复制量是「每对分叉一次」，不是「每个写入者一次」**。这直接决定了「并行采样 N 条」的额外显存是 N 页而不是 N×L 页。

---

## 8. `test_eviction_drops_cache_entry`

```python
bm = BlockManager(num_blocks=2, block_size=4, enable_prefix_caching=True)
first = make_seq(1, tokens(1, 8))
bm.allocate(first, 8)
first.num_computed_tokens = 8
bm.register_computed_blocks(first)
assert len(bm.hash_to_block) == 2

bm.free(first)
assert len(bm.hash_to_block) == 2      # 空闲但仍是有效缓存
assert bm.num_free_blocks == 2

other = make_seq(2, tokens(9, 4))
bm.allocate(other, 4)
assert len(bm.hash_to_block) == 1      # 有一项被驱逐了
```

守的不变量：**释放不等于失效。**

* `free` 之后块回到空闲列表，但它的哈希仍在 `hash_to_block` 里（**内容还有效**，后续同前缀请求能命中）
* 只有当这个块被**重新分配给别人**时，`_evict` 才会把索引删掉

这是「懒释放」的核心，也是前缀缓存能跨请求工作的前提。

池子只有 2 块，所以分配 1 块必然驱逐 1 项——用极小的池子让行为和断言确定。

---

## 9. `test_no_double_free`

```python
seq = make_seq(1, tokens(1, 4))
bm.allocate(seq, 4)
bm.free(seq)
bm.free(seq)                     # 幂等
assert bm.num_free_blocks == 4
```

第二次 `free` 时 `seq.block_table` 已经空了（第一次 `clear()` 过），所以循环体不执行，什么都不会发生。

**这条测试守的是「幂等性」**：重复释放不应该让计数变成 5。

而真正危险的「两个序列共享一块、其中一个被 free 两次」由 `free` 里的 `if blk.ref_count < 0: raise AssertionError` 兜底——那条断言没有专门的测试（很难在不破坏封装的前提下构造），但保留着。

---

## 测试覆盖了什么、没覆盖什么

| 覆盖 | 方式 |
|---|---|
| 分配/释放计数守恒 | test 1 |
| 按需分配（核心命题） | test 2 |
| 容量判断与异常 | test 3 |
| 只在跨页时增长 | test 4 |
| 前缀缓存共享 + 引用计数 | test 5 |
| 关闭缓存时零命中 | test 6 |
| COW 时机与次数 | test 7 |
| 懒释放与驱逐 | test 8 |
| 幂等 | test 9 |

**没覆盖**：哈希碰撞（无法构造）、并发（本项目单线程）、`AllocStatus.NEVER`（未使用）。

---

## 用到的 pytest / Python 特性

| 特性 | 说明 |
|---|---|
| `pytest.raises(Exc)` | 断言抛异常 |
| `is` 比较枚举 | 单例身份比较 |
| `list[:1]` / `list[:a]` | 切片做子页表比较 |
| 断言里带消息 | `test 7` 里 `!=` 直接断言 |
| 极小池子（2 块）| 让驱逐行为确定 |

---

## 设计取舍

**为什么这些测试不需要 GPU？**

因为 `BlockManager` 完全不知道张量的存在。它只产出「哪块归谁」「要复制哪几页」这样的**结构决策**，真正的显存操作在 `ModelRunner.apply_cow`。这个分层让 9 条最重要的不变量可以在 0.1 秒内验证完。

**为什么用 `enable_prefix_caching=False` 做大部分测试？**

为了让「分页」和「前缀缓存」两个机制**分别**被测。如果一直开着缓存，`test_allocate_and_free` 里的块数断言会被 `match_prefix` 干扰（`tokens()` 保证不重叠，所以其实不会命中，但显式关掉更清楚）。

---

## 常见误解

* 「`num_free_blocks == 0` 就是 OOM」——不，它表示「所有块都有人引用或作为缓存躺着」。真正的 OOM 是 `_acquire_free_block` 找不到可用块。
* 「`free` 之后块内容被清零」——没有。清零只在 `_acquire_free_block` 里（`blk.tokens = ()`，且显存内容根本不归它管）。
* 「共享块在某个序列结束后会立刻回收」——只要 `ref_count > 0` 就不会。

---

## 动手练习

1. 删掉 `_acquire_free_block` 里的 `if not self._in_free[...]: continue`，跑 `test_prefix_cache_shares_blocks`，观察它怎么挂（会分配到正在被共享的块）。
2. 把 `test_growth_allocates_only_when_crossing_a_boundary` 里的 `block_size` 从 8 改回 4，确认测试失败——然后想清楚为什么原来的断言是错的。
3. 给 `test_eviction_drops_cache_entry` 加一步：分配后再次 `match_prefix(first.prompt_token_ids)`，确认返回 `[]`（缓存已失效）。
4. 加一条新测试：`fork` 之后让子序列写满整整一页，验证 `num_cow_ops` 只增加 1（不是 2）。
