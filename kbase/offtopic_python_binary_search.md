---
doc_id: offtopic_python_binary_search
title: Python 二分查找实现与复杂度
category: offtopic
source_type: self
version: n/a
source_url: internal
license: project-internal
retrieved_at: 2026-10-09
---

# Python 二分查找实现与复杂度

本文件与英雄联盟数据无关，仅作为检索与拒答能力的干扰样本保留。

## 实现

二分查找要求**序列已排序**。核心思路是每次比较中间元素，
根据比较结果把搜索区间缩小一半。

```python
def binary_search(items: list[int], target: int) -> int:
    low, high = 0, len(items) - 1
    while low <= high:
        mid = (low + high) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
```

## 时间复杂度

每次迭代把问题规模减半，因此需要 log₂n 次比较，
时间复杂度为 **O(log n)**，空间复杂度为 O(1)（迭代写法）。

## 常见陷阱

- 中间位置写成 `(low + high) // 2` 在极端大整数下可能溢出，
  更稳妥的写法是 `low + (high - low) // 2`。
- 循环条件必须是 `low <= high`，写成 `<` 会漏掉单元素区间。
- 边界更新漏掉 `+1` 或 `-1` 会导致死循环。
