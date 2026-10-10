# 英雄 ID 名称映射

> **本文件的内容已迁入知识库 `kbase/`，此处保留作为原始记录。**
> 当前生效的说明见 `kbase/project_champion_name_mapping.md`，
> 版本冲突样本见 `kbase/conflict_urgot_rename.md` 与
> `kbase/conflict_champion_labels.md`。检索层只索引 `kbase/`。

## 映射来源与字段

项目使用 Riot Data Dragon 7.17.1 的 zh_CN 英雄静态数据。映射文件 champion_mapping_7_17_1.json 保留全部 138 个数据英雄 ID 的官方字段：

- name：Data Dragon 中的中文名称，例如金克丝对应“暴走萝莉”。
- title：同一条目中的中文 title，例如金克丝对应“金克丝”、亚索对应“亚索”。
- english_id：Data Dragon 的英文英雄标识符，例如 Jinx、Yasuo。
- aliases：需要额外兼容的历史名称；目前 ID 6 保留 7.11.1 使用的“首领之傲”。

原始 CSV 的英雄字段存储数值 ID，不含英雄名称；统计工具在查询和展示时使用映射。调用时统一通过 champion_id 参数传入数值 ID、name、title 或英文标识符。统计工具对上述字段执行精确匹配，不做模糊匹配。按全量 138 个 ID 核对，name、title 和 english_id 均存在，且无跨英雄名称冲突。金克丝和亚索无需专门的手工别名：它们已经分别由 Data Dragon title 字段提供。其他英雄也采用同一规则解析。

## 版本边界

数据集时间范围为 2017-06-08 至 2017-09-06 UTC，跨越多个补丁，但 CSV 不含逐场 patch 字段。7.17.1 映射用于名称查询和展示，不能证明每一场比赛发生时的名称或精确补丁，也不能用于按补丁切片或评估当前版本。

对照 7.11.1 与 7.17.1 时，ID 6 的中文 name 从“首领之傲”变为“无畏战车”；项目用 7.17.1 的 name 作为展示名称，并将旧名称作为兼容 alias。

## 数据来源

- Riot Developer Portal：<https://developer.riotgames.com/docs/lol>
- Riot Data Dragon 7.17.1，简体中文英雄数据：<https://ddragon.leagueoflegends.com/cdn/7.17.1/data/zh_CN/champion.json>
