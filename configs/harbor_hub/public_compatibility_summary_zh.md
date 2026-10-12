# Harbor Hub 兼容性汇总

HarborRL 直接消费 Harbor 任务格式。公开 Hub dataset/task 只要符合 Harbor 任务契约，并能通过 HarborRL 的内容寻址任务检查，就可以接入；HarborRL 不要求 dataset 私有任务格式。

本页面只公开聚合统计，不发布完整 task 清单，因为 Hub 成员、license、revision 和上游可用性会随时间变化。

## 验证汇总

| 口径 | 数量 |
| --- | ---: |
| 扫描过的 Hub dataset | 340 |
| 至少存在一个静态兼容 task 的 dataset | 181 |
| 静态兼容 task 版本 | 524,041 |
| 至少存在一个动态验证 task 的 dataset | 50 |
| 动态验证通过 task 版本 | 210 |

计数按 task 内容版本统计，而不是按可变的 Hub revision 标签统计。静态验证检查 native Harbor 任务契约、镜像/环境声明、instruction、verifier 和 reward 契约；动态验证额外完成 task 构建或物化，并验证 reward 输出。动态通过数量不代表该 dataset 的所有 task 都完成过 RL 训练。新物化的 catalog 在训练前应先运行 `harborrl doctor` 校验。
