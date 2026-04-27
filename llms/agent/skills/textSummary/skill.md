## 技能信息
name: textSummarization
description: 对输入的长文本进行精炼摘要，保留核心信息。当用户发送长文章、会议记录或要求总结时使用。

## 参数：
  - text：原始文本（必填）

## 文本摘要 Skill
- 功能：使用 deepseek-r1:latest 模型生成文本摘要
- 返回：摘要字符串
- 示例：
  ```python
  from skills.textSummarySkill.textSummarization import TextSummarization
  ts = TextSummarization()
  print(ts.execute("这是一段需要摘要的长文本..."))

