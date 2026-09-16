import os
import re
import pandas as pd

class AbbrReplacer:
    def __init__(self, excel_path):
        self.abbr_mapping = {}
        self.pattern = None
        self.combined_dict = {}
        
        if os.path.exists(excel_path):
            try:
                df = pd.read_excel(excel_path, sheet_name="简称")
                if '简称' in df.columns and '全称' in df.columns:
                    for _, row in df.iterrows():
                        self.abbr_mapping[str(row['简称']).strip()] = str(row['全称']).strip()
                else:
                    for _, row in df.iterrows():
                        self.abbr_mapping[str(row.iloc[0]).strip()] = str(row.iloc[1]).strip()
                
                print(f"✅ 简称词典加载完成，共 {len(self.abbr_mapping)} 条记录。")
                self._build_pattern() # 构建正则模式
                
            except Exception as e:
                print(f"⚠️ 加载简称字典失败: {e}")
        else:
            print(f"⚠️ 简称字典文件不存在: {excel_path}")

    def _build_pattern(self):
        if not self.abbr_mapping:
            return
            
        for abbr, full in self.abbr_mapping.items():
            if abbr and abbr != 'nan':
                self.combined_dict[full] = full  # 保护全称不变
                self.combined_dict[abbr] = full  # 替换简称
                
        sorted_keys = sorted(self.combined_dict.keys(), key=len, reverse=True)
        
        escaped_keys = [re.escape(k) for k in sorted_keys]
        self.pattern = re.compile("|".join(escaped_keys))

    def process(self, text):
        if not self.pattern or not text:
            return text
            
        return self.pattern.sub(lambda m: self.combined_dict[m.group(0)], text)

