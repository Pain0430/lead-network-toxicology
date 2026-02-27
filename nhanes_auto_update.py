#!/usr/bin/env python3
"""
NHANES数据自动更新工具
NHANES Data Auto-Update Tool

功能:
1. 定期检查NHANES新数据发布
2. 自动下载更新的数据
3. 与现有数据对比
4. 通知用户更新内容

作者: Pain's AI Assistant
日期: 2026-02-22
"""

import os
import json
import time
import hashlib
import requests
from datetime import datetime, timedelta
from urllib.parse import urljoin
import smtplib
from email.mime.text import MIMEText
import argparse

# 配置
OUTPUT_DIR = "nhanes_data"
CONFIG_FILE = "nhanes_update_config.json"
LOG_FILE = "nhanes_update.log"

# NHANES 数据URL (2021-2023)
NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/"

# 重要的数据文件
NHANES_KEY_FILES = {
    # 重金属 (最重要)
    "PBCD_L": {
        "name": "Blood Lead, Cadmium, Mercury, Selenium, Manganese",
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/PBCD_L.xpt",
        "category": "laboratory",
        "importance": "high",
    },
    "IHGEM_L": {
        "name": "Inorganic Mercury & Methylmercury", 
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/IHGEM_L.xpt",
        "category": "laboratory",
        "importance": "high",
    },
    # 生化指标
    "CBC_L": {
        "name": "Complete Blood Count",
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.xpt",
        "category": "laboratory",
        "importance": "medium",
    },
    "GHB_L": {
        "name": "Glycohemoglobin",
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.xpt",
        "category": "laboratory",
        "importance": "medium",
    },
    "HDL_L": {
        "name": "High-Density Lipoprotein",
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/HDL_L.xpt",
        "category": "laboratory",
        "importance": "medium",
    },
    "TRIGLY_L": {
        "name": "LDL & Triglycerides",
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TRIGLY_L.xpt",
        "category": "laboratory",
        "importance": "medium",
    },
    # 体检
    "BPX_L": {
        "name": "Blood Pressure",
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPX_L.xpt",
        "category": "examination",
        "importance": "high",
    },
    "BMX_L": {
        "name": "Body Measures",
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt",
        "category": "examination",
        "importance": "medium",
    },
    # 问卷
    "DEMO_L": {
        "name": "Demographics",
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt",
        "category": "questionnaire",
        "importance": "high",
    },
    "MCQ_L": {
        "name": "Medical Conditions",
        "url": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/MCQ_L.xpt",
        "category": "questionnaire",
        "importance": "medium",
    },
}


class NHANESUpdater:
    """NHANES数据更新器"""
    
    def __init__(self, output_dir=OUTPUT_DIR, config_file=CONFIG_FILE):
        self.output_dir = output_dir
        self.config_file = config_file
        self.config = self.load_config()
        os.makedirs(output_dir, exist_ok=True)
        
    def load_config(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "last_check": None,
                "last_update": None,
                "file_hashes": {},
                "download_history": [],
                "notification_email": None,
            }
            
    def save_config(self):
        """保存配置"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
            
    def check_file_hash(self, filepath):
        """计算文件hash"""
        if not os.path.exists(filepath):
            return None
            
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def check_updates(self, verbose=True):
        """检查更新"""
        updates = []
        
        if verbose:
            print("=" * 60)
            print("🔍 检查 NHANES 数据更新...")
            print("=" * 60)
            
        for file_id, info in NHANES_KEY_FILES.items():
            filepath = os.path.join(self.output_dir, f"{file_id}.xpt")
            filename = f"{file_id}.xpt"
            
            # 检查文件是否存在
            if not os.path.exists(filepath):
                updates.append({
                    "file_id": file_id,
                    "filename": filename,
                    "status": "new",
                    "info": info,
                })
                if verbose:
                    print(f"  📥 新文件: {file_id}")
                continue
                
            # 检查hash是否变化
            current_hash = self.check_file_hash(filepath)
            stored_hash = self.config["file_hashes"].get(file_id)
            
            if current_hash != stored_hash:
                updates.append({
                    "file_id": file_id,
                    "filename": filename,
                    "status": "updated",
                    "info": info,
                    "old_hash": stored_hash,
                    "new_hash": current_hash,
                })
                if verbose:
                    print(f"  🔄 更新: {file_id}")
            else:
                if verbose:
                    print(f"  ✅ 无变化: {file_id}")
                    
        self.config["last_check"] = datetime.now().isoformat()
        self.save_config()
        
        return updates
        
    def download_file(self, url, filepath, verbose=True):
        """下载文件"""
        try:
            if verbose:
                print(f"  📥 下载: {url}")
                
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
                
            size = os.path.getsize(filepath) / 1024  # KB
            if verbose:
                print(f"     ✅ 完成 ({size:.1f} KB)")
                
            return True, size
            
        except Exception as e:
            if verbose:
                print(f"     ❌ 失败: {e}")
            return False, 0
            
    def download_updates(self, updates, verbose=True):
        """下载所有更新"""
        if not updates:
            if verbose:
                print("\n✅ 没有需要更新的文件")
            return []
            
        if verbose:
            print("\n" + "=" * 60)
            print("📥 开始下载更新...")
            print("=" * 60)
            
        downloaded = []
        
        for update in updates:
            file_id = update["file_id"]
            info = update["info"]
            filename = f"{file_id}.xpt"
            filepath = os.path.join(self.output_dir, filename)
            
            success, size = self.download_file(info["url"], filepath, verbose)
            
            if success:
                # 更新hash
                new_hash = self.check_file_hash(filepath)
                self.config["file_hashes"][file_id] = new_hash
                
                downloaded.append({
                    "file_id": file_id,
                    "filename": filename,
                    "size_kb": size,
                    "status": update["status"],
                })
                
        self.config["last_update"] = datetime.now().isoformat()
        self.config["download_history"].append({
            "date": datetime.now().isoformat(),
            "downloaded": downloaded,
        })
        self.save_config()
        
        return downloaded
        
    def check_nhanes_cycle(self, verbose=True):
        """检查NHANES数据周期信息"""
        # NHANES数据发布规律:
        # - 2017-2018, 2019-2020 (COVID影响)
        # - 2021-2023 (当前最新)
        # - 通常2年为一个周期
        
        current_cycle = "2021-2023"
        
        if verbose:
            print("\n📅 NHANES数据周期信息:")
            print(f"   当前周期: {current_cycle}")
            print(f"   预计下次发布: 2025年初 (2023-2025数据)")
            print(f"   最后检查: {self.config.get('last_check', '从未')}")
            
        return current_cycle
        
    def generate_report(self, updates, downloaded):
        """生成更新报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "updates_found": len(updates),
            "downloaded": downloaded,
            "files": [],
        }
        
        # 列出当前文件
        for f in os.listdir(self.output_dir):
            if f.endswith('.xpt'):
                filepath = os.path.join(self.output_dir, f)
                size = os.path.getsize(filepath) / 1024
                report["files"].append({
                    "filename": f,
                    "size_kb": size,
                })
                
        return report
        
    def notify(self, downloaded, method="print"):
        """发送通知"""
        if not downloaded:
            return
            
        message = f"""
🔔 NHANES数据更新通知

时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

更新的文件:
"""
        for item in downloaded:
            message += f"- {item['file_id']}: {item['size_kb']:.1f} KB ({item['status']})\n"
            
        message += f"""
总更新: {len(downloaded)} 个文件

数据目录: {self.output_dir}
"""
        
        if method == "print":
            print("\n" + "=" * 60)
            print(message)
            print("=" * 60)
        elif method == "email" and self.config.get("notification_email"):
            self._send_email(message)
            
    def _send_email(self, message):
        """发送邮件通知 (需要配置SMTP)"""
        # 需要在配置中设置SMTP参数
        pass


def main():
    parser = argparse.ArgumentParser(description="NHANES数据自动更新工具")
    parser.add_argument("--check", action="store_true", help="仅检查更新")
    parser.add_argument("--download", action="store_true", help="下载更新")
    parser.add_argument("--report", action="store_true", help="生成报告")
    parser.add_argument("--auto", action="store_true", help="自动检查并下载")
    parser.add_argument("--notify", default="print", choices=["print", "email"], help="通知方式")
    
    args = parser.parse_args()
    
    updater = NHANESUpdater()
    
    # 检查周期信息
    updater.check_nhanes_cycle()
    
    if args.check or args.auto:
        # 检查更新
        updates = updater.check_updates()
        
        if not updates:
            print("\n✅ 没有可用更新")
            return
            
        print(f"\n发现 {len(updates)} 个需要处理的项目")
        
    if args.download or args.auto:
        # 检查并下载
        updates = updater.check_updates(verbose=False)
        downloaded = updater.download_updates(updates)
        
        # 通知
        updater.notify(downloaded, method=args.notify)
        
    if args.report:
        # 生成报告
        updates = updater.check_updates(verbose=False)
        downloaded = []
        report = updater.generate_report(updates, downloaded)
        
        report_file = "nhanes_update_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 报告已保存: {report_file}")
        
        print("\n📁 当前数据文件:")
        for f in report["files"]:
            print(f"   {f['filename']}: {f['size_kb']:.1f} KB")


if __name__ == "__main__":
    main()
