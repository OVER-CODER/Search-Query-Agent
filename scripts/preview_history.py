#!/usr/bin/env python3
"""
Preview browser history safely before training.
Provides user confirmation and options for privacy protection.
"""

import os
import sys
import sqlite3
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import hashlib


class BrowserHistoryReader:
    """Safely read and preview browser history."""
    
    def __init__(self):
        self.chrome_paths = [
            "~/Library/Application Support/Google/Chrome/Default/History",  # macOS
            "~/.config/google-chrome/Default/History",  # Linux
            "~/AppData/Local/Google/Chrome/User Data/Default/History",  # Windows
        ]
        self.firefox_paths = [
            "~/Library/Application Support/Firefox/Profiles/*/places.sqlite",  # macOS
            "~/.mozilla/firefox/*/places.sqlite",  # Linux
            "~/AppData/Roaming/Mozilla/Firefox/Profiles/*/places.sqlite",  # Windows
        ]
    
    def find_chrome_history(self) -> Optional[Path]:
        """Find Chrome history file."""
        for path_str in self.chrome_paths:
            path = Path(path_str).expanduser()
            if path.exists():
                return path
        return None
    
    def find_firefox_history(self) -> Optional[Path]:
        """Find Firefox history file."""
        import glob
        for path_pattern in self.firefox_paths:
            paths = glob.glob(str(Path(path_pattern).expanduser()))
            if paths:
                return Path(paths[0])
        return None
    
    def read_chrome_history(self, db_path: Path, limit: int = 1000) -> List[Dict]:
        """Read Chrome history safely."""
        # Copy to temp file to avoid locking issues
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            shutil.copy2(db_path, tmp.name)
            tmp_path = tmp.name
        
        try:
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
            
            query = """
                SELECT url, title, visit_count, last_visit_time
                FROM urls 
                WHERE visit_count > 0 
                ORDER BY last_visit_time DESC 
                LIMIT ?
            """
            
            cursor.execute(query, (limit,))
            results = []
            
            for url, title, visit_count, last_visit_time in cursor.fetchall():
                # Convert Chrome timestamp (microseconds since 1601) to datetime
                if last_visit_time:
                    timestamp = datetime.fromtimestamp(
                        (last_visit_time - 11644473600000000) / 1000000
                    )
                else:
                    timestamp = None
                
                results.append({
                    'url': url,
                    'title': title or '',
                    'visit_count': visit_count,
                    'last_visit': timestamp,
                    'source': 'chrome'
                })
            
            conn.close()
            return results
            
        finally:
            os.unlink(tmp_path)
    
    def read_firefox_history(self, db_path: Path, limit: int = 1000) -> List[Dict]:
        """Read Firefox history safely."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            shutil.copy2(db_path, tmp.name)
            tmp_path = tmp.name
        
        try:
            conn = sqlite3.connect(tmp_path)
            cursor = conn.cursor()
            
            query = """
                SELECT p.url, p.title, p.visit_count, h.visit_date
                FROM moz_places p
                LEFT JOIN moz_historyvisits h ON p.id = h.place_id
                WHERE p.visit_count > 0
                ORDER BY h.visit_date DESC
                LIMIT ?
            """
            
            cursor.execute(query, (limit,))
            results = []
            
            for url, title, visit_count, visit_date in cursor.fetchall():
                # Convert Firefox timestamp (microseconds since 1970)
                if visit_date:
                    timestamp = datetime.fromtimestamp(visit_date / 1000000)
                else:
                    timestamp = None
                
                results.append({
                    'url': url,
                    'title': title or '',
                    'visit_count': visit_count,
                    'last_visit': timestamp,
                    'source': 'firefox'
                })
            
            conn.close()
            return results
            
        finally:
            os.unlink(tmp_path)
    
    def hash_url(self, url: str) -> str:
        """Hash URL for privacy."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def preview_history(self, limit: int = 20, hash_urls: bool = False) -> List[Dict]:
        """Preview browser history with privacy options."""
        all_history = []
        
        # Try Chrome
        chrome_path = self.find_chrome_history()
        if chrome_path:
            print(f"Found Chrome history: {chrome_path}")
            try:
                chrome_history = self.read_chrome_history(chrome_path, limit)
                all_history.extend(chrome_history)
                print(f"Loaded {len(chrome_history)} Chrome entries")
            except Exception as e:
                print(f"Error reading Chrome history: {e}")
        
        # Try Firefox
        firefox_path = self.find_firefox_history()
        if firefox_path:
            print(f"Found Firefox history: {firefox_path}")
            try:
                firefox_history = self.read_firefox_history(firefox_path, limit)
                all_history.extend(firefox_history)
                print(f"Loaded {len(firefox_history)} Firefox entries")
            except Exception as e:
                print(f"Error reading Firefox history: {e}")
        
        if not all_history:
            print("No browser history found!")
            return []
        
        # Sort by last visit time
        all_history.sort(key=lambda x: x['last_visit'] or datetime.min, reverse=True)
        
        # Apply privacy protection
        if hash_urls:
            for entry in all_history:
                entry['url_hash'] = self.hash_url(entry['url'])
                entry['url'] = f"[HASHED: {entry['url_hash']}]"
        
        return all_history[:limit]


def main():
    """Main preview function with user confirmation."""
    print("=== Browser History Preview ===")
    print("This tool will preview your browser history for training dataset creation.")
    print("Your privacy is important - you have full control over what data is used.")
    print()
    
    # Privacy options
    print("Privacy Options:")
    print("1. Preview with full URLs")
    print("2. Preview with hashed URLs (more private)")
    print("3. Cancel")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "3":
        print("Cancelled.")
        return
    
    hash_urls = choice == "2"
    
    # Read history
    reader = BrowserHistoryReader()
    history = reader.preview_history(limit=20, hash_urls=hash_urls)
    
    if not history:
        print("No browser history found. Make sure Chrome or Firefox is installed.")
        return
    
    print(f"\n=== Preview of {len(history)} Recent Entries ===")
    for i, entry in enumerate(history[:10], 1):
        last_visit = entry['last_visit'].strftime('%Y-%m-%d %H:%M') if entry['last_visit'] else 'Unknown'
        print(f"{i:2d}. [{entry['source']}] {entry['url']}")
        print(f"    Title: {entry['title'][:80]}...")
        print(f"    Visits: {entry['visit_count']}, Last: {last_visit}")
        print()
    
    if len(history) > 10:
        print(f"... and {len(history) - 10} more entries")
    
    print(f"\nTotal entries available: {len(history)}")
    print(f"Privacy mode: {'Hashed URLs' if hash_urls else 'Full URLs'}")
    
    # Confirmation
    print("\n=== Confirmation Required ===")
    print("Do you want to proceed with dataset creation using this browser history?")
    print("This will create a training dataset for personalizing the search model.")
    
    if hash_urls:
        print("URLs will be hashed for privacy protection.")
    else:
        print("WARNING: Full URLs will be included in the training data.")
    
    confirm = input("Proceed? (yes/no): ").strip().lower()
    
    if confirm in ['yes', 'y']:
        print("✓ Confirmed. You can now run prepare_dataset.py")
        
        # Save settings for dataset preparation
        settings = {
            'hash_urls': hash_urls,
            'confirmed': True,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open('.privacy_settings.json', 'w') as f:
            json.dump(settings, f, indent=2)
        
        print("Privacy settings saved to .privacy_settings.json")
    else:
        print("Cancelled. No dataset will be created.")


if __name__ == "__main__":
    main()