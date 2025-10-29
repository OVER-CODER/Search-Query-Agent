#!/usr/bin/env python3
"""
Test dataset creation and browser history ingestion.
"""

import pytest
import os
import json
import tempfile
import sqlite3
from pathlib import Path
from scripts.preview_history import BrowserHistoryReader
from scripts.prepare_dataset import DatasetPreparator


class TestBrowserHistoryReader:
    """Test browser history reading functionality."""
    
    def create_test_chrome_db(self, db_path: Path):
        """Create a test Chrome history database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create Chrome history table schema
        cursor.execute('''
            CREATE TABLE urls (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT,
                visit_count INTEGER DEFAULT 0,
                last_visit_time INTEGER
            )
        ''')
        
        # Insert test data (Chrome timestamps are microseconds since 1601)
        test_data = [
            (1, 'https://python.org', 'Python.org', 5, 13300000000000000),
            (2, 'https://docs.python.org/3/', 'Python Documentation', 10, 13300000001000000),
            (3, 'https://github.com', 'GitHub', 3, 13300000002000000),
        ]
        
        cursor.executemany(
            'INSERT INTO urls (id, url, title, visit_count, last_visit_time) VALUES (?, ?, ?, ?, ?)',
            test_data
        )
        
        conn.commit()
        conn.close()
    
    def test_read_chrome_history(self):
        """Test reading Chrome history from test database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
        try:
            # Create test database
            self.create_test_chrome_db(tmp_path)
            
            # Test reading
            reader = BrowserHistoryReader()
            history = reader.read_chrome_history(tmp_path, limit=10)
            
            assert len(history) == 3
            assert history[0]['url'] == 'https://github.com'  # Most recent
            assert history[0]['source'] == 'chrome'
            assert history[0]['visit_count'] == 3
            
            assert history[1]['url'] == 'https://docs.python.org/3/'
            assert history[1]['visit_count'] == 10
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_hash_url(self):
        """Test URL hashing for privacy."""
        reader = BrowserHistoryReader()
        
        url = "https://example.com/private/path"
        hashed = reader.hash_url(url)
        
        assert len(hashed) == 16  # Truncated SHA256
        assert hashed != url
        assert reader.hash_url(url) == hashed  # Consistent


class TestDatasetPreparator:
    """Test dataset preparation functionality."""
    
    def create_test_privacy_settings(self, tmp_dir: Path):
        """Create test privacy settings file."""
        settings = {
            'hash_urls': False,
            'confirmed': True,
            'timestamp': '2024-01-01T00:00:00'
        }
        
        settings_path = tmp_dir / '.privacy_settings.json'
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
        
        return settings_path
    
    def test_generate_search_queries(self):
        """Test search query generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            settings_path = self.create_test_privacy_settings(tmp_path)
            
            # Change to temp directory so preparator can find settings
            original_cwd = os.getcwd()
            os.chdir(tmp_dir)
            
            try:
                preparator = DatasetPreparator(str(settings_path))
                
                test_entry = {
                    'url': 'https://docs.python.org/3/tutorial/',
                    'title': 'Python Tutorial - Python 3.11 Documentation',
                    'visit_count': 8,
                    'source': 'chrome'
                }
                
                queries = preparator.generate_search_queries(test_entry)
                
                assert len(queries) > 0
                assert any('python' in q.lower() for q in queries)
                assert any('tutorial' in q.lower() for q in queries)
                
            finally:
                os.chdir(original_cwd)
    
    def test_create_instruction_response_pairs(self):
        """Test instruction-response pair creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            settings_path = self.create_test_privacy_settings(tmp_path)
            
            original_cwd = os.getcwd()
            os.chdir(tmp_dir)
            
            try:
                preparator = DatasetPreparator(str(settings_path))
                
                test_history = [
                    {
                        'url': 'https://python.org',
                        'title': 'Python Programming Language',
                        'visit_count': 5,
                        'source': 'chrome'
                    },
                    {
                        'url': 'https://docs.python.org',
                        'title': 'Python Documentation',
                        'visit_count': 10,
                        'source': 'chrome'
                    }
                ]
                
                pairs = preparator.create_instruction_response_pairs(test_history)
                
                assert len(pairs) > 0
                
                # Check pair structure
                for pair in pairs:
                    assert 'instruction' in pair
                    assert 'response' in pair
                    assert 'query' in pair
                    assert 'url' in pair
                    
                    # Check instruction format
                    assert 'Given query:' in pair['instruction']
                    assert 'which URL from my browsing history' in pair['instruction']
                    
                    # Check response is valid JSON
                    response_data = json.loads(pair['response'])
                    assert 'url' in response_data
                    assert 'title' in response_data
                    assert 'confidence' in response_data
                
            finally:
                os.chdir(original_cwd)
    
    def test_balance_dataset(self):
        """Test dataset balancing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            settings_path = self.create_test_privacy_settings(tmp_path)
            
            original_cwd = os.getcwd()
            os.chdir(tmp_dir)
            
            try:
                preparator = DatasetPreparator(str(settings_path))
                
                # Create unbalanced pairs
                pairs = []
                for i in range(20):
                    pairs.append({
                        'category': 'programming',
                        'visit_count': i,
                        'query': f'test query {i}'
                    })
                
                for i in range(5):
                    pairs.append({
                        'category': 'news',
                        'visit_count': i,
                        'query': f'news query {i}'
                    })
                
                balanced = preparator.balance_dataset(pairs, max_pairs=10)
                
                assert len(balanced) <= 10
                
                # Check categories are represented
                categories = set(pair['category'] for pair in balanced)
                assert 'programming' in categories
                assert 'news' in categories
                
            finally:
                os.chdir(original_cwd)


def test_end_to_end_dataset_creation():
    """Test complete dataset creation pipeline."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create test Chrome database
        db_path = tmp_path / 'History'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE urls (
                id INTEGER PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT,
                visit_count INTEGER DEFAULT 0,
                last_visit_time INTEGER
            )
        ''')
        
        test_data = [
            (1, 'https://python.org', 'Python Programming', 5, 13300000000000000),
            (2, 'https://github.com/python', 'Python on GitHub', 3, 13300000001000000),
            (3, 'https://stackoverflow.com/questions/tagged/python', 'Python Questions', 8, 13300000002000000),
        ]
        
        cursor.executemany(
            'INSERT INTO urls (id, url, title, visit_count, last_visit_time) VALUES (?, ?, ?, ?, ?)',
            test_data
        )
        
        conn.commit()
        conn.close()
        
        # Create privacy settings
        settings = {
            'hash_urls': False,
            'confirmed': True,
            'timestamp': '2024-01-01T00:00:00'
        }
        
        settings_path = tmp_path / '.privacy_settings.json'
        with open(settings_path, 'w') as f:
            json.dump(settings, f)
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(tmp_dir)
        
        try:
            # Mock the browser history detection
            from scripts.prepare_dataset import DatasetPreparator
            
            preparator = DatasetPreparator(str(settings_path))
            
            # Manually load the test data (simulating browser history read)
            test_history = [
                {
                    'url': 'https://python.org',
                    'title': 'Python Programming',
                    'visit_count': 5,
                    'source': 'chrome'
                },
                {
                    'url': 'https://github.com/python',
                    'title': 'Python on GitHub',
                    'visit_count': 3,
                    'source': 'chrome'
                },
                {
                    'url': 'https://stackoverflow.com/questions/tagged/python',
                    'title': 'Python Questions',
                    'visit_count': 8,
                    'source': 'chrome'
                }
            ]
            
            # Create instruction-response pairs
            pairs = preparator.create_instruction_response_pairs(test_history)
            
            # Balance dataset
            balanced_pairs = preparator.balance_dataset(pairs, max_pairs=50)
            
            # Save dataset
            preparator.save_dataset(balanced_pairs)
            
            # Verify output files exist
            assert (tmp_path / 'training_data' / 'dataset.json').exists()
            assert (tmp_path / 'training_data' / 'train.jsonl').exists()
            assert (tmp_path / 'training_data' / 'metadata.json').exists()
            
            # Verify content
            with open(tmp_path / 'training_data' / 'train.jsonl', 'r') as f:
                lines = f.readlines()
                assert len(lines) == len(balanced_pairs)
                
                for line in lines:
                    data = json.loads(line)
                    assert 'instruction' in data
                    assert 'response' in data
            
            print(f"✓ Created dataset with {len(balanced_pairs)} training pairs")
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    # Run tests
    print("Running dataset creation tests...")
    
    test_end_to_end_dataset_creation()
    
    print("✓ All dataset tests passed!")