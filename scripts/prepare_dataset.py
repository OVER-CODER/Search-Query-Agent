#!/usr/bin/env python3
"""
Prepare training dataset from browser history.
Creates instruction-response pairs for QLoRA fine-tuning.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import random
import re
from urllib.parse import urlparse
from scripts.preview_history import BrowserHistoryReader


class DatasetPreparator:
    """Prepare browser history for QLoRA training."""
    
    def __init__(self, privacy_settings_path: str = ".privacy_settings.json"):
        self.privacy_settings = self.load_privacy_settings(privacy_settings_path)
        self.reader = BrowserHistoryReader()
    
    def load_privacy_settings(self, path: str) -> Dict:
        """Load privacy settings from preview step."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Privacy settings not found. Please run preview_history.py first."
            )
        
        with open(path, 'r') as f:
            settings = json.load(f)
        
        if not settings.get('confirmed', False):
            raise ValueError("User confirmation required. Please run preview_history.py first.")
        
        return settings
    
    def clean_title(self, title: str) -> str:
        """Clean page title for better training."""
        if not title:
            return ""
        
        # Remove common suffixes
        title = re.sub(r' - (Google|YouTube|Wikipedia|Stack Overflow|GitHub).*$', '', title)
        title = re.sub(r' \| .*$', '', title)  # Remove after pipe
        title = re.sub(r' :: .*$', '', title)  # Remove after double colon
        
        # Clean up whitespace
        title = ' '.join(title.split())
        
        return title[:100]  # Limit length
    
    def extract_domain(self, url: str) -> str:
        """Extract clean domain name."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return ""
    
    def categorize_url(self, url: str, title: str) -> str:
        """Categorize URL based on domain and content."""
        domain = self.extract_domain(url)
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Programming/Tech
        tech_domains = ['stackoverflow.com', 'github.com', 'python.org', 'docs.python.org']
        if any(d in domain for d in tech_domains):
            return "programming"
        
        # Documentation
        if 'docs.' in domain or 'documentation' in title_lower:
            return "documentation"
        
        # News
        news_domains = ['news.', 'cnn.com', 'bbc.com', 'reuters.com']
        if any(d in domain for d in news_domains):
            return "news"
        
        # Social
        social_domains = ['twitter.com', 'reddit.com', 'linkedin.com']
        if any(d in domain for d in social_domains):
            return "social"
        
        # Shopping
        if 'shop' in domain or 'amazon.com' in domain:
            return "shopping"
        
        return "general"
    
    def generate_search_queries(self, entry: Dict) -> List[str]:
        """Generate realistic search queries that would lead to this URL."""
        title = entry['title']
        url = entry['url']
        domain = self.extract_domain(url)
        category = self.categorize_url(url, title)
        
        queries = []
        
        # Title-based queries
        if title:
            clean_title = self.clean_title(title)
            if clean_title:
                # Full title
                queries.append(clean_title)
                
                # First few words
                words = clean_title.split()
                if len(words) > 3:
                    queries.append(' '.join(words[:3]))
                
                # Extract key terms
                key_terms = [w for w in words if len(w) > 4 and w.isalpha()]
                if key_terms:
                    queries.append(' '.join(key_terms[:2]))
        
        # Domain-based queries
        if domain:
            domain_parts = domain.split('.')
            if len(domain_parts) > 1:
                main_domain = domain_parts[-2]  # e.g., 'github' from 'github.com'
                queries.append(main_domain)
        
        # Category-based queries
        category_queries = {
            "programming": ["python tutorial", "coding help", "programming guide"],
            "documentation": ["api documentation", "user guide", "manual"],
            "news": ["latest news", "breaking news", "current events"],
            "social": ["social media", "community discussion"],
            "shopping": ["buy online", "product review", "shopping"],
            "general": ["information", "learn about", "find out"]
        }
        
        if category in category_queries:
            queries.extend(random.sample(category_queries[category], 1))
        
        # URL path-based queries
        path_parts = urlparse(url).path.split('/')
        for part in path_parts:
            if part and len(part) > 3 and part.replace('-', '').replace('_', '').isalpha():
                queries.append(part.replace('-', ' ').replace('_', ' '))
        
        # Return unique, non-empty queries
        unique_queries = list(set(q.strip() for q in queries if q.strip()))
        return unique_queries[:5]  # Limit to 5 queries per URL
    
    def create_instruction_response_pairs(self, history: List[Dict]) -> List[Dict]:
        """Create training pairs in instruction-response format."""
        pairs = []
        
        for entry in history:
            url = entry['url']
            title = self.clean_title(entry['title'])
            domain = self.extract_domain(url)
            visit_count = entry['visit_count']
            
            # Skip URLs that are too generic or low-quality
            if visit_count < 2:
                continue
            
            # Generate search queries for this URL
            queries = self.generate_search_queries(entry)
            
            for query in queries:
                # Create instruction-response pair
                instruction = f"Given query: '{query}', which URL from my browsing history best answers it?"
                
                response_data = {
                    "url": url,
                    "title": title,
                    "domain": domain,
                    "confidence": min(visit_count / 10.0, 1.0),  # Normalize visit count
                    "reason": f"You previously visited this {entry['source']} page {visit_count} times"
                }
                
                if self.privacy_settings.get('hash_urls', False):
                    response_data["url_hash"] = self.reader.hash_url(url)
                
                response = json.dumps(response_data, indent=None)
                
                pairs.append({
                    "instruction": instruction,
                    "response": response,
                    "query": query,
                    "url": url,
                    "domain": domain,
                    "visit_count": visit_count,
                    "category": self.categorize_url(url, title)
                })
        
        return pairs
    
    def balance_dataset(self, pairs: List[Dict], max_pairs: int = 1000) -> List[Dict]:
        """Balance dataset across categories and remove duplicates."""
        # Group by category
        by_category = {}
        for pair in pairs:
            category = pair['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(pair)
        
        # Balance categories
        balanced_pairs = []
        pairs_per_category = max_pairs // len(by_category)
        
        for category, category_pairs in by_category.items():
            # Sort by visit count (higher = better)
            category_pairs.sort(key=lambda x: x['visit_count'], reverse=True)
            
            # Take top pairs from this category
            selected = category_pairs[:pairs_per_category]
            balanced_pairs.extend(selected)
        
        # Shuffle to mix categories
        random.shuffle(balanced_pairs)
        
        return balanced_pairs[:max_pairs]
    
    def save_dataset(self, pairs: List[Dict], output_dir: str = "training_data"):
        """Save dataset in multiple formats."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON for inspection
        with open(f"{output_dir}/dataset.json", 'w') as f:
            json.dump(pairs, f, indent=2, default=str)
        
        # Save as JSONL for training
        with open(f"{output_dir}/train.jsonl", 'w') as f:
            for pair in pairs:
                training_example = {
                    "instruction": pair["instruction"],
                    "response": pair["response"]
                }
                f.write(json.dumps(training_example) + '\n')
        
        # Save metadata
        metadata = {
            "total_pairs": len(pairs),
            "categories": {},
            "privacy_settings": self.privacy_settings,
            "created_at": datetime.now().isoformat()
        }
        
        for pair in pairs:
            category = pair['category']
            metadata["categories"][category] = metadata["categories"].get(category, 0) + 1
        
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_dir}/")
        print(f"Total training pairs: {len(pairs)}")
        print(f"Categories: {metadata['categories']}")
    
    def prepare_dataset(self, max_history: int = 5000, max_pairs: int = 1000):
        """Main dataset preparation pipeline."""
        print("=== Preparing Training Dataset ===")
        
        # Load browser history
        print("Loading browser history...")
        history = self.reader.preview_history(limit=max_history, 
                                              hash_urls=self.privacy_settings.get('hash_urls', False))
        
        if not history:
            raise ValueError("No browser history found!")
        
        print(f"Loaded {len(history)} history entries")
        
        # Create instruction-response pairs
        print("Creating instruction-response pairs...")
        pairs = self.create_instruction_response_pairs(history)
        print(f"Generated {len(pairs)} initial pairs")
        
        # Balance and limit dataset
        print("Balancing dataset...")
        balanced_pairs = self.balance_dataset(pairs, max_pairs)
        print(f"Final dataset: {len(balanced_pairs)} pairs")
        
        # Save dataset
        self.save_dataset(balanced_pairs)
        
        return balanced_pairs


def main():
    """Main function."""
    try:
        preparator = DatasetPreparator()
        dataset = preparator.prepare_dataset()
        
        print("\n✓ Dataset preparation complete!")
        print("Next steps:")
        print("1. Review the dataset in training_data/dataset.json")
        print("2. Run training/train_qlora.py to start fine-tuning")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you've run scripts/preview_history.py first")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())