# tools/web_fetcher.py
#
# Fetch and parse web content
# - Get page content
# - Extract text, links, images
# - Summarize (uses local model)
# - No external APIs (privacy-first)

import re
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    """Extract text content from HTML."""
    
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.current_tag = None
        self.skip_tags = {'script', 'style', 'nav', 'footer', 'header', 'aside'}
        self.links = []
        self.images = []
        self.meta = {}
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag.lower()
        attrs_dict = dict(attrs)
        
        if tag.lower() == 'a' and 'href' in attrs_dict:
            self.links.append({
                'url': attrs_dict['href'],
                'text': ''
            })
        elif tag.lower() == 'img' and 'src' in attrs_dict:
            self.images.append({
                'src': attrs_dict['src'],
                'alt': attrs_dict.get('alt', '')
            })
        elif tag.lower() == 'meta':
            name = attrs_dict.get('name') or attrs_dict.get('property', '')
            content = attrs_dict.get('content', '')
            if name and content:
                self.meta[name] = content
    
    def handle_endtag(self, tag):
        self.current_tag = None
    
    def handle_data(self, data):
        if self.current_tag in self.skip_tags:
            return
        
        text = data.strip()
        if text:
            self.text_parts.append(text)
            
            # Add text to last link if we're inside an anchor
            if self.current_tag == 'a' and self.links:
                self.links[-1]['text'] += ' ' + text
    
    def get_text(self) -> str:
        return ' '.join(self.text_parts)


class WebFetcher:
    """Fetch and parse web content without external APIs."""

    def __init__(self, timeout: int = 30, max_size: int = 500000):
        self.timeout = timeout
        self.max_size = max_size
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; NeuralAI/1.0)',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'en-US,en;q=0.9',
        }

    def fetch(self, url: str) -> Dict[str, Any]:
        """
        Fetch a URL and return parsed content.
        
        Returns:
            {
                "success": bool,
                "url": str,
                "status": int,
                "title": str,
                "text": str,
                "links": list,
                "images": list,
                "meta": dict
            }
        """
        try:
            request = urllib.request.Request(url, headers=self.headers)
            
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = response.status
                content = response.read(self.max_size).decode('utf-8', errors='ignore')
            
            # Parse HTML
            parser = TextExtractor()
            parser.feed(content)
            
            # Extract title
            title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else ""
            
            # Clean title
            title = re.sub(r'\s+', ' ', title)
            
            return {
                "success": True,
                "url": url,
                "status": status,
                "title": title,
                "text": parser.get_text()[:10000],  # Limit text length
                "links": parser.links[:50],  # Limit links
                "images": parser.images[:20],  # Limit images
                "meta": parser.meta
            }
            
        except urllib.error.HTTPError as e:
            return {
                "success": False,
                "error": f"HTTP Error: {e.code} {e.reason}",
                "url": url,
                "status": e.code,
                "title": "",
                "text": "",
                "links": [],
                "images": [],
                "meta": {}
            }
        except urllib.error.URLError as e:
            return {
                "success": False,
                "error": f"URL Error: {str(e.reason)}",
                "url": url,
                "status": 0,
                "title": "",
                "text": "",
                "links": [],
                "images": [],
                "meta": {}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "status": 0,
                "title": "",
                "text": "",
                "links": [],
                "images": [],
                "meta": {}
            }

    def extract_text(self, url: str) -> Dict[str, Any]:
        """
        Extract only text content from a URL.
        
        Returns:
            {
                "success": bool,
                "url": str,
                "title": str,
                "text": str,
                "word_count": int
            }
        """
        result = self.fetch(url)
        
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "url": url,
                "title": "",
                "text": "",
                "word_count": 0
            }
        
        text = result["text"]
        words = text.split()
        
        return {
            "success": True,
            "url": url,
            "title": result["title"],
            "text": text,
            "word_count": len(words)
        }

    def extract_links(self, url: str) -> Dict[str, Any]:
        """
        Extract all links from a URL.
        
        Returns:
            {
                "success": bool,
                "url": str,
                "links": [{"url": str, "text": str}],
                "total": int
            }
        """
        result = self.fetch(url)
        
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "url": url,
                "links": [],
                "total": 0
            }
        
        # Filter and clean links
        clean_links = []
        for link in result["links"]:
            href = link.get("url", "")
            text = link.get("text", "").strip()
            
            # Skip empty or javascript links
            if not href or href.startswith("javascript:"):
                continue
            
            # Make relative URLs absolute
            if href.startswith("/"):
                from urllib.parse import urljoin
                href = urljoin(url, href)
            
            clean_links.append({
                "url": href,
                "text": text[:100]  # Limit text length
            })
        
        return {
            "success": True,
            "url": url,
            "links": clean_links,
            "total": len(clean_links)
        }

    def get_meta(self, url: str) -> Dict[str, Any]:
        """
        Extract metadata (OpenGraph, Twitter cards, etc) from a URL.
        
        Returns:
            {
                "success": bool,
                "url": str,
                "title": str,
                "description": str,
                "image": str,
                "meta": dict
            }
        """
        result = self.fetch(url)
        
        if not result["success"]:
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "url": url,
                "title": "",
                "description": "",
                "image": "",
                "meta": {}
            }
        
        meta = result["meta"]
        
        # Extract common metadata
        title = (
            meta.get("og:title") or 
            meta.get("twitter:title") or 
            result["title"]
        )
        description = (
            meta.get("og:description") or 
            meta.get("twitter:description") or 
            meta.get("description", "")
        )
        image = (
            meta.get("og:image") or 
            meta.get("twitter:image", "")
        )
        
        return {
            "success": True,
            "url": url,
            "title": title,
            "description": description,
            "image": image,
            "meta": meta
        }

    def check_status(self, url: str) -> Dict[str, Any]:
        """
        Check if a URL is accessible (HEAD request).
        
        Returns:
            {
                "success": bool,
                "url": str,
                "status": int,
                "reachable": bool
            }
        """
        try:
            request = urllib.request.Request(url, headers=self.headers, method='HEAD')
            
            with urllib.request.urlopen(request, timeout=10) as response:
                return {
                    "success": True,
                    "url": url,
                    "status": response.status,
                    "reachable": response.status < 400
                }
                
        except urllib.error.HTTPError as e:
            return {
                "success": True,
                "url": url,
                "status": e.code,
                "reachable": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "status": 0,
                "reachable": False
            }


if __name__ == "__main__":
    # Test the web fetcher
    fetcher = WebFetcher()
    
    print("Testing fetch...")
    result = fetcher.fetch("https://example.com")
    print(f"Title: {result.get('title')}")
    print(f"Text length: {len(result.get('text', ''))}")
    print(f"Links: {len(result.get('links', []))}")
    
    print("\nTesting meta extraction...")
    result = fetcher.get_meta("https://example.com")
    print(f"Meta: {result}")
