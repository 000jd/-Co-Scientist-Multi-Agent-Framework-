"""Fetch and extract readable text from webpages."""

import httpx
from typing import Optional
from bs4 import BeautifulSoup


class WebReader:
    """Fetch and extract readable text from webpages."""
    
    def __init__(self, max_length: int = 8000):
        self.max_length = max_length
        
    async def fetch(self, url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Remove script/style
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                    
                # Try to find main content
                main = soup.find("main") or soup.find("article") or soup.find("body")
                text = main.get_text(separator="\n", strip=True) if main else ""
                
                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = "\n".join(lines)
                
                return text[:self.max_length]
        except Exception:
            return None
