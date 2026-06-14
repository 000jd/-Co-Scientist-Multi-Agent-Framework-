"""
LocalActionBridge — Kimi Work WebBridge equivalent.

Uses Playwright CDP to drive a real browser session.
Falls back to httpx if Playwright is unavailable.

Install: pip install playwright && playwright install chromium
"""

from __future__ import annotations

import re
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("local_action_bridge")

MAX_CONTENT_CHARS = 12_000


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class BrowseResult:
    def __init__(self, url: str, title: str, content: str, success: bool, error: Optional[str] = None):
        self.url = url
        self.title = title
        self.content = content
        self.success = success
        self.error = error


class LocalActionBridge:
    """
    Browser automation bridge using Playwright.
    
    Kimi Work pattern: drives your real browser session so authenticated
    pages (SaaS dashboards, paywalled papers, internal tools) are reachable.
    
    In this implementation we launch a persistent browser context that can
    optionally load an existing Chrome profile (for auth cookies).
    """
    
    def __init__(
        self,
        headless: bool = True,
        user_data_dir: Optional[str] = None,  # load existing Chrome profile
        timeout_ms: int = 30_000,
    ):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.timeout_ms = timeout_ms
        self._playwright = None
        self._browser = None
        self._context = None
        self._available = False
        
    async def start(self) -> bool:
        """Launch browser. Returns False if Playwright not installed."""
        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            
            launch_kwargs: Dict[str, Any] = {
                "headless": self.headless,
                "args": ["--no-sandbox", "--disable-dev-shm-usage"],
            }
            
            if self.user_data_dir:
                from pathlib import Path
                profile_dir = Path("./data/browser_profiles") / Path(self.user_data_dir).name
                profile_dir.mkdir(parents=True, exist_ok=True)
                # Persistent context = inherits existing cookies/auth
                self._context = await self._playwright.chromium.launch_persistent_context(
                    str(profile_dir),
                    **launch_kwargs
                )
            else:
                self._browser = await self._playwright.chromium.launch(**launch_kwargs)
                self._context = await self._browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                
            self._available = True
            logger.info("LocalActionBridge: Playwright browser launched")
            return True
            
        except ImportError:
            logger.warning("Playwright not installed. Falling back to httpx. "
                          "Run: pip install playwright && playwright install chromium")
            self._available = False
            return False
        except Exception as e:
            logger.warning(f"Playwright launch failed: {e}. Falling back to httpx.")
            self._available = False
            return False
            
    async def navigate(self, url: str) -> BrowseResult:
        """Navigate to URL and extract readable text content."""
        if not self._available:
            return await self._httpx_fallback(url)
            
        try:
            page = await self._context.new_page()
            try:
                response = await page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                await page.wait_for_load_state("networkidle", timeout=5000)
                
                title = await page.title()
                
                # Extract text — remove nav/footer noise
                content = await page.evaluate("""() => {
                    const remove = ['script','style','nav','footer','header','aside',
                                   '[role="navigation"]','[role="banner"]'];
                    remove.forEach(sel => document.querySelectorAll(sel).forEach(e => e.remove()));
                    const main = document.querySelector('main,article,[role="main"]') || document.body;
                    return main ? main.innerText : document.body.innerText;
                }""")
                
                content = _clean(content)[:MAX_CONTENT_CHARS]
                return BrowseResult(url=url, title=title, content=content, success=True)
                
            finally:
                await page.close()
                
        except Exception as e:
            logger.debug(f"Playwright navigate failed for {url}: {e}")
            return await self._httpx_fallback(url)
            
    async def click_and_extract(self, url: str, selector: str) -> BrowseResult:
        """Click an element then extract updated page content. For JS-driven pages."""
        if not self._available:
            return BrowseResult(url=url, title="", content="", success=False, error="Playwright not available")
            
        try:
            page = await self._context.new_page()
            try:
                await page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                await page.click(selector, timeout=5000)
                await page.wait_for_load_state("networkidle", timeout=5000)
                
                content = await page.evaluate("() => document.body.innerText")
                return BrowseResult(url=url, title=await page.title(), 
                                   content=_clean(content)[:MAX_CONTENT_CHARS], success=True)
            finally:
                await page.close()
        except Exception as e:
            return BrowseResult(url=url, title="", content="", success=False, error=str(e))
            
    async def screenshot(self, url: str) -> Optional[bytes]:
        """Capture screenshot for multimodal K2.6 input."""
        if not self._available:
            return None
        try:
            page = await self._context.new_page()
            try:
                await page.goto(url, timeout=self.timeout_ms)
                return await page.screenshot(type="png", full_page=False)
            finally:
                await page.close()
        except Exception:
            return None
            
    async def _httpx_fallback(self, url: str) -> BrowseResult:
        """Passive HTTP fetch when Playwright unavailable."""
        try:
            import httpx
            from bs4 import BeautifulSoup
            
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                    
                title = soup.find("title")
                title_text = title.get_text() if title else ""
                main = soup.find("main") or soup.find("article") or soup.find("body")
                text = main.get_text(separator="\n", strip=True) if main else ""
                
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                content = _clean("\n".join(lines))[:MAX_CONTENT_CHARS]
                
                return BrowseResult(url=url, title=title_text, content=content, success=True)
        except Exception as e:
            return BrowseResult(url=url, title="", content="", success=False, error=str(e))
            
    async def stop(self):
        """Clean up browser resources."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
