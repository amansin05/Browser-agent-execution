"""Browser Agent — Groq Scout brain + Playwright MCP hands.

Importing the package loads environment variables from a local .env once, so any module
(CLI, server, scripts, tests) sees GROQ_API_KEY / PLAYWRIGHT_MCP_EXTENSION_TOKEN.
"""

from dotenv import load_dotenv

load_dotenv()

__version__ = "1.0.0"
