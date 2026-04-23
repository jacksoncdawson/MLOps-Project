"""
PacSun retail scraper for trndly trend signals.

Scrapes PacSun's "New Arrivals" and category pages using a real browser
(Playwright) to count how often each color, category, and material attribute
appears across featured product listings. Normalizes those counts to 0–1 and
writes the result as trend_signals.csv.

WHERE EACH ATTRIBUTE COMES FROM
--------------------------------
- category : product title keywords  ("jeans" → pants, "hoodie" → tops, etc.)
- material  : product title keywords  ("linen", "denim", "knit", etc.)
             + category inference when no material keyword is in the title
- color     : color swatch aria-labels  (PacSun puts the color name in the
             aria-label / title attribute of each swatch button, NOT in the
             product title text)
             + product title keywords as a fallback

ZERO TITLES ON SUBSEQUENT PAGES
--------------------------------
PacSun is a React/Next.js SPA. After navigating to a new URL the product grid
re-renders asynchronously, so we use page.wait_for_selector() to block until
at least one product tile is visible before extracting.

BOT MANAGEMENT
---------------
PacSun is fronted by PerimeterX / HUMAN bot defender. It throws a
"Press & Hold" CAPTCHA at any visitor whose browser fingerprint looks
automated. That challenge cannot be solved purely by waiting — it needs a
real human mouse-down event with realistic pressure/timing data.

Strategy used here:

1. Use a PERSISTENT browser profile (--profile-dir). Once you solve the
   challenge by hand, PerimeterX drops a `_px*` cookie that's good for
   hours/days. Subsequent runs reuse that cookie and skip the challenge.
2. On the first run pass --headless false --manual-solve. The scraper
   will open a real browser window, pause when it detects the challenge,
   and let you press-and-hold to clear it. Then it continues automatically.
3. After that, plain `python pacsun_scraper.py` re-uses the profile.

If you cannot interactively solve the challenge (e.g. running on a server),
this site realistically requires a residential-proxy + stealth solution
or an official partner data feed.

SETUP (one-time)
----------------
  pip install playwright
  playwright install chromium

Usage:
  # First run — visible browser, you solve the press-and-hold once:
  python pacsun_scraper.py --headless false --manual-solve

  # Subsequent runs — re-uses cached PerimeterX cookie:
  python pacsun_scraper.py
  python pacsun_scraper.py --output-path path/to/trend_signals.csv
  python pacsun_scraper.py --existing-path trend_signals.csv --blend-weight 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipelines.training.feature_contract import (  # noqa: E402
    DEFAULT_MISSING_SCORE,
    FEATURE_TYPES,
    validate_trend_signals_frame,
)

# --------------------------------------------------------------------------- #
# Target pages                                                                  #
# --------------------------------------------------------------------------- #

PACSUN_PAGES = [
    {"url": "https://www.pacsun.com/new-arrivals/womens/",          "label": "women new arrivals"},
    {"url": "https://www.pacsun.com/new-arrivals/mens/",            "label": "men new arrivals"},
    {"url": "https://www.pacsun.com/womens/tops/",                  "label": "women tops"},
    {"url": "https://www.pacsun.com/mens/tops/",                    "label": "men tops"},
    {"url": "https://www.pacsun.com/womens/bottoms/",               "label": "women bottoms"},
    {"url": "https://www.pacsun.com/mens/bottoms/",                 "label": "men bottoms"},
    {"url": "https://www.pacsun.com/womens/jeans/",                 "label": "women jeans"},
    {"url": "https://www.pacsun.com/mens/jeans/",                   "label": "men jeans"},
    {"url": "https://www.pacsun.com/womens/dresses/",               "label": "women dresses"},
    {"url": "https://www.pacsun.com/womens/jackets-and-coats/",     "label": "women outerwear"},
    {"url": "https://www.pacsun.com/mens/jackets-and-coats/",       "label": "men outerwear"},
    {"url": "https://www.pacsun.com/womens/shoes/",                 "label": "women shoes"},
    {"url": "https://www.pacsun.com/mens/shoes/",                   "label": "men shoes"},
    {"url": "https://www.pacsun.com/womens/accessories/",           "label": "women accessories"},
    {"url": "https://www.pacsun.com/mens/accessories/",             "label": "men accessories"},
]

# --------------------------------------------------------------------------- #
# Selectors                                                                     #
#                                                                               #
# PRODUCT NAME selectors — tried in order, first non-empty match wins.         #
# PacSun uses a Salesforce Commerce Cloud / Next.js storefront with           #
# generated class names; these cover known patterns.                          #
# --------------------------------------------------------------------------- #

PRODUCT_NAME_SELECTORS = [
    "[data-testid='product-name']",
    "[data-testid='product-tile-name']",
    "[data-testid='product-card-title']",
    ".product-tile .product-name",
    ".product-tile .pdp-link",
    ".product-name a",
    ".product-name",
    "[class*='ProductName']",
    "[class*='product-name']",
    "[class*='productName']",
    "[class*='ProductTile'] [class*='name']",
    "[class*='product-tile'] [class*='name']",
    "[class*='ProductCard'] h2",
    "[class*='ProductCard'] h3",
    "[class*='product-card'] h2",
    "[class*='product-card'] h3",
    "article h2",
    "article h3",
    "li[class*='product'] h2",
    "li[class*='product'] h3",
    "a[class*='product'] [class*='title']",
]

# COLOR SWATCH selectors — PacSun attaches the color name to swatch buttons
# via aria-label / title (e.g. aria-label="Black") rather than putting it in
# the product title. We try these to get per-product color data.
COLOR_SWATCH_SELECTORS = [
    "[data-testid='color-swatch'] [aria-label]",
    "[class*='ColorSwatch'] [aria-label]",
    "[class*='color-swatch'] [aria-label]",
    ".swatches .swatch-circle[title]",
    ".swatches a[title]",
    ".color-swatches a[title]",
    "[class*='Swatch'] button[aria-label]",
    "[class*='swatch'] button[aria-label]",
    "[class*='swatch'] a[title]",
    "button[aria-label][class*='color']",
    "button[aria-label][class*='Color']",
    "[data-color]",
    "[data-color-name]",
    "[data-attr-value]",
]

# Selector to wait for before extracting — signals the product grid is ready
PRODUCT_GRID_WAIT_SELECTORS = [
    "[data-testid='product-grid']",
    "[data-testid='product-tile']",
    ".product-grid",
    ".product-tile",
    "[class*='ProductGrid']",
    "[class*='product-grid']",
    "[class*='ProductTile']",
    "[class*='product-tile']",
    "article",
    "li[class*='product']",
    "[class*='ProductCard']",
]

# --------------------------------------------------------------------------- #
# Attribute keyword maps                                                        #
# --------------------------------------------------------------------------- #

# For COLOR: checked against swatch aria-labels first, then product title.
# PacSun color names are usually plain ("Black", "White", "Washed Blue", etc.)
COLOR_KEYWORDS: list[tuple[str, str]] = [
    ("navy", "navy"),
    ("midnight", "black"),
    ("jet black", "black"),
    ("black", "black"),
    ("off white", "white"),
    ("off-white", "white"),
    ("white", "white"),
    ("cream", "white"),
    ("ivory", "white"),
    ("red", "red"),
    ("burgundy", "red"),
    ("maroon", "red"),
    ("wine", "red"),
    ("sage", "green"),
    ("olive", "green"),
    ("forest", "green"),
    ("army", "green"),
    ("green", "green"),
    ("washed blue", "blue"),
    ("medium wash", "blue"),
    ("light wash", "blue"),
    ("dark wash", "blue"),
    ("indigo", "blue"),
    ("denim", "blue"),
    ("sky blue", "blue"),
    ("cobalt", "blue"),
    ("blue", "blue"),
    ("light beige", "beige"),
    ("dark beige", "beige"),
    ("beige", "beige"),
    ("tan", "beige"),
    ("camel", "beige"),
    ("sand", "beige"),
    ("taupe", "beige"),
    ("khaki", "beige"),
    ("mocha", "brown"),
    ("chocolate", "brown"),
    ("coffee", "brown"),
    ("brown", "brown"),
    ("blush", "pink"),
    ("dusty pink", "pink"),
    ("hot pink", "pink"),
    ("mauve", "pink"),
    ("rose", "pink"),
    ("pink", "pink"),
    ("lavender", "purple"),
    ("lilac", "purple"),
    ("violet", "purple"),
    ("purple", "purple"),
    ("charcoal", "gray"),
    ("heather gray", "gray"),
    ("heather grey", "gray"),
    ("light gray", "gray"),
    ("dark gray", "gray"),
    ("grey", "gray"),
    ("gray", "gray"),
]

# For CATEGORY: checked against product title.
CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("baggy jean", "pants"),
    ("barrel jean", "pants"),
    ("straight jean", "pants"),
    ("slim jean", "pants"),
    ("skinny jean", "pants"),
    ("wide-leg jean", "pants"),
    ("wide leg jean", "pants"),
    ("flare jean", "pants"),
    ("bootcut", "pants"),
    ("jeans", "pants"),
    ("trouser", "pants"),
    ("chino", "pants"),
    ("legging", "pants"),
    ("jogger", "pants"),
    ("sweatpant", "pants"),
    ("cargo pant", "pants"),
    ("parachute", "pants"),
    ("pant", "pants"),
    ("shorts", "shorts"),
    ("short", "shorts"),
    ("mini skirt", "skirt"),
    ("midi skirt", "skirt"),
    ("maxi skirt", "skirt"),
    ("skirt", "skirt"),
    ("romper", "dress"),
    ("jumpsuit", "dress"),
    ("dress", "dress"),
    ("puffer", "outerwear"),
    ("parka", "outerwear"),
    ("jacket", "outerwear"),
    ("coat", "outerwear"),
    ("blazer", "outerwear"),
    ("cardigan", "outerwear"),
    ("vest", "outerwear"),
    ("hoodie", "tops"),
    ("sweatshirt", "tops"),
    ("sweater", "tops"),
    ("pullover", "tops"),
    ("flannel", "tops"),
    ("polo", "tops"),
    ("button-up", "tops"),
    ("button up", "tops"),
    ("shirt", "tops"),
    ("tee", "tops"),
    ("t-shirt", "tops"),
    ("crop", "tops"),
    ("blouse", "tops"),
    ("cami", "tops"),
    ("tank", "tops"),
    ("bodysuit", "tops"),
    ("corset", "tops"),
    ("top", "tops"),
    ("sneaker", "shoes"),
    ("boot", "shoes"),
    ("sandal", "shoes"),
    ("slide", "shoes"),
    ("flip flop", "shoes"),
    ("shoe", "shoes"),
    ("backpack", "accessories"),
    ("bag", "accessories"),
    ("purse", "accessories"),
    ("tote", "accessories"),
    ("belt", "accessories"),
    ("hat", "accessories"),
    ("beanie", "accessories"),
    ("cap", "accessories"),
    ("scarf", "accessories"),
    ("sock", "accessories"),
    ("jewelry", "accessories"),
    ("necklace", "accessories"),
    ("earring", "accessories"),
    ("sunglasses", "accessories"),
]

# For MATERIAL: checked against product title.
MATERIAL_KEYWORDS: list[tuple[str, str]] = [
    ("denim", "denim"),
    ("jean", "denim"),
    ("linen-blend", "linen"),
    ("linen", "linen"),
    ("silk", "silk"),
    ("satin", "silk"),
    ("cashmere", "wool"),
    ("wool", "wool"),
    ("fleece", "wool"),
    ("sherpa", "wool"),
    ("faux leather", "leather"),
    ("vegan leather", "leather"),
    ("pleather", "leather"),
    ("leather", "leather"),
    ("suede", "leather"),
    ("rib-knit", "knit"),
    ("ribbed", "knit"),
    ("knit", "knit"),
    ("crochet", "knit"),
    ("waffle", "knit"),
    ("mesh", "polyester"),
    ("nylon", "polyester"),
    ("polyester", "polyester"),
    ("recycled", "polyester"),
    ("poplin", "cotton"),
    ("twill", "cotton"),
    ("terry", "cotton"),
    ("jersey", "cotton"),
    ("cotton", "cotton"),
]

# When no material keyword is in the title, infer from category
CATEGORY_TO_MATERIAL_DEFAULT: dict[str, str] = {
    "pants": "denim",
    "shorts": "cotton",
    "dress": "cotton",
    "tops": "cotton",
    "outerwear": "polyester",
    "shoes": "leather",
    "accessories": "cotton",
    "skirt": "cotton",
}


# --------------------------------------------------------------------------- #
# Attribute extraction                                                          #
# --------------------------------------------------------------------------- #

def _first_match(text: str, keyword_map: list[tuple[str, str]]) -> str | None:
    lowered = text.lower()
    for keyword, mapped in keyword_map:
        if keyword in lowered:
            return mapped
    return None


def extract_color(text: str) -> str | None:
    return _first_match(text, COLOR_KEYWORDS)


def extract_category(text: str) -> str | None:
    return _first_match(text, CATEGORY_KEYWORDS)


def extract_material(text: str, inferred_category: str | None = None) -> str | None:
    result = _first_match(text, MATERIAL_KEYWORDS)
    if result:
        return result
    if inferred_category:
        return CATEGORY_TO_MATERIAL_DEFAULT.get(inferred_category)
    return None


# --------------------------------------------------------------------------- #
# Browser helpers                                                               #
# --------------------------------------------------------------------------- #

# The PerimeterX press-and-hold page contains very specific text/markup.
# We use these as signals that the challenge is currently up.
PX_CHALLENGE_TEXT_SIGNALS = [
    "press & hold",
    "press and hold",
    "verify you are a human",
    "verify you are human",
    "are you a human",
    "human verification",
    "checking your browser",
    "just a moment",
    "attention required",
]

PX_CHALLENGE_DOM_SELECTORS = [
    "#px-captcha",
    "[id^='px-captcha']",
    "iframe[src*='captcha.px-cdn.net']",
    "iframe[src*='perimeterx']",
    "div[class*='px-']",
]


def _is_on_challenge(page: "Page") -> bool:
    """Detect whether the current page is a PerimeterX press-and-hold challenge."""
    try:
        title = (page.title() or "").lower()
    except Exception:
        title = ""
    for signal in PX_CHALLENGE_TEXT_SIGNALS:
        if signal in title:
            return True
    for selector in PX_CHALLENGE_DOM_SELECTORS:
        try:
            if page.query_selector(selector):
                return True
        except Exception:
            continue
    try:
        body_text = page.evaluate("() => (document.body && document.body.innerText) || ''").lower()
    except Exception:
        body_text = ""
    return any(signal in body_text for signal in PX_CHALLENGE_TEXT_SIGNALS)


def _wait_out_bot_challenge(
    page: "Page",
    max_wait_secs: float = 20.0,
    manual_solve: bool = False,
) -> bool:
    """
    Detect a PerimeterX "Press & Hold" challenge.

    - If manual_solve is False: poll for up to max_wait_secs in case the
      challenge resolves itself (it usually won't — PerimeterX requires a
      real human mouse-down event).
    - If manual_solve is True: print instructions and wait up to 3 minutes
      for the user to solve the challenge by hand in the visible browser.
      Once solved, PerimeterX sets a `_px*` cookie that is reused on every
      subsequent navigation.

    Returns True once we're back on real PacSun content, False otherwise.
    """
    if not _is_on_challenge(page):
        return True

    if manual_solve:
        print(
            "    >>> PerimeterX press-and-hold challenge detected.\n"
            "    >>> In the visible browser window: press AND HOLD the\n"
            "    >>> button until it turns green, then release. The scraper\n"
            "    >>> will continue automatically once the real page loads."
        )
        deadline = time.time() + 180.0
        while time.time() < deadline:
            time.sleep(2.0)
            if not _is_on_challenge(page):
                print("    >>> challenge cleared, continuing.")
                # Give the real page a moment to render after the redirect
                time.sleep(2.0)
                return True
        print("    !!! manual challenge solve timed out after 3 minutes")
        return False

    deadline = time.time() + max_wait_secs
    while time.time() < deadline:
        time.sleep(1.5)
        if not _is_on_challenge(page):
            return True
    return False


def _wait_for_products(page: "Page", timeout_ms: int = 15_000) -> bool:
    """
    Wait until at least one product grid selector appears on the page.
    Returns True if found, False if all selectors timed out.
    """
    for selector in PRODUCT_GRID_WAIT_SELECTORS:
        try:
            page.wait_for_selector(selector, timeout=timeout_ms)
            return True
        except Exception:
            continue
    return False


def _scroll_to_bottom(page: "Page", pause_secs: float = 1.2, max_scrolls: int = 8) -> None:
    for _ in range(max_scrolls):
        page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
        time.sleep(pause_secs)


def _extract_product_names(page: "Page") -> list[str]:
    """Extract product title text using the first selector that returns results."""
    for selector in PRODUCT_NAME_SELECTORS:
        try:
            elements = page.query_selector_all(selector)
            texts = [el.inner_text().strip() for el in elements if el.inner_text().strip()]
            if texts:
                return texts
        except Exception:
            continue
    return []


def _extract_swatch_colors(page: "Page") -> list[str]:
    """
    Extract color names from swatch button aria-labels / title attributes.
    These are per-swatch labels like "Black", "Washed Blue", "Sage", etc.
    """
    for selector in COLOR_SWATCH_SELECTORS:
        try:
            elements = page.query_selector_all(selector)
            if not elements:
                continue
            labels: list[str] = []
            for el in elements:
                label = (
                    el.get_attribute("aria-label")
                    or el.get_attribute("title")
                    or el.get_attribute("data-color")
                    or el.get_attribute("data-color-name")
                    or el.get_attribute("data-attr-value")
                    or ""
                )
                if label.strip():
                    labels.append(label.strip())
            if labels:
                return labels
        except Exception:
            continue
    return []


# --------------------------------------------------------------------------- #
# Main scraping loop                                                            #
# --------------------------------------------------------------------------- #

# Stealth init script — masks the most obvious headless / automation tells
# that PerimeterX's fingerprinter inspects on every page load.
STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
window.chrome = { runtime: {}, loadTimes: function(){}, csi: function(){} };
Object.defineProperty(navigator, 'plugins', {
    get: () => [1, 2, 3, 4, 5].map(i => ({ name: 'Plugin ' + i }))
});
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'platform', { get: () => 'MacIntel' });
Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
const originalQuery = window.navigator.permissions && window.navigator.permissions.query;
if (originalQuery) {
    window.navigator.permissions.query = (params) => (
        params.name === 'notifications'
            ? Promise.resolve({ state: Notification.permission })
            : originalQuery(params)
    );
}
const getParameter = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(p) {
    if (p === 37445) return 'Intel Inc.';
    if (p === 37446) return 'Intel Iris OpenGL Engine';
    return getParameter.call(this, p);
};
"""


DEFAULT_PROFILE_DIR = (
    Path(__file__).resolve().parent / ".pacsun_browser_profile"
)


def scrape_pacsun(
    sleep_between_pages: float = 3.0,
    headless: bool = True,
    manual_solve: bool = False,
    profile_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    """
    Scrape all PacSun pages.

    Uses a PERSISTENT browser profile (so the PerimeterX `_px*` cookie sticks
    between runs). On the first run pass headless=False and manual_solve=True
    so you can press-and-hold the captcha by hand; later runs reuse the cookie.

    Returns (product_titles, swatch_color_labels).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright is not installed.")
        print("  Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    profile_dir = (profile_dir or DEFAULT_PROFILE_DIR).expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"  using browser profile: {profile_dir}")

    all_titles: list[str] = []
    all_swatch_colors: list[str] = []

    with sync_playwright() as pw:
        # launch_persistent_context = browser + context that share a real
        # on-disk user-data-dir, so cookies/localStorage survive between runs.
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/Los_Angeles",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            },
        )
        context.add_init_script(STEALTH_INIT_SCRIPT)

        page = context.pages[0] if context.pages else context.new_page()

        for idx, page_info in enumerate(PACSUN_PAGES):
            url = page_info["url"]
            label = page_info["label"]
            print(f"  [{label}] → {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45_000)

                # Step 1: handle the PerimeterX press-and-hold if it shows up.
                # Only the FIRST navigation usually needs a manual solve —
                # after that the _px* cookie carries us through.
                challenge_passed = _wait_out_bot_challenge(
                    page,
                    max_wait_secs=20.0,
                    manual_solve=manual_solve,
                )
                if not challenge_passed:
                    print(
                        f"    WARNING: bot challenge did not resolve — skipping page.\n"
                        f"             Re-run with --headless false --manual-solve\n"
                        f"             to clear it once and cache the cookie."
                    )
                    continue

                # Step 2: wait for the product grid to actually render
                grid_found = _wait_for_products(page, timeout_ms=15_000)
                if not grid_found:
                    page_title = page.title()
                    print(f"    WARNING: product grid not found (page title: '{page_title}')")

                time.sleep(1.5)
                _scroll_to_bottom(page)

                titles = _extract_product_names(page)
                swatches = _extract_swatch_colors(page)

                print(f"    {len(titles)} product titles, {len(swatches)} swatch colors")
                all_titles.extend(titles)
                all_swatch_colors.extend(swatches)

            except Exception as exc:
                print(f"    ERROR: {exc}")

            time.sleep(sleep_between_pages)

        context.close()

    return all_titles, all_swatch_colors


# --------------------------------------------------------------------------- #
# Frequency counting and normalization                                          #
# --------------------------------------------------------------------------- #

def count_attribute_frequencies(
    titles: list[str],
    swatch_colors: list[str],
) -> dict[str, dict[str, int]]:
    """
    Count occurrences of each feature value across all products.

    Color is sourced from swatch_colors first (more reliable), then title fallback.
    Category and material come from title keyword matching.
    """
    counts: dict[str, dict[str, int]] = {ft: {} for ft in FEATURE_TYPES}

    # Colors from swatches — each swatch label is one occurrence of that color
    for swatch_label in swatch_colors:
        color = extract_color(swatch_label)
        if color:
            counts["color"][color] = counts["color"].get(color, 0) + 1

    # Category, material, and fallback color from product titles
    for title in titles:
        category = extract_category(title)
        if category:
            counts["category"][category] = counts["category"].get(category, 0) + 1

        material = extract_material(title, inferred_category=category)
        if material:
            counts["material"][material] = counts["material"].get(material, 0) + 1

        # Color from title only if swatch extraction was empty
        if not swatch_colors:
            color = extract_color(title)
            if color:
                counts["color"][color] = counts["color"].get(color, 0) + 1

    return counts


def normalize_counts(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for feature_type, value_counts in counts.items():
        if not value_counts:
            scores[feature_type] = {}
            continue
        max_count = max(value_counts.values())
        scores[feature_type] = {
            value: round(count / max_count, 6)
            for value, count in value_counts.items()
        }
    return scores


def build_trend_signals_frame(
    scores: dict[str, dict[str, float]],
    known_feature_values: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    for feature_type, values in known_feature_values.items():
        type_scores = scores.get(feature_type, {})
        for feature_value in values:
            rows.append({
                "feature_type": feature_type,
                "feature_value": feature_value,
                "current": type_scores.get(feature_value, DEFAULT_MISSING_SCORE),
            })
    return pd.DataFrame(rows)


def blend_with_existing(
    scraped: pd.DataFrame,
    existing_path: Path,
    blend_weight: float,
) -> pd.DataFrame:
    existing = pd.read_csv(existing_path)
    existing_validated = validate_trend_signals_frame(existing)
    existing_map = {
        (row["feature_type"], row["feature_value"]): row["current"]
        for _, row in existing_validated.iterrows()
    }
    blended = scraped.copy()
    for idx, row in blended.iterrows():
        key = (row["feature_type"], row["feature_value"])
        existing_score = existing_map.get(key, DEFAULT_MISSING_SCORE)
        blended.at[idx, "current"] = round(
            blend_weight * float(row["current"]) + (1.0 - blend_weight) * existing_score,
            6,
        )
    return blended


# --------------------------------------------------------------------------- #
# Argument parsing                                                              #
# --------------------------------------------------------------------------- #

KNOWN_FEATURE_VALUES: dict[str, list[str]] = {
    "color":    ["black", "white", "blue", "red", "green", "beige", "pink", "gray", "navy", "brown", "purple"],
    "category": ["pants", "shorts", "skirt", "dress", "tops", "outerwear", "shoes", "accessories"],
    "material": ["cotton", "denim", "linen", "silk", "wool", "polyester", "leather", "knit"],
}


def parse_args() -> argparse.Namespace:
    default_output = (
        Path(__file__).resolve().parents[1]
        / "training" / "synthetic_data" / "trend_signals.csv"
    )
    parser = argparse.ArgumentParser(
        description="Scrape PacSun new arrivals and write trend_signals.csv."
    )
    parser.add_argument("--output-path", default=str(default_output))
    parser.add_argument(
        "--existing-path", default=None,
        help="Existing trend_signals.csv to blend with scraped scores.",
    )
    parser.add_argument(
        "--blend-weight", type=float, default=0.5,
        help="Weight for scraped scores when blending (default 0.5).",
    )
    parser.add_argument(
        "--sleep", type=float, default=3.0,
        help="Seconds between page loads (default 3.0).",
    )
    parser.add_argument(
        "--headless", type=lambda v: v.lower() != "false", default=True,
        help="Run headless browser. Pass 'false' for a visible window (default: true).",
    )
    parser.add_argument(
        "--manual-solve", action="store_true",
        help=(
            "Pause when PerimeterX press-and-hold appears so you can solve "
            "it by hand. Use the FIRST time together with --headless false; "
            "the cookie is then cached in --profile-dir for future runs."
        ),
    )
    parser.add_argument(
        "--profile-dir", default=None,
        help=(
            "Directory used as the persistent Chromium profile (default: "
            "trndly/pipelines/collectors/.pacsun_browser_profile). Caching "
            "the PerimeterX cookie here lets later runs skip the captcha."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"PacSun retail scraper\n"
        f"  pages: {len(PACSUN_PAGES)}  headless: {args.headless}  "
        f"manual_solve: {args.manual_solve}\n"
        f"  output: {output_path}"
    )

    titles, swatch_colors = scrape_pacsun(
        sleep_between_pages=args.sleep,
        headless=args.headless,
        manual_solve=args.manual_solve,
        profile_dir=Path(args.profile_dir) if args.profile_dir else None,
    )

    print(f"\nTotal collected: {len(titles)} product titles, {len(swatch_colors)} swatch colors")

    if not titles and not swatch_colors:
        print(
            "\nWARNING: Nothing was scraped. PacSun uses PerimeterX press-and-hold\n"
            "bot protection which cannot be cleared by headless Chrome alone.\n"
            "Run once with:\n"
            "    python pacsun_scraper.py --headless false --manual-solve\n"
            "and press-and-hold the captcha in the visible window. The cookie\n"
            "is cached in the persistent profile, so future runs (even headless)\n"
            "will reuse it for as long as PerimeterX accepts the cookie."
        )

    counts = count_attribute_frequencies(titles, swatch_colors)
    scores = normalize_counts(counts)

    print("\nAttribute coverage:")
    for feature_type in FEATURE_TYPES:
        found = len(counts.get(feature_type, {}))
        total = len(KNOWN_FEATURE_VALUES[feature_type])
        print(f"  {feature_type}: {found}/{total} values seen")
        for value, score in sorted(scores.get(feature_type, {}).items(), key=lambda x: -x[1]):
            count = counts[feature_type].get(value, 0)
            print(f"    {value:<15} score={score:.3f}  (count={count})")

    frame = build_trend_signals_frame(scores=scores, known_feature_values=KNOWN_FEATURE_VALUES)

    if args.existing_path and Path(args.existing_path).exists():
        frame = blend_with_existing(
            scraped=frame,
            existing_path=Path(args.existing_path),
            blend_weight=args.blend_weight,
        )
        print(f"\nBlended with existing: {args.existing_path}")

    validated = validate_trend_signals_frame(frame)
    validated.to_csv(output_path, index=False)
    print(f"\nWrote {len(validated)} rows → {output_path}")


if __name__ == "__main__":
    main()
