"""
Hollister retail scraper for trndly trend signals.

Scrapes Hollister's "New Arrivals" and category pages using a real browser
(Playwright) to count how often each color, category, and material attribute
appears across featured product listings. Normalizes those counts to 0–1 and
writes the result as trend_signals.csv.

WHERE EACH ATTRIBUTE COMES FROM
--------------------------------
- category : product title keywords  ("jeans" → pants, "hoodie" → tops, etc.)
- material  : product title keywords  ("linen", "denim", "knit", etc.)
             + category inference when no material keyword is in the title
- color     : color swatch aria-labels  (Hollister puts the color name in the
             aria-label of each swatch button, NOT in the product title text)
             + product title keywords as a fallback

ZERO TITLES ON SUBSEQUENT PAGES (what was wrong before)
---------------------------------------------------------
Hollister is a React SPA. After navigating to a new URL the product grid
re-renders asynchronously. The previous version checked for titles immediately
after networkidle, which fired before React finished rendering the grid.
The fix: use page.wait_for_selector() to block until at least one product
tile is visible, then extract.

AKAMAI BOT MANAGER
-------------------
Hollister uses Akamai Bot Manager. Headless Chrome with default settings is
fingerprinted and served a "Client Challenge" page instead of real content.
This scraper adds several de-fingerprinting steps, but if you hit consistent
empty pages, running it through a real (non-headless) browser or a residential
proxy may be needed. Use --headless false to open a visible browser window.

SETUP (one-time)
----------------
  pip install playwright
  playwright install chromium

Usage:
  python hollister_scraper.py
  python hollister_scraper.py --output-path path/to/trend_signals.csv
  python hollister_scraper.py --existing-path trend_signals.csv --blend-weight 0.5
  python hollister_scraper.py --headless false   # visible browser, harder to detect
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

HOLLISTER_PAGES = [
    {"url": "https://www.hollisterco.com/shop/us/womens-new-arrivals",       "label": "women new arrivals"},
    {"url": "https://www.hollisterco.com/shop/us/mens-new-arrivals",        "label": "men new arrivals"},
    {"url": "https://www.hollisterco.com/shop/us/womens-tops",               "label": "women tops"},
    {"url": "https://www.hollisterco.com/shop/us/mens-tops",                "label": "men tops"},
    {"url": "https://www.hollisterco.com/shop/us/womens-bottoms",            "label": "women bottoms"},
    {"url": "https://www.hollisterco.com/shop/us/mens-bottoms",             "label": "men bottoms"},
    {"url": "https://www.hollisterco.com/shop/us/womens-dresses-and-rompers","label": "women dresses"},
    {"url": "https://www.hollisterco.com/shop/us/mens-jackets-and-coats",   "label": "men outerwear"},
    {"url": "https://www.hollisterco.com/shop/us/womens-jackets-and-coats",  "label": "women outerwear"},
]

# --------------------------------------------------------------------------- #
# Selectors                                                                     #
#                                                                               #
# PRODUCT NAME selectors — tried in order, first non-empty match wins.         #
# Hollister uses React with generated class names; these cover known patterns. #
# --------------------------------------------------------------------------- #

PRODUCT_NAME_SELECTORS = [
    "[data-testid='product-name']",
    "[data-testid='product-card-title']",
    "[class*='ProductName']",
    "[class*='product-name']",
    "[class*='productName']",
    "[class*='ProductTitle']",
    "[class*='product-title']",
    "[class*='ProductCard'] h2",
    "[class*='ProductCard'] h3",
    "[class*='product-card'] h2",
    "[class*='product-card'] h3",
    "article h2",
    "article h3",
    "li[class*='product'] h2",
    "li[class*='product'] h3",
]

# COLOR SWATCH selectors — Hollister attaches the color name to swatch buttons
# via aria-label (e.g. aria-label="Rinse Black") rather than putting it in the
# product title. We try these to get per-product color data.
#
# Hollister has shipped multiple swatch implementations over time:
#   - <button aria-label="Rinse Black"> with class names like *Swatch*
#   - <img alt="Rinse Black" class="...swatch..."> for image-based swatches
#   - <a title="Rinse Black"> on the older PWA
#   - data-* attributes on the parent element
# These selectors are tried in order; the first non-empty match wins.
COLOR_SWATCH_SELECTORS = [
    # Hollister 2025 markup: swatches are labeled by hidden <label> elements
    # whose INNER TEXT is "<Color> swatch" (e.g. "Brown swatch", "Washed
    # Black swatch"). Extractor reads innerText from these.
    "[data-testid='catalog-product-card-swatch-tile'] label.screen-reader-text",
    "[data-testid='catalog-product-card-swatch-tile'] label",
    "label.screen-reader-text[for*='__swatch']",
    "label.screen-reader-text[for*='swatch']",
    # Modern testids
    "[data-testid='color-swatch'] [aria-label]",
    "[data-testid*='swatch'] [aria-label]",
    "[data-test-id*='swatch'] [aria-label]",
    "[data-testid*='color'] [aria-label]",
    # React/styled-component class patterns
    "[class*='ColorSwatch'] [aria-label]",
    "[class*='color-swatch'] [aria-label]",
    "[class*='colorSwatch'] [aria-label]",
    "[class*='Swatch'] button[aria-label]",
    "[class*='swatch'] button[aria-label]",
    "[class*='Swatch'] a[aria-label]",
    "[class*='swatch'] a[aria-label]",
    # Image-based swatches: the color name lives in alt= on the <img>
    "[class*='Swatch'] img[alt]",
    "[class*='swatch'] img[alt]",
    "img[class*='swatch'][alt]",
    "img[class*='Swatch'][alt]",
    # Title-attribute fallback (older PWA / accessibility tooltips)
    "[class*='Swatch'] [title]",
    "[class*='swatch'] [title]",
    "a[class*='swatch'][title]",
    # Generic aria-label probes
    "button[aria-label][class*='color']",
    "button[aria-label][class*='Color']",
    "button[aria-label][class*='swatch']",
    "button[aria-label][class*='Swatch']",
    # Direct data-* attribute reads
    "[data-color]",
    "[data-color-name]",
    "[data-swatch-name]",
    "[data-attr-color]",
    "[data-variation-color]",
]

# Selector to wait for before extracting — signals the product grid is ready
PRODUCT_GRID_WAIT_SELECTORS = [
    "[data-testid='product-grid']",
    "[class*='ProductGrid']",
    "[class*='product-grid']",
    "article",
    "li[class*='product']",
    "[class*='ProductCard']",
]

# --------------------------------------------------------------------------- #
# Attribute keyword maps                                                        #
# --------------------------------------------------------------------------- #

# For COLOR: checked against swatch aria-labels first, then product title.
# Hollister color names use brand words like "Rinse Black", "Cloud White", etc.
COLOR_KEYWORDS: list[tuple[str, str]] = [
    # Hollister-specific brand names
    ("rinse black", "black"),
    ("washed black", "black"),
    ("jet black", "black"),
    ("cloud white", "white"),
    ("optic white", "white"),
    ("indigo wash", "blue"),
    ("grey wash", "gray"),
    # Shade-qualified generics (color_spectrum from lookup: Dark, Dusty Light, Light, Medium, Bright)
    ("dusty light blue", "blue"),
    ("medium dusty blue", "blue"),
    ("light blue", "blue"),
    ("medium blue", "blue"),
    ("dark blue", "blue"),
    ("bright blue", "blue"),
    ("light green", "green"),
    ("dark green", "green"),
    ("bright green", "green"),
    ("dusty green", "green"),
    ("light pink", "pink"),
    ("bright pink", "pink"),
    ("hot pink", "pink"),
    ("dark pink", "pink"),
    ("light red", "red"),
    ("bright red", "red"),
    ("dark red", "red"),
    ("light brown", "brown"),
    ("dark brown", "brown"),
    ("light purple", "purple"),
    ("dark purple", "purple"),
    ("mustard", "beige"),
    ("butter", "beige"),
    ("golden", "beige"),
    ("terracotta", "red"),
    ("rust", "red"),
    ("brick", "red"),
    # Standard generics
    ("navy", "navy"),
    ("black", "black"),
    ("white", "white"),
    ("cream", "white"),
    ("ivory", "white"),
    ("off white", "white"),
    ("red", "red"),
    ("burgundy", "red"),
    ("maroon", "red"),
    ("wine", "red"),
    ("coral", "pink"),
    ("peach", "pink"),
    ("sage", "green"),
    ("olive", "green"),
    ("forest", "green"),
    ("moss", "green"),
    ("khaki green", "green"),
    ("green", "green"),
    ("medium wash", "blue"),
    ("light wash", "blue"),
    ("dark wash", "blue"),
    ("indigo", "blue"),
    ("teal", "blue"),
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
    ("ecru", "beige"),
    ("khaki", "beige"),
    ("amber", "brown"),
    ("mocha", "brown"),
    ("chocolate", "brown"),
    ("cognac", "brown"),
    ("espresso", "brown"),
    ("brown", "brown"),
    ("blush", "pink"),
    ("dusty pink", "pink"),
    ("mauve", "pink"),
    ("rose", "pink"),
    ("pink", "pink"),
    ("lavender", "purple"),
    ("lilac", "purple"),
    ("plum", "purple"),
    ("violet", "purple"),
    ("purple", "purple"),
    ("charcoal", "gray"),
    ("heather gray", "gray"),
    ("heather grey", "gray"),
    ("slate", "gray"),
    ("stone", "gray"),
    ("light gray", "gray"),
    ("dark gray", "gray"),
    ("grey", "gray"),
    ("gray", "gray"),
]

# For CATEGORY: checked against product title.
CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    # Pants / bottoms
    ("barrel jean", "pants"),
    ("straight jean", "pants"),
    ("slim jean", "pants"),
    ("wide-leg jean", "pants"),
    ("jeans", "pants"),
    ("trouser", "pants"),
    ("chino", "pants"),
    ("legging", "pants"),
    ("jogger", "pants"),
    ("sweatpant", "pants"),
    ("dungarees", "pants"),
    ("overalls", "pants"),
    ("pant", "pants"),
    # Shorts
    ("shorts", "shorts"),
    # Skirt
    ("sarong", "skirt"),
    ("skirt", "skirt"),
    # Dress / full body
    ("playsuit", "dress"),
    ("romper", "dress"),
    ("jumpsuit", "dress"),
    ("bodysuit", "dress"),
    ("body suit", "dress"),
    ("dress", "dress"),
    # Outerwear
    ("anorak", "outerwear"),
    ("windbreaker", "outerwear"),
    ("gilet", "outerwear"),
    ("waistcoat", "outerwear"),
    ("jacket", "outerwear"),
    ("coat", "outerwear"),
    ("parka", "outerwear"),
    ("puffer", "outerwear"),
    ("blazer", "outerwear"),
    ("cardigan", "outerwear"),
    # Tops
    ("polo", "tops"),
    ("corset", "tops"),
    ("hoodie", "tops"),
    ("sweatshirt", "tops"),
    ("sweater", "tops"),
    ("pullover", "tops"),
    ("flannel", "tops"),
    ("shirt", "tops"),
    ("tee", "tops"),
    ("t-shirt", "tops"),
    ("crop", "tops"),
    ("blouse", "tops"),
    ("cami", "tops"),
    ("tank", "tops"),
    ("vest", "tops"),
    ("top", "tops"),
    # Shoes
    ("pump", "shoes"),
    ("heel", "shoes"),
    ("loafer", "shoes"),
    ("mule", "shoes"),
    ("clog", "shoes"),
    ("ballerina", "shoes"),
    ("slipper", "shoes"),
    ("flip flop", "shoes"),
    ("wedge", "shoes"),
    ("sneaker", "shoes"),
    ("boot", "shoes"),
    ("sandal", "shoes"),
    ("shoe", "shoes"),
    # Accessories
    ("sunglasses", "accessories"),
    ("glasses", "accessories"),
    ("watch", "accessories"),
    ("wallet", "accessories"),
    ("bracelet", "accessories"),
    ("necklace", "accessories"),
    ("earring", "accessories"),
    ("ring", "accessories"),
    ("gloves", "accessories"),
    ("bag", "accessories"),
    ("belt", "accessories"),
    ("hat", "accessories"),
    ("beanie", "accessories"),
    ("scarf", "accessories"),
    ("sock", "accessories"),
]

# For MATERIAL: checked against product title.
# Sourced from lookup.csv material list + common retail keyword variants.
MATERIAL_KEYWORDS: list[tuple[str, str]] = [
    # Denim
    ("denim", "denim"),
    ("jean", "denim"),
    # Linen
    ("linen-blend", "linen"),
    ("linen", "linen"),
    # Silk / silk-like
    ("chiffon", "silk"),
    ("crepe", "silk"),
    ("georgette", "silk"),
    ("silk", "silk"),
    ("satin", "silk"),
    # Wool / wool-like
    ("cashmere", "wool"),
    ("shearling", "wool"),
    ("sherpa", "wool"),
    ("faux fur", "wool"),
    ("wool", "wool"),
    ("fleece", "wool"),
    # Leather / leather-like
    ("imitation leather", "leather"),
    ("imitation suede", "leather"),
    ("faux leather", "leather"),
    ("vegan leather", "leather"),
    ("suede", "leather"),
    ("leather", "leather"),
    # Knit / knit-like
    ("rib-knit", "knit"),
    ("ribbed", "knit"),
    ("jersey", "knit"),
    ("velvet", "knit"),
    ("velour", "knit"),
    ("knit", "knit"),
    ("crochet", "knit"),
    ("waffle", "knit"),
    # Polyester / synthetics
    ("nylon", "polyester"),
    ("acrylic", "polyester"),
    ("tulle", "polyester"),
    ("mesh", "polyester"),
    ("spandex", "polyester"),
    ("elastane", "polyester"),
    ("polyester", "polyester"),
    ("recycled", "polyester"),
    # Cotton / cellulosics
    ("poplin", "cotton"),
    ("twill", "cotton"),
    ("terry", "cotton"),
    ("corduroy", "cotton"),
    ("canvas", "cotton"),
    ("tencel", "cotton"),
    ("lyocell", "cotton"),
    ("modal", "cotton"),
    ("viscose", "cotton"),
    ("rayon", "cotton"),
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

def _wait_out_akamai_challenge(page: "Page", max_wait_secs: float = 20.0) -> bool:
    """
    Hollister uses Akamai Bot Manager which serves a "Client Challenge" page
    before showing real content. The challenge runs JavaScript, sets a cookie,
    then redirects back to the original URL.

    This function detects the challenge by checking the page title and waits
    for the redirect to complete. Returns True once real content is available,
    False if the challenge never resolved within max_wait_secs.
    """
    deadline = time.time() + max_wait_secs
    while time.time() < deadline:
        title = page.title().lower()
        url = page.url
        if "challenge" in title or "checking" in title or "just a moment" in title:
            time.sleep(1.5)
            continue
        # If we're on the right URL with no challenge in the title, we passed
        if "hollisterco.com" in url:
            return True
        time.sleep(1.0)
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


def _clean_swatch_label(raw: str) -> str:
    """
    Normalize a raw swatch label like "Brown swatch" / "Washed Black swatch"
    down to just the color name ("Brown" / "Washed Black"). Hollister's
    screen-reader labels always end with " swatch".
    """
    cleaned = raw.strip()
    # Strip Hollister's trailing " swatch" suffix (case-insensitive)
    lowered = cleaned.lower()
    if lowered.endswith(" swatch"):
        cleaned = cleaned[: -len(" swatch")].strip()
    elif lowered.endswith("swatch"):
        cleaned = cleaned[: -len("swatch")].strip()
    return cleaned


def _extract_swatch_colors(page: "Page") -> list[str]:
    """
    Extract color names from swatch elements.

    Tries every selector in COLOR_SWATCH_SELECTORS and, for each element,
    reads whichever of the following is first non-empty:
      1. inner text  (Hollister 2025: "<Color> swatch")
      2. aria-label / alt / title
      3. data-* color attributes
    Returns the FIRST selector's non-empty hit so we don't double-count
    when multiple selectors target the same elements.
    """
    color_attrs = (
        "aria-label", "alt", "title",
        "data-color", "data-color-name", "data-swatch-name",
        "data-attr-color", "data-variation-color", "data-name",
    )
    for selector in COLOR_SWATCH_SELECTORS:
        try:
            elements = page.query_selector_all(selector)
            if not elements:
                continue
            labels: list[str] = []
            for el in elements:
                value = ""
                # inner text first (Hollister's <label class="screen-reader-text">)
                try:
                    text = el.inner_text().strip()
                except Exception:
                    text = ""
                if text:
                    value = text
                else:
                    for attr in color_attrs:
                        attr_value = el.get_attribute(attr)
                        if attr_value and attr_value.strip():
                            value = attr_value.strip()
                            break
                if value:
                    labels.append(_clean_swatch_label(value))
            # Drop anything that cleaned to an empty string
            labels = [label for label in labels if label]
            if labels:
                return labels
        except Exception:
            continue
    return []


def _extract_colors_from_jsonld(page: "Page") -> list[str]:
    """
    Many e-commerce pages embed product data in <script type='application/ld+json'>
    using schema.org/Product. The `color` field (or `name` on each variant) often
    holds the human-readable color even when the DOM doesn't. This is a reliable
    fallback when swatch selectors miss.
    """
    try:
        return page.evaluate(
            """
            () => {
                const out = [];
                const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                for (const s of scripts) {
                    try {
                        const data = JSON.parse(s.textContent);
                        const items = Array.isArray(data) ? data : [data];
                        for (const item of items) {
                            const stack = [item];
                            while (stack.length) {
                                const node = stack.pop();
                                if (!node || typeof node !== 'object') continue;
                                if (typeof node.color === 'string') out.push(node.color);
                                for (const k of Object.keys(node)) {
                                    const v = node[k];
                                    if (Array.isArray(v)) v.forEach(x => stack.push(x));
                                    else if (v && typeof v === 'object') stack.push(v);
                                }
                            }
                        }
                    } catch (e) {}
                }
                return out;
            }
            """
        ) or []
    except Exception:
        return []


def _dump_first_tile_html(page: "Page", out_path: Path) -> None:
    """
    Debug helper: write the outerHTML of the first product tile we can find
    so you can manually inspect what swatch markup actually exists today.
    Only used when --debug is passed.
    """
    snippet = ""
    for selector in PRODUCT_GRID_WAIT_SELECTORS:
        try:
            el = page.query_selector(selector)
            if el:
                snippet = el.evaluate("el => el.outerHTML") or ""
                if snippet:
                    break
        except Exception:
            continue
    if snippet:
        out_path.write_text(snippet, encoding="utf-8")
        print(f"    [debug] dumped {len(snippet):,} chars of tile HTML → {out_path}")


# --------------------------------------------------------------------------- #
# Main scraping loop                                                            #
# --------------------------------------------------------------------------- #

def scrape_hollister(
    sleep_between_pages: float = 3.0,
    headless: bool = True,
    debug: bool = False,
    debug_dir: Path | None = None,
) -> tuple[list[str], list[str]]:
    """
    Scrape all Hollister pages.

    If debug=True, on the first page where we find product titles but ZERO
    swatch colors we dump the first product tile's HTML so you can inspect
    Hollister's current swatch markup and add new selectors.

    Returns (product_titles, swatch_color_labels).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright is not installed.")
        print("  Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    all_titles: list[str] = []
    all_swatch_colors: list[str] = []
    debug_dumped = False
    if debug:
        debug_dir = (debug_dir or Path(__file__).resolve().parent / ".hollister_debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [debug] tile dumps will be written to: {debug_dir}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            },
        )
        # Mask the webdriver flag that Akamai checks
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            "window.chrome = {runtime: {}};"
        )

        page = context.new_page()

        for page_info in HOLLISTER_PAGES:
            url = page_info["url"]
            label = page_info["label"]
            print(f"  [{label}] → {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                # Step 1: wait out the Akamai "Client Challenge" page.
                # Hollister serves a bot-detection challenge on every navigation.
                # The challenge JS runs, sets a cookie, and redirects back.
                challenge_passed = _wait_out_akamai_challenge(page, max_wait_secs=20.0)
                if not challenge_passed:
                    print(f"    WARNING: Akamai challenge did not resolve — skipping page")
                    continue

                # Step 2: wait for the product grid to actually render in React
                grid_found = _wait_for_products(page, timeout_ms=15_000)
                if not grid_found:
                    page_title = page.title()
                    print(f"    WARNING: product grid not found (page title: '{page_title}')")

                time.sleep(1.5)
                _scroll_to_bottom(page)

                titles = _extract_product_names(page)
                swatches = _extract_swatch_colors(page)

                # Fallback: schema.org/Product JSON-LD often contains color
                # data even when the DOM swatch markup misses.
                jsonld_source = ""
                if not swatches:
                    jsonld_colors = _extract_colors_from_jsonld(page)
                    if jsonld_colors:
                        swatches = jsonld_colors
                        jsonld_source = " (via JSON-LD)"

                print(
                    f"    {len(titles)} product titles, "
                    f"{len(swatches)} swatch colors{jsonld_source}"
                )
                all_titles.extend(titles)
                all_swatch_colors.extend(swatches)

                # Debug dump: capture first product tile when we got titles
                # but no swatches, so the user can see the actual markup.
                if debug and not debug_dumped and titles and not swatches:
                    safe_label = label.replace(" ", "_")
                    _dump_first_tile_html(
                        page, debug_dir / f"tile_{safe_label}.html",
                    )
                    debug_dumped = True

            except Exception as exc:
                print(f"    ERROR: {exc}")

            time.sleep(sleep_between_pages)

        browser.close()

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


def normalize_counts(
    counts: dict[str, dict[str, int]],
    total_items: int,
) -> dict[str, dict[str, float]]:
    """
    Normalize raw feature counts to proportion scores.

    score = count / total_items  (actual market-share proportion)

    Using total items scraped as the denominator (instead of the per-feature
    max count) makes scores directly comparable across retailers and over time:
    a score of 0.30 always means "30% of products on this site had this value".
    """
    denom = max(total_items, 1)
    scores: dict[str, dict[str, float]] = {}
    for feature_type, value_counts in counts.items():
        scores[feature_type] = {
            value: round(count / denom, 6)
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
    # Each scraper writes to its own per-retailer file so multiple retailers
    # can be run independently, updated on their own schedule, and then
    # combined later by combine_trend_signals.py.
    default_output = (
        Path(__file__).resolve().parents[1]
        / "training" / "synthetic_data" / "trend_signals_hollister.csv"
    )
    parser = argparse.ArgumentParser(
        description="Scrape Hollister new arrivals and write trend_signals_hollister.csv."
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
        "--debug", action="store_true",
        help=(
            "When set, dumps the first product tile's HTML on the first page "
            "where titles are found but swatch colors are not. Useful for "
            "discovering Hollister's current swatch markup so new selectors "
            "can be added to COLOR_SWATCH_SELECTORS."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Hollister retail scraper\n"
        f"  pages: {len(HOLLISTER_PAGES)}  headless: {args.headless}\n"
        f"  output: {output_path}"
    )

    titles, swatch_colors = scrape_hollister(
        sleep_between_pages=args.sleep,
        headless=args.headless,
        debug=args.debug,
    )

    print(f"\nTotal collected: {len(titles)} product titles, {len(swatch_colors)} swatch colors")

    if not titles and not swatch_colors:
        print(
            "\nWARNING: Nothing was scraped. Hollister's Akamai bot protection\n"
            "may be blocking headless Chrome. Try running with --headless false\n"
            "to open a visible browser window, which is harder to detect."
        )

    counts = count_attribute_frequencies(titles, swatch_colors)
    scores = normalize_counts(counts, total_items=len(titles))

    print(f"\nTotal items used as proportion denominator: {len(titles)}")
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
