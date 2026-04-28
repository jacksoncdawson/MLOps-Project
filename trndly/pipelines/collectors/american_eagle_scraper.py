"""
American Eagle retail scraper for trndly trend signals.

Scrapes American Eagle's "New Arrivals" and category pages using a real
browser (Playwright) to count how often each color, category, and material
attribute appears across featured product listings. Normalizes those counts
to 0–1 and writes the result as trend_signals_american_eagle.csv.

WHERE EACH ATTRIBUTE COMES FROM
--------------------------------
- category : product title keywords  ("jeans" → pants, "hoodie" → tops, etc.)
- material  : product title keywords  ("linen", "denim", "knit", etc.)
             + category inference when no material keyword is in the title
- color     : color swatch aria-labels  (American Eagle puts the color name
             in the aria-label / data-color of each swatch button, NOT in
             the product title text)
             + product title keywords as a fallback

ZERO TITLES ON SUBSEQUENT PAGES
--------------------------------
ae.com is a Next.js SPA. After navigating to a new URL the product grid
re-renders asynchronously, so the scraper calls page.wait_for_selector() to
block until at least one product tile is visible before extracting.

BOT PROTECTION
--------------
ae.com does not currently serve the kind of interstitial bot challenge that
Hollister does, so this scraper skips the Akamai-style waiter. Some
de-fingerprinting init scripts are still applied to reduce the chance of
headless Chrome being served an empty / simplified page. If you hit
consistent empty pages, try --headless false to open a visible browser.

SETUP (one-time)
----------------
  pip install playwright
  playwright install chromium

Usage:
  python american_eagle_scraper.py
  python american_eagle_scraper.py --output-path path/to/trend_signals_american_eagle.csv
  python american_eagle_scraper.py --existing-path trend_signals_american_eagle.csv --blend-weight 0.5
  python american_eagle_scraper.py --headless false   # visible browser
"""

from __future__ import annotations

import argparse
import json
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
#                                                                               #
# ae.com occasionally renames category paths. The plain paths below redirect   #
# correctly in a browser; if a page returns 0 titles, update the URL here.     #
# --------------------------------------------------------------------------- #

AMERICAN_EAGLE_PAGES = [
    {"url": "https://www.ae.com/us/en/c/women/new-arrivals/brg_dyn_hqya6u718b",    "label": "women new arrivals"},
    {"url": "https://www.ae.com/us/en/c/men/new-arrivals/brg_dyn_fiqvft6w17",      "label": "men new arrivals"},
    {"url": "https://www.ae.com/us/en/c/women/tops/cat10049",                      "label": "women tops"},
    {"url": "https://www.ae.com/us/en/c/men/tops/cat10025",                        "label": "men tops"},
    {"url": "https://www.ae.com/us/en/c/women/bottoms/cat10051",                   "label": "women bottoms"},
    {"url": "https://www.ae.com/us/en/c/men/bottoms/cat10027",                     "label": "men bottoms"},
    {"url": "https://www.ae.com/us/en/c/women/dresses/cat1320034",                 "label": "women dresses"},
    {"url": "https://www.ae.com/us/en/c/women/tops/jackets/cat4260032",            "label": "women outerwear"},
    {"url": "https://www.ae.com/us/en/c/men/tops/jackets/cat380145",               "label": "men outerwear"},
]

# --------------------------------------------------------------------------- #
# Selectors                                                                     #
#                                                                               #
# PRODUCT NAME selectors — tried in order, first non-empty match wins.         #
# AE's product tiles expose `data-testid="product-tile"` with a nested title   #
# element; we fall back to more generic selectors if that changes.             #
# --------------------------------------------------------------------------- #

PRODUCT_NAME_SELECTORS = [
    # Confirmed from live inspection:
    # <h3 class="product-name ..." data-product-name="AE Double Take Tube Top" data-testid="name">
    "[data-testid='name']",
    "h3[data-testid='name']",
    "[class*='product-name'][data-testid='name']",
    "[data-product-name]",
    # Generic fallbacks
    "[class*='product-name']",
    "[class*='ProductName']",
    "[class*='ProductCard'] h3",
    "[class*='product-card'] h3",
    "article h2",
    "article h3",
    "li[class*='product'] h2",
    "li[class*='product'] h3",
]

# COLOR SWATCH selectors — AE shows a small row of color swatches on each
# tile; the color name is exposed as aria-label or data-color on the swatch.
COLOR_SWATCH_SELECTORS = [
    # Confirmed from live inspection:
    # <img class="_swatch-img_..." alt="Gatsby Green" title="Gatsby Green">
    # Color name lives in the `alt` (and `title`) attribute of the swatch image.
    "img[class*='swatch-img']",
    "img[class*='swatch_img']",
    "[data-test-color-swatch] img",
    "[class*='_swatch_'] img",
    "[class*='swatch'] img[alt]",
    # Button wrappers — AE sometimes puts the color name on the parent button
    "button[class*='swatch'][aria-label]",
    "button[class*='color'][aria-label]",
    "[class*='swatch-container'] button[aria-label]",
    "[data-qa-color-swatch]",
    "[data-qa-color-swatch] img",
    # Generic fallbacks
    "[class*='ColorSwatch'] [aria-label]",
    "[class*='color-swatch'] [aria-label]",
    "[class*='Swatch'] button[aria-label]",
    "button[aria-label][class*='color']",
    "[data-color]",
    "[data-color-name]",
]

# Selector to wait for before extracting — signals the product grid is ready
PRODUCT_GRID_WAIT_SELECTORS = [
    # Confirmed — wait for the product name element to appear
    "[data-testid='name']",
    "[class*='product-name']",
    # Generic fallbacks
    "[class*='ProductGrid']",
    "[class*='product-grid']",
    "[class*='ProductTile']",
    "article",
    "li[class*='product']",
]

# --------------------------------------------------------------------------- #
# Attribute keyword maps                                                        #
# --------------------------------------------------------------------------- #

# For COLOR: checked against swatch aria-labels first, then product title.
# AE uses brand color names like "Twilight", "Stormy", "Cognac" — these are
# listed up-front so they match before generic fallbacks.
COLOR_KEYWORDS: list[tuple[str, str]] = [
    # AE brand-specific color names (checked before generics)
    ("twilight", "blue"),
    ("stormy", "gray"),
    ("onyx", "black"),
    ("wisteria", "purple"),
    ("currant", "red"),
    ("marigold", "beige"),
    ("dusty jade", "green"),
    ("dark rinse", "blue"),
    ("medium rinse", "blue"),
    ("light rinse", "blue"),
    ("rinse", "blue"),
    ("dark wash", "blue"),
    ("medium wash", "blue"),
    ("light wash", "blue"),
    ("acid wash", "blue"),
    ("faded black", "black"),
    ("true black", "black"),
    ("vintage black", "black"),
    ("washed black", "black"),
    ("rinse black", "black"),
    ("midnight", "navy"),
    ("chambray", "blue"),
    ("heritage", "beige"),
    ("clay", "beige"),
    ("wheat", "beige"),
    ("parchment", "white"),
    ("birch", "white"),
    ("bone", "white"),
    ("eggshell", "white"),
    ("chalk", "white"),
    ("oatmeal", "beige"),
    ("cognac", "brown"),
    ("slate", "gray"),
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
    ("amber", "brown"),
    # Standard generics
    ("navy", "navy"),
    ("black", "black"),
    ("off white", "white"),
    ("cloud white", "white"),
    ("white", "white"),
    ("cream", "white"),
    ("ivory", "white"),
    ("burgundy", "red"),
    ("maroon", "red"),
    ("wine", "red"),
    ("coral", "pink"),
    ("rust", "red"),
    ("terracotta", "red"),
    ("brick", "red"),
    ("red", "red"),
    ("sage", "green"),
    ("olive", "green"),
    ("khaki green", "green"),
    ("forest", "green"),
    ("moss", "green"),
    ("green", "green"),
    ("sky blue", "blue"),
    ("cobalt", "blue"),
    ("indigo", "blue"),
    ("teal", "blue"),
    ("blue", "blue"),
    ("light beige", "beige"),
    ("dark beige", "beige"),
    ("beige", "beige"),
    ("tan", "beige"),
    ("camel", "beige"),
    ("sand", "beige"),
    ("taupe", "beige"),
    ("khaki", "beige"),
    ("ecru", "beige"),
    ("mocha", "brown"),
    ("chocolate", "brown"),
    ("espresso", "brown"),
    ("brown", "brown"),
    ("blush", "pink"),
    ("dusty pink", "pink"),
    ("mauve", "pink"),
    ("rose", "pink"),
    ("peach", "pink"),
    ("pink", "pink"),
    ("lavender", "purple"),
    ("lilac", "purple"),
    ("plum", "purple"),
    ("violet", "purple"),
    ("purple", "purple"),
    ("charcoal", "gray"),
    ("heather gray", "gray"),
    ("heather grey", "gray"),
    ("light gray", "gray"),
    ("dark gray", "gray"),
    ("stone", "gray"),
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
    ("wide leg jean", "pants"),
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
    Extract color names from AE swatch image elements.

    AE uses <img class="_swatch-img_..." alt="Gatsby Green" title="Gatsby Green">
    inside each swatch button. The color name can be in the `alt` / `title`
    attribute of the image, or in the `aria-label` of the wrapping button.
    We try all variants so that selector drift on one path still finds the other.
    """
    all_labels: list[str] = []
    seen: set[str] = set()

    for selector in COLOR_SWATCH_SELECTORS:
        try:
            elements = page.query_selector_all(selector)
            if not elements:
                continue
            for el in elements:
                label = (
                    el.get_attribute("alt")
                    or el.get_attribute("aria-label")
                    or el.get_attribute("title")
                    or el.get_attribute("data-color")
                    or el.get_attribute("data-color-name")
                    or ""
                ).strip()
                if label and label not in seen:
                    seen.add(label)
                    all_labels.append(label)
        except Exception:
            continue

    return all_labels


# --------------------------------------------------------------------------- #
# API / page-data color extraction                                              #
# --------------------------------------------------------------------------- #

# Field names (lowercased) whose string values are likely color names.
_COLOR_FIELD_NAMES: frozenset[str] = frozenset([
    "colorname", "color_name", "colordisplayname", "color_display_name",
    "swatchname", "swatch_name", "swatchlabel", "swatch_label",
    "colorlabel", "color_label", "colortitle", "color_title",
    "displayname", "display_name", "label",
])

# Parent-key names that contain arrays of color/swatch objects.
_COLOR_ARRAY_FIELD_NAMES: frozenset[str] = frozenset([
    "colors", "colour", "colours", "swatches", "coloroptions",
    "color_options", "colorfacets", "color_facets", "colorways",
])


def _walk_json_for_colors(data: object, found: list[str], depth: int = 0) -> None:
    """
    Recursively walk a JSON structure looking for color name strings.

    Targets fields whose keys suggest they hold a color name (e.g.
    "colorName", "swatchName", "label" inside a swatches array).
    Avoids going deeper than 10 levels to stay fast on large payloads.
    """
    if depth > 10:
        return
    if isinstance(data, dict):
        for key, value in data.items():
            key_lower = key.lower()
            if key_lower in _COLOR_FIELD_NAMES and isinstance(value, str) and value.strip():
                found.append(value.strip())
            elif key_lower in _COLOR_ARRAY_FIELD_NAMES and isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        found.append(item.strip())
                    elif isinstance(item, dict):
                        _walk_json_for_colors(item, found, depth + 1)
            elif isinstance(value, (dict, list)):
                _walk_json_for_colors(value, found, depth + 1)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _walk_json_for_colors(item, found, depth + 1)


def _extract_colors_from_next_data(page: "Page") -> list[str]:
    """
    Try to extract color names from the Next.js __NEXT_DATA__ JSON block
    embedded in the page HTML. This is server-side rendered and available
    immediately after domcontentloaded, before any XHR fires.
    """
    try:
        script_text: str = page.evaluate(
            "() => { const el = document.getElementById('__NEXT_DATA__'); "
            "return el ? el.textContent : ''; }"
        )
        if not script_text:
            return []
        data = json.loads(script_text)
        colors: list[str] = []
        _walk_json_for_colors(data, colors)
        return colors
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Main scraping loop                                                            #
# --------------------------------------------------------------------------- #

def scrape_american_eagle(
    sleep_between_pages: float = 3.0,
    headless: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Scrape all American Eagle pages.
    Returns (product_titles, color_labels).

    Colors come from three sources tried in priority order:
      1. Network response interception — AE's Next.js app fires XHR/fetch
         calls to load products; we walk those JSON payloads for color fields.
      2. __NEXT_DATA__ extraction — the server-rendered JSON block embedded
         in the HTML often contains the first page of products with colors.
      3. DOM swatch elements — kept as a last-resort fallback in case the
         above two find nothing on a given page.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright is not installed.")
        print("  Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    all_titles: list[str] = []
    all_swatch_colors: list[str] = []
    api_colors: list[str] = []

    def _handle_response(response: object) -> None:
        """Intercept AE XHR/fetch responses and walk JSON for color names."""
        try:
            url: str = response.url  # type: ignore[attr-defined]
            if "ae.com" not in url:
                return
            content_type: str = response.headers.get(  # type: ignore[attr-defined]
                "content-type", ""
            )
            if "json" not in content_type:
                return
            if response.status != 200:  # type: ignore[attr-defined]
                return
            data = response.json()  # type: ignore[attr-defined]
            _walk_json_for_colors(data, api_colors)
        except Exception:
            pass

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
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            "window.chrome = {runtime: {}};"
        )

        page = context.new_page()
        page.on("response", _handle_response)

        for page_info in AMERICAN_EAGLE_PAGES:
            url = page_info["url"]
            label = page_info["label"]
            print(f"  [{label}] → {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                # Wait for the product grid to actually render (Next.js hydration)
                grid_found = _wait_for_products(page, timeout_ms=15_000)
                if not grid_found:
                    page_title = page.title()
                    print(f"    WARNING: product grid not found (page title: '{page_title}')")

                # Pull colors from __NEXT_DATA__ (server-rendered, no XHR needed)
                next_data_colors = _extract_colors_from_next_data(page)
                api_colors.extend(next_data_colors)

                time.sleep(1.5)
                _scroll_to_bottom(page)

                titles = _extract_product_names(page)
                swatches = _extract_swatch_colors(page)

                print(
                    f"    {len(titles)} product titles, "
                    f"{len(swatches)} DOM swatches, "
                    f"{len(next_data_colors)} __NEXT_DATA__ colors, "
                    f"{len(api_colors)} API colors so far"
                )
                all_titles.extend(titles)
                all_swatch_colors.extend(swatches)

            except Exception as exc:
                print(f"    ERROR: {exc}")

            time.sleep(sleep_between_pages)

        browser.close()

    # Priority: API/next-data colors → DOM swatches
    final_colors = api_colors if api_colors else all_swatch_colors
    if api_colors:
        print(f"  Using {len(api_colors)} API/page-data color names")
    else:
        print(f"  API interception yielded nothing — falling back to {len(all_swatch_colors)} DOM swatch labels")

    return all_titles, final_colors


# --------------------------------------------------------------------------- #
# Frequency counting and normalization                                          #
# --------------------------------------------------------------------------- #

def count_attribute_frequencies(
    titles: list[str],
    swatch_colors: list[str],
) -> dict[str, dict[str, int]]:
    """
    Count occurrences of each feature value across all products.

    Color comes from API/page-data interception (swatch_colors), supplemented
    by title keyword extraction. Category and material come from title keywords.
    """
    counts: dict[str, dict[str, int]] = {ft: {} for ft in FEATURE_TYPES}

    # Colors from swatches — each swatch label is one occurrence of that color
    for swatch_label in swatch_colors:
        color = extract_color(swatch_label)
        if color:
            counts["color"][color] = counts["color"].get(color, 0) + 1

    # Category, material, and supplementary color from product titles
    for title in titles:
        category = extract_category(title)
        if category:
            counts["category"][category] = counts["category"].get(category, 0) + 1

        material = extract_material(title, inferred_category=category)
        if material:
            counts["material"][material] = counts["material"].get(material, 0) + 1

        # Color from title: always extracted as a supplement because AE swatch
        # alt attributes use brand-specific names (e.g. "Twilight", "Onyx") that
        # may not match a keyword, while titles sometimes include plain color
        # words ("AE Black Denim Jacket", "Blue Plaid Flannel Shirt").
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
    default_output = (
        Path(__file__).resolve().parents[1]
        / "training" / "synthetic_data" / "trend_signals_american_eagle.csv"
    )
    parser = argparse.ArgumentParser(
        description="Scrape American Eagle new arrivals and write trend_signals_american_eagle.csv."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"American Eagle retail scraper\n"
        f"  pages: {len(AMERICAN_EAGLE_PAGES)}  headless: {args.headless}\n"
        f"  color source: swatch alt/aria-label + title keyword supplement\n"
        f"  output: {output_path}"
    )

    titles, swatch_colors = scrape_american_eagle(
        sleep_between_pages=args.sleep,
        headless=args.headless,
    )

    print(f"\nTotal collected: {len(titles)} product titles, {len(swatch_colors)} swatch colors")

    if not titles and not swatch_colors:
        print(
            "\nWARNING: Nothing was scraped. American Eagle's anti-bot measures\n"
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
