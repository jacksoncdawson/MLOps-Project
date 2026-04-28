"""
Gap retail scraper for trndly trend signals.

Scrapes Gap's "New Arrivals" and category pages using a real browser
(Playwright) to count how often each color, category, and material attribute
appears across featured product listings. Normalizes those counts to 0–1 and
writes the result as trend_signals.csv.

WHERE EACH ATTRIBUTE COMES FROM
--------------------------------
- category : product title keywords  ("jeans" → pants, "hoodie" → tops, etc.)
- material  : product title keywords  ("linen", "denim", "knit", etc.)
             + category inference when no material keyword is in the title
- color     : color swatch aria-labels  (Gap puts the color name in the
             aria-label of each swatch button)
             + product title keywords as a fallback

BOT PROTECTION
--------------
Gap does not use Akamai, PerimeterX, or Imperva, making it well-suited for
headless / pipeline use. The standard de-fingerprinting init scripts are
applied as a precaution. If you hit consistent empty pages, try --headless
false.

NOTE ON URLS
-------------
Gap's category pages use readable path-based URLs which are generally stable.
If a page returns 0 titles, open the URL in a browser and confirm the path
hasn't changed, then update it here.

SETUP (one-time)
----------------
  pip install playwright
  playwright install chromium

Output:
  By default writes to
    trndly/pipelines/training/synthetic_data/trend_signals_gap.csv
  (so it sits alongside trend_signals_hollister.csv, trend_signals_pacsun.csv,
   etc.). Run combine_trend_signals.py afterwards to merge all retailer
   files into the canonical trend_signals.csv.

Usage:
  python gap_scraper.py
  python gap_scraper.py --output-path path/to/trend_signals_gap.csv
  python gap_scraper.py --existing-path trend_signals_gap.csv --blend-weight 0.5
  python gap_scraper.py --headless false   # visible browser
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

GAP_PAGES = [
    {"url": "https://www.gap.com/browse/women/new-arrivals?cid=8792",   "label": "women new arrivals"},
    {"url": "https://www.gap.com/browse/men/new-arrivals?cid=11900",     "label": "men new arrivals"},
    {"url": "https://www.gap.com/browse/women/shirts-and-tops?cid=34608","label": "women tops"},
    {"url": "https://www.gap.com/browse/men/shirts?cid=15043",             "label": "men tops"},
    {"url": "https://www.gap.com/browse/women/jeans?cid=5664",          "label": "women bottoms"},
    {"url": "https://www.gap.com/browse/men/jeans?cid=6998",            "label": "men bottoms"},
    {"url": "https://www.gap.com/browse/women/dresses?cid=13658",        "label": "women dresses"},
    {"url": "https://www.gap.com/browse/women/outerwear-and-jackets?cid=5736", "label": "women outerwear"},
    {"url": "https://www.gap.com/browse/men/coats-and-jackets?cid=5168",   "label": "men outerwear"},
]

# --------------------------------------------------------------------------- #
# Selectors                                                                     #
#                                                                               #
# PRODUCT NAME selectors — tried in order, first non-empty match wins.         #
# Gap's product cards use `data-testid="product-card"` wrappers with a         #
# nested name element.                                                         #
# --------------------------------------------------------------------------- #

PRODUCT_NAME_SELECTORS = [
    # Confirmed from live inspection: h3[data-testid="plp_product-name"]
    "[data-testid='plp_product-name']",
    "h3[data-testid='plp_product-name']",
    "[class*='plp_product-card-name']",
    "[class*='fds__card-copy']",
    # Generic fallbacks
    "[class*='product-name']",
    "[class*='ProductName']",
    "article h2",
    "article h3",
    "li[class*='product'] h2",
    "li[class*='product'] h3",
]

# COLOR SWATCH selectors — Gap puts the color name in aria-label on the swatch.
# The plp_ prefix naming convention suggests swatches may follow a similar pattern.
COLOR_SWATCH_SELECTORS = [
    # Confirmed from live inspection:
    # <label data-testid="fds_selector-swatch" for="pdp-buybox-color-swatch--Navy-blue-dots-mergedGroup-0-0">
    # Color name is parsed from the `for` attribute in _extract_swatch_colors below.
    "[data-testid='fds_selector-swatch']",
    "[class*='fds_selector-swatch']",
    # Generic fallbacks
    "[class*='color-swatch'] [aria-label]",
    "[class*='ColorSwatch'] [aria-label]",
    "[class*='Swatch'] button[aria-label]",
    "[class*='swatch'] button[aria-label]",
    "button[aria-label][class*='color']",
    "[data-color]",
    "[data-color-name]",
]

# Selector to wait for before extracting — signals the product grid is ready
PRODUCT_GRID_WAIT_SELECTORS = [
    # Confirmed from live inspection
    "[data-testid='plp_product-name']",
    "[class*='plp_product-card']",
    "[class*='plp_product']",
    # Generic fallbacks
    "[class*='ProductGrid']",
    "[class*='product-grid']",
    "article",
    "li[class*='product']",
]

# --------------------------------------------------------------------------- #
# Attribute keyword maps                                                        #
# --------------------------------------------------------------------------- #

# For COLOR: checked against swatch aria-labels first, then product title.
# Gap uses some brand color names like "True Black", "Pure White",
# "New Classic Navy", "Natural" — listed up-front so they match first.
COLOR_KEYWORDS: list[tuple[str, str]] = [
    # Gap-specific brand names
    ("true black", "black"),
    ("pure white", "white"),
    ("warm white", "white"),
    ("new classic navy", "navy"),
    ("classic navy", "navy"),
    ("natural", "beige"),
    ("light natural", "beige"),
    ("dark natural", "beige"),
    ("washed black", "black"),
    ("vintage navy", "navy"),
    ("dark indigo", "blue"),
    ("light indigo", "blue"),
    ("medium indigo", "blue"),
    ("light wash", "blue"),
    ("medium wash", "blue"),
    ("dark wash", "blue"),
    ("heather grey", "gray"),
    ("heather gray", "gray"),
    ("light heather", "gray"),
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
    ("bright purple", "purple"),
    ("mustard", "beige"),
    ("butter", "beige"),
    ("golden", "beige"),
    ("terracotta", "red"),
    # Standard generics
    ("navy", "navy"),
    ("rinse black", "black"),
    ("black", "black"),
    ("off white", "white"),
    ("white", "white"),
    ("cream", "white"),
    ("ivory", "white"),
    ("burgundy", "red"),
    ("maroon", "red"),
    ("wine", "red"),
    ("red", "red"),
    ("rust", "red"),
    ("brick", "red"),
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
    ("ecru", "beige"),
    ("khaki", "beige"),
    ("amber", "brown"),
    ("mocha", "brown"),
    ("chocolate", "brown"),
    ("espresso", "brown"),
    ("cognac", "brown"),
    ("brown", "brown"),
    ("blush", "pink"),
    ("dusty pink", "pink"),
    ("mauve", "pink"),
    ("rose", "pink"),
    ("coral", "pink"),
    ("peach", "pink"),
    ("pink", "pink"),
    ("lavender", "purple"),
    ("lilac", "purple"),
    ("plum", "purple"),
    ("violet", "purple"),
    ("purple", "purple"),
    ("charcoal", "gray"),
    ("light gray", "gray"),
    ("dark gray", "gray"),
    ("slate", "gray"),
    ("stone", "gray"),
    ("grey", "gray"),
    ("gray", "gray"),
]

# For CATEGORY: checked against product title.
CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    # Pants / bottoms (product_type: Trousers, Leggings/Tights, Dungarees, Outdoor trousers)
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
    # Dress / full body (product_type: Dress, Jumpsuit/Playsuit, Bodysuit, Romper)
    ("playsuit", "dress"),
    ("romper", "dress"),
    ("jumpsuit", "dress"),
    ("bodysuit", "dress"),
    ("body suit", "dress"),
    ("dress", "dress"),
    # Outerwear (product_type: Jacket, Coat, Blazer, Cardigan, Outdoor Waistcoat, Tailored Waistcoat)
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
    # Tops (product_type: T-shirt, Top, Blouse, Vest top, Hoodie, Sweater, Polo shirt, Shirt)
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
    # Shoes (product_type: Boots, Sneakers, Sandals, Heeled sandals, Flat shoe, Pumps, Wedge, Ballerinas, etc.)
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
    # Accessories (product_type: Bag, Belt, Hat/beanie, Scarf, Socks, Earring, Necklace, Watch, etc.)
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
# Multi-word phrases must appear before their single-word substrings.
MATERIAL_KEYWORDS: list[tuple[str, str]] = [
    # Denim
    ("denim", "denim"),
    ("jean", "denim"),
    # Linen
    ("linen-blend", "linen"),
    ("linen", "linen"),
    # Silk / silk-like (product_type lookup: satin, chiffon, crepe, georgette)
    ("chiffon", "silk"),
    ("crepe", "silk"),
    ("georgette", "silk"),
    ("silk", "silk"),
    ("satin", "silk"),
    # Wool / wool-like (lookup: cashmere, fleece, faux fur, shearling, sherpa)
    ("cashmere", "wool"),
    ("shearling", "wool"),
    ("sherpa", "wool"),
    ("faux fur", "wool"),
    ("wool", "wool"),
    ("fleece", "wool"),
    # Leather / leather-like (lookup: imitation leather, imitation suede, suede)
    ("imitation leather", "leather"),
    ("imitation suede", "leather"),
    ("faux leather", "leather"),
    ("vegan leather", "leather"),
    ("suede", "leather"),
    ("leather", "leather"),
    # Knit / knit-like (lookup: jersey, velvet, velour)
    ("rib-knit", "knit"),
    ("ribbed", "knit"),
    ("jersey", "knit"),
    ("velvet", "knit"),
    ("velour", "knit"),
    ("knit", "knit"),
    ("crochet", "knit"),
    ("waffle", "knit"),
    # Polyester / synthetics (lookup: nylon, acrylic, tulle, mesh, spandex, elastane)
    ("nylon", "polyester"),
    ("acrylic", "polyester"),
    ("tulle", "polyester"),
    ("mesh", "polyester"),
    ("spandex", "polyester"),
    ("elastane", "polyester"),
    ("polyester", "polyester"),
    ("recycled", "polyester"),
    # Cotton / cellulosics (lookup: poplin, twill, terry, corduroy, canvas, tencel, lyocell, modal, viscose, rayon)
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
    Extract color names from Gap swatch label elements.

    Gap uses <label data-testid="fds_selector-swatch"> elements where the
    color name is embedded in the `for` attribute, e.g.:
      for="pdp-buybox-color-swatch--Navy-blue-dots-mergedGroup-0-0"
    → color name: "Navy blue dots"

    Falls back to aria-label / data-color for any generic swatch elements.
    """
    import re

    for selector in COLOR_SWATCH_SELECTORS:
        try:
            elements = page.query_selector_all(selector)
            if not elements:
                continue
            labels: list[str] = []
            for el in elements:
                label = (
                    el.get_attribute("aria-label")
                    or el.get_attribute("data-color")
                    or el.get_attribute("data-color-name")
                    or ""
                )
                # Gap-specific: color name is encoded in the `for` attribute.
                # Pattern: pdp-buybox-color-swatch--{Color-name}-mergedGroup-N-N
                if not label:
                    for_attr = el.get_attribute("for") or ""
                    match = re.search(
                        r"pdp-buybox-color-swatch--(.+?)-mergedGroup", for_attr
                    )
                    if match:
                        label = match.group(1).replace("-", " ")
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

def scrape_gap(
    sleep_between_pages: float = 3.0,
    headless: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Scrape all Gap pages.
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
        # Mask the webdriver flag that common bot detectors check
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            "window.chrome = {runtime: {}};"
        )

        page = context.new_page()

        for page_info in GAP_PAGES:
            url = page_info["url"]
            label = page_info["label"]
            print(f"  [{label}] → {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                # Wait for the product grid to actually render
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
        / "training" / "synthetic_data" / "trend_signals_gap.csv"
    )
    parser = argparse.ArgumentParser(
        description="Scrape Gap new arrivals and write trend_signals_gap.csv."
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
        f"Gap retail scraper\n"
        f"  pages: {len(GAP_PAGES)}  headless: {args.headless}\n"
        f"  output: {output_path}"
    )

    titles, swatch_colors = scrape_gap(
        sleep_between_pages=args.sleep,
        headless=args.headless,
    )

    print(f"\nTotal collected: {len(titles)} product titles, {len(swatch_colors)} swatch colors")

    if not titles and not swatch_colors:
        print(
            "\nWARNING: Nothing was scraped. Gap's bot protection may be\n"
            "blocking headless Chrome. Try running with --headless false\n"
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
