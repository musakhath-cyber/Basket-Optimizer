"""
Buying Manual Ingestion
========================
Parses real-world supplier "buying manuals" (Excel, CSV, or PDF price
lists) into the catalog dict format the MILP solver expects:

    {
        "Item Name (unit)": {"price": float, "category": str, "perishable": bool},
        ...
    }

Handles the two messy realities of these documents (per the blueprint's
Module 1 spec):
  1. Column names vary supplier-to-supplier ("Product"/"Item"/"Description",
     "Unit Price"/"Price"/"Cost", etc.) — resolved via fuzzy header matching.
  2. Prices are quoted per pack/case/bag, not per base unit
     (e.g. "R246.10 per 12.5kg bag" -> R19.69/kg) — resolved via a UOM
     normalizer that extracts a pack size from the unit text.
"""

import re
import pandas as pd

# ----------------------------------------------------------------------
# Header resolution — map varied column names to canonical fields
# ----------------------------------------------------------------------

CANONICAL_HEADERS = {
    "item": ["item", "product", "description", "product name", "item name", "sku name"],
    "price": ["price", "unit price", "cost", "price (excl vat)", "price excl vat", "rate"],
    "uom": ["uom", "unit", "unit of measure", "pack size", "pack"],
    "category": ["category", "dept", "department", "product category"],
}

# Categories that default to perishable / non-perishable when the source
# file has no explicit "perishable" column — used by the buffer logic.
PERISHABLE_CATEGORIES = {
    "fresh veg", "fresh produce", "produce", "meat", "poultry", "dairy",
    "butchery", "seafood", "fish", "bakery",
}
NON_PERISHABLE_CATEGORIES = {
    "dry goods", "chemicals", "cleaning", "packaging", "beverages", "grocery",
}


def _resolve_headers(columns):
    """Map a DataFrame's actual column names to canonical field names."""
    lower_cols = {c: str(c).strip().lower() for c in columns}
    resolved = {}
    for canon, aliases in CANONICAL_HEADERS.items():
        for col, low in lower_cols.items():
            if low in aliases:
                resolved[canon] = col
                break
    return resolved


# ----------------------------------------------------------------------
# UOM normalization — "R246.10 per 12.5kg bag" -> R19.69/kg
# ----------------------------------------------------------------------

_PACK_SIZE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg|g|l|ml|ea|each|unit|units)", re.IGNORECASE
)


def normalize_unit_price(raw_price, uom_text):
    """
    Given a pack price and a UOM description (e.g. '12.5kg bag', '20L',
    '1kg', 'each'), return (unit_price, base_unit).

    If no pack size is found in the UOM text, assumes the price is
    already per-unit and returns it unchanged with base_unit='ea'.
    """
    if not uom_text or not isinstance(uom_text, str):
        return raw_price, "ea"

    match = _PACK_SIZE_RE.search(uom_text)
    if not match:
        return raw_price, "ea"

    pack_size = float(match.group(1))
    unit = match.group(2).lower()
    unit = {"g": "kg", "ml": "l", "each": "ea", "units": "ea", "unit": "ea"}.get(unit, unit)

    if pack_size <= 0:
        return raw_price, unit

    # Normalize gram/ml packs into kg/l base units
    if match.group(2).lower() == "g":
        pack_size = pack_size / 1000.0
    elif match.group(2).lower() == "ml":
        pack_size = pack_size / 1000.0

    return round(raw_price / pack_size, 4), unit


def _infer_perishable(category):
    if not category or not isinstance(category, str):
        return True  # conservative default — treat unknowns as perishable (not buffer-eligible)
    cat_lower = category.strip().lower()
    if cat_lower in NON_PERISHABLE_CATEGORIES:
        return False
    if cat_lower in PERISHABLE_CATEGORIES:
        return True
    return True


# ----------------------------------------------------------------------
# Public loaders
# ----------------------------------------------------------------------

def load_catalog_from_table(df, normalize_uom=True):
    """
    Convert a raw DataFrame (from Excel/CSV/PDF table extraction) into
    the solver's catalog dict: {item_name: {"price", "category", "perishable"}}
    """
    headers = _resolve_headers(df.columns)
    if "item" not in headers or "price" not in headers:
        raise ValueError(
            f"Could not find item/price columns. Found columns: {list(df.columns)}. "
            f"Resolved: {headers}. Rename columns or extend CANONICAL_HEADERS."
        )

    catalog = {}
    for _, row in df.iterrows():
        item_name = row.get(headers["item"])
        raw_price = row.get(headers["price"])
        if pd.isna(item_name) or pd.isna(raw_price):
            continue

        try:
            raw_price = float(str(raw_price).replace("R", "").replace(",", "").strip())
        except ValueError:
            continue

        uom_text = row.get(headers["uom"]) if "uom" in headers else None
        category = row.get(headers["category"]) if "category" in headers else None

        if normalize_uom and uom_text:
            unit_price, base_unit = normalize_unit_price(raw_price, str(uom_text))
            display_name = f"{item_name} ({base_unit})"
        else:
            unit_price = raw_price
            display_name = str(item_name)

        catalog[display_name] = {
            "price": unit_price,
            "category": category if isinstance(category, str) else "Uncategorized",
            "perishable": _infer_perishable(category),
        }

    return catalog


def load_catalog_from_excel(path, sheet_name=0):
    df = pd.read_excel(path, sheet_name=sheet_name)
    return load_catalog_from_table(df)


def load_catalog_from_csv(path):
    df = pd.read_csv(path)
    return load_catalog_from_table(df)


def load_catalog_from_pdf(path, page_numbers=None):
    """
    Extracts table(s) from a PDF buying manual using pdfplumber and
    merges them into a single catalog dict. Falls back gracefully if a
    page has no detectable table.
    """
    import pdfplumber

    frames = []
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages if page_numbers is None else [pdf.pages[i] for i in page_numbers]
        for page in pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                header, *rows = table
                df = pd.DataFrame(rows, columns=header)
                frames.append(df)

    if not frames:
        raise ValueError(
            f"No tables detected in {path}. The PDF may be scanned (no text layer) — "
            f"try OCR first, or rasterize pages and extract manually."
        )

    combined = pd.concat(frames, ignore_index=True)
    return load_catalog_from_table(combined)


def load_supplier(name, moq, delivery_fee, path=None, free_delivery_threshold=None,
                   catalog=None, file_type="auto"):
    """
    Convenience wrapper: builds one supplier entry (in the solver's
    `suppliers` dict format) either from a file path or a pre-built catalog.
    """
    if catalog is None:
        if path is None:
            raise ValueError("Must provide either `path` or `catalog`.")
        ext = file_type if file_type != "auto" else path.rsplit(".", 1)[-1].lower()
        if ext in ("xlsx", "xls"):
            catalog = load_catalog_from_excel(path)
        elif ext == "csv":
            catalog = load_catalog_from_csv(path)
        elif ext == "pdf":
            catalog = load_catalog_from_pdf(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    return {
        "moq": moq,
        "free_delivery_threshold": free_delivery_threshold if free_delivery_threshold is not None else moq,
        "delivery_fee": delivery_fee,
        "catalog": catalog,
    }
