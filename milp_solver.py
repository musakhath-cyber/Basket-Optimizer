"""
Multi-Supplier Basket Optimization — Real MILP Solver (v2)
=============================================================
Same MILP formulation as v1 (see milp_solver.py for the full derivation),
generalized so buffer-item eligibility is derived from catalog metadata
rather than a hardcoded list — so it works with any supplier catalog,
including ones loaded from real buying manuals via ingest.py.

Catalog item schema (per supplier):
    "Item Name (unit)": {"price": float, "category": str, "perishable": bool}

Any catalog item that is NOT in `required_demand` is automatically
buffer-eligible IF it's marked non-perishable (or has no perishable flag
set to True) — the solver decides whether buying it is cost-optimal.
Perishable items you don't explicitly need are never auto-purchased
(no point stockpiling lettuce).
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

VMAX = 1_000_000.0          # Big-M
ADMIN_PO_FEE_DEFAULT = 100.0
DEFAULT_BUFFER_CAP = 5.0    # units, when a supplier doesn't specify holding capacity


def build_and_solve(suppliers, required_demand, admin_po_fee=ADMIN_PO_FEE_DEFAULT,
                     buffer_caps=None, vmax=VMAX):
    """
    suppliers: {
        supplier_name: {
            "moq": float,
            "free_delivery_threshold": float,
            "delivery_fee": float,
            "catalog": {item_name: {"price": float, "perishable": bool, ...}}
        }
    }
    required_demand: {item_name: quantity}
    buffer_caps: optional {item_name: max_forward_buy_qty} override;
                 defaults to DEFAULT_BUFFER_CAP for any non-perishable
                 item not otherwise capped.
    """
    buffer_caps = buffer_caps or {}
    supplier_names = list(suppliers.keys())
    n_suppliers = len(supplier_names)

    # Identify buffer-eligible items: present in a catalog, not required,
    # and not flagged perishable.
    buffer_items = set()
    for s in supplier_names:
        for item, meta in suppliers[s]["catalog"].items():
            if item in required_demand:
                continue
            if not meta.get("perishable", True):
                buffer_items.add(item)

    pairs = []
    for s in supplier_names:
        for item in suppliers[s]["catalog"]:
            if item in required_demand or item in buffer_items:
                pairs.append((item, s))

    n_x = len(pairs)
    pair_index = {pair: idx for idx, pair in enumerate(pairs)}
    n_y = n_suppliers
    n_z = n_suppliers
    n_vars = n_x + n_y + n_z
    y_offset = n_x
    z_offset = n_x + n_y

    # ---- Objective ----
    c = np.zeros(n_vars)
    for (item, s), idx in pair_index.items():
        c[idx] = suppliers[s]["catalog"][item]["price"]
    for j, s in enumerate(supplier_names):
        f_s = suppliers[s]["delivery_fee"]
        c[y_offset + j] = admin_po_fee + f_s
        c[z_offset + j] = -f_s

    # ---- Bounds ----
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    for (item, s), idx in pair_index.items():
        if item in buffer_items:
            ub[idx] = buffer_caps.get(item, DEFAULT_BUFFER_CAP)
        else:
            ub[idx] = 100_000  # generous cap for required items
    for j in range(n_suppliers):
        lb[y_offset + j], ub[y_offset + j] = 0, 1
        lb[z_offset + j], ub[z_offset + j] = 0, 1

    integrality = np.zeros(n_vars)
    integrality[y_offset:] = 1
    bounds = Bounds(lb, ub)

    # ---- Constraints ----
    constraints = []

    for item, d_i in required_demand.items():
        row = np.zeros(n_vars)
        found = False
        for s in supplier_names:
            if item in suppliers[s]["catalog"]:
                row[pair_index[(item, s)]] = 1.0
                found = True
        if not found:
            raise ValueError(f"Required item '{item}' not found in any supplier catalog.")
        constraints.append(LinearConstraint(row, d_i, d_i))

    for j, s in enumerate(supplier_names):
        spend_row = np.zeros(n_vars)
        for item in suppliers[s]["catalog"]:
            if (item, s) in pair_index:
                spend_row[pair_index[(item, s)]] = suppliers[s]["catalog"][item]["price"]

        M_s = suppliers[s]["moq"]
        T_s = suppliers[s]["free_delivery_threshold"]

        row2 = spend_row.copy()
        row2[y_offset + j] = -vmax
        constraints.append(LinearConstraint(row2, -np.inf, 0))

        row3 = -spend_row.copy()
        row3[y_offset + j] = M_s
        constraints.append(LinearConstraint(row3, -np.inf, 0))

        row4a = -spend_row.copy()
        row4a[z_offset + j] = T_s
        constraints.append(LinearConstraint(row4a, -np.inf, 0))

        row4b = np.zeros(n_vars)
        row4b[z_offset + j] = 1
        row4b[y_offset + j] = -1
        constraints.append(LinearConstraint(row4b, -np.inf, 0))

    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    return result, pairs, pair_index, supplier_names, y_offset, z_offset, buffer_items


def report(result, pairs, pair_index, supplier_names, y_offset, z_offset,
           suppliers, buffer_items, admin_po_fee=ADMIN_PO_FEE_DEFAULT):
    if not result.success:
        print("SOLVER FAILED:", result.message)
        return None

    x = result.x
    print("=" * 78)
    print(" MILP-OPTIMIZED PROCUREMENT BASKET (scipy.optimize.milp / HiGHS)")
    print("=" * 78)
    print(f"Total Landed Cost (optimal): R{result.fun:.2f}\n")

    breakdown = {}
    for j, s in enumerate(supplier_names):
        y_val = x[y_offset + j]
        z_val = x[z_offset + j]
        if y_val < 0.5:
            continue

        items_bought = []
        spend = 0.0
        for (item, sup), idx in pair_index.items():
            if sup != s:
                continue
            qty = x[idx]
            if qty > 1e-6:
                price = suppliers[s]["catalog"][item]["price"]
                subtotal = qty * price
                spend += subtotal
                kind = "Buffer" if item in buffer_items else "Required"
                items_bought.append((item, qty, price, subtotal, kind))

        delivery_fee = suppliers[s]["delivery_fee"] if z_val < 0.5 else 0.0
        po_total = spend + delivery_fee + admin_po_fee
        breakdown[s] = {"spend": spend, "delivery_fee": delivery_fee,
                         "admin_fee": admin_po_fee, "po_total": po_total, "items": items_bought}

        print(f"SUPPLIER: {s.upper()}")
        print(f"  Spend: R{spend:.2f}  |  MOQ: R{suppliers[s]['moq']:.2f}  |  "
              f"Free-delivery met: {'YES' if z_val > 0.5 else 'NO'}")
        print(f"  Delivery Fee: R{delivery_fee:.2f}   Admin PO Fee: R{admin_po_fee:.2f}")
        print(f"  Purchase Order Total: R{po_total:.2f}")
        for item, qty, price, subtotal, kind in items_bought:
            print(f"    - {item}: {qty:.2f} @ R{price:.2f} = R{subtotal:.2f}  [{kind}]")
        print()

    print("=" * 78)
    return breakdown


# ----------------------------------------------------------------------
# Demo using the same data as v1, but now with the generalized catalog
# schema (price/category/perishable) so it exercises the auto buffer
# detection instead of a hardcoded item list.
# ----------------------------------------------------------------------

if __name__ == "__main__":
    suppliers = {
        "Bell Ceres": {
            "moq": 500.0, "free_delivery_threshold": 500.0, "delivery_fee": 80.0,
            "catalog": {
                "Lettuce Fresh (ea)": {"price": 25.0, "category": "Fresh Veg", "perishable": True},
                "Tomatoes (kg)": {"price": 21.0, "category": "Fresh Veg", "perishable": True},
                "Potatoes Large (kg)": {"price": 13.80, "category": "Fresh Veg", "perishable": True},
                "Onions White (kg)": {"price": 13.00, "category": "Fresh Veg", "perishable": True},
                "Cucumbers Fresh (kg)": {"price": 35.00, "category": "Fresh Veg", "perishable": True},
            },
        },
        "Grocery Express": {
            "moq": 1000.0, "free_delivery_threshold": 1000.0, "delivery_fee": 150.0,
            "catalog": {
                "Cucumbers Fresh (kg)": {"price": 20.00, "category": "Fresh Veg", "perishable": True},
                "Potatoes Large (kg)": {"price": 18.00, "category": "Fresh Veg", "perishable": True},
                "Cooking Oil 20L (ea)": {"price": 980.00, "category": "Dry Goods", "perishable": False},
                "Rice White 10kg (ea)": {"price": 220.00, "category": "Dry Goods", "perishable": False},
                "Sugar White 25kg (ea)": {"price": 763.44, "category": "Dry Goods", "perishable": False},
            },
        },
        "Cuyler Butchery": {
            "moq": 800.0, "free_delivery_threshold": 800.0, "delivery_fee": 100.0,
            "catalog": {
                "Beef Mince (kg)": {"price": 99.94, "category": "Meat", "perishable": True},
            },
        },
        "Crickley Dairy": {
            "moq": 400.0, "free_delivery_threshold": 400.0, "delivery_fee": 60.0,
            "catalog": {
                "Fresh Milk 2L (ea)": {"price": 31.44, "category": "Dairy", "perishable": True},
                "Cheddar Cheese Bulk (kg)": {"price": 112.22, "category": "Dairy", "perishable": True},
            },
        },
        "Unick Foods": {
            "moq": 700.0, "free_delivery_threshold": 700.0, "delivery_fee": 100.0,
            "catalog": {
                "Chicken Thighs (kg)": {"price": 58.84, "category": "Meat", "perishable": True},
            },
        },
    }

    required_demand = {
        "Lettuce Fresh (ea)": 15, "Tomatoes (kg)": 30, "Potatoes Large (kg)": 50,
        "Onions White (kg)": 30, "Cucumbers Fresh (kg)": 20, "Beef Mince (kg)": 25,
        "Chicken Thighs (kg)": 35, "Fresh Milk 2L (ea)": 20, "Cheddar Cheese Bulk (kg)": 10,
    }

    result, pairs, pair_index, supplier_names, y_offset, z_offset, buffer_items = build_and_solve(
        suppliers, required_demand
    )
    print(f"Auto-detected buffer-eligible items: {sorted(buffer_items)}\n")
    report(result, pairs, pair_index, supplier_names, y_offset, z_offset, suppliers, buffer_items)
