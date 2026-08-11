
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ingest import load_catalog_from_excel, load_catalog_from_csv, load_catalog_from_pdf
from milp_solver import build_and_solve, report

st.set_page_config(
    page_title="Procurement Optimizer",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 Procurement Optimizer")
st.caption("Upload supplier buying manuals, enter demand, and calculate an optimized purchasing basket.")

if "suppliers" not in st.session_state:
    st.session_state.suppliers = {}
if "demand" not in st.session_state:
    st.session_state.demand = {}

st.sidebar.header("Supplier setup")

with st.sidebar.form("supplier_form", clear_on_submit=True):
    supplier_name = st.text_input("Supplier name")
    moq = st.number_input("Minimum order value (R)", min_value=0.0, value=500.0, step=50.0)
    free_threshold = st.number_input("Free-delivery threshold (R)", min_value=0.0, value=500.0, step=50.0)
    delivery_fee = st.number_input("Delivery fee (R)", min_value=0.0, value=100.0, step=10.0)
    uploaded = st.file_uploader("Buying manual", type=["xlsx", "xls", "csv", "pdf"])
    add_supplier = st.form_submit_button("Add supplier")

if add_supplier:
    if not supplier_name.strip():
        st.sidebar.error("Enter a supplier name.")
    elif uploaded is None:
        st.sidebar.error("Upload a buying manual.")
    elif supplier_name in st.session_state.suppliers:
        st.sidebar.error("That supplier already exists.")
    else:
        suffix = Path(uploaded.name).suffix.lower()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.getvalue())
        tmp.close()

        try:
            if suffix in [".xlsx", ".xls"]:
                catalog = load_catalog_from_excel(tmp.name)
            elif suffix == ".csv":
                catalog = load_catalog_from_csv(tmp.name)
            else:
                catalog = load_catalog_from_pdf(tmp.name)

            st.session_state.suppliers[supplier_name] = {
                "moq": moq,
                "free_delivery_threshold": free_threshold,
                "delivery_fee": delivery_fee,
                "catalog": catalog,
            }
            st.sidebar.success(f"Added {supplier_name}: {len(catalog)} items.")
        except Exception as e:
            st.sidebar.error(f"Could not read the buying manual: {e}")
        finally:
            Path(tmp.name).unlink(missing_ok=True)

st.sidebar.divider()
st.sidebar.subheader("Suppliers")

if st.session_state.suppliers:
    for name, data in list(st.session_state.suppliers.items()):
        c1, c2 = st.sidebar.columns([4, 1])
        c1.write(f"**{name}** — {len(data['catalog'])} items")
        if c2.button("✕", key=f"remove_{name}"):
            del st.session_state.suppliers[name]
            st.rerun()
else:
    st.sidebar.info("No suppliers added yet.")

tab1, tab2, tab3 = st.tabs(["📦 Demand", "⚙️ Optimize", "📊 Results"])

with tab1:
    st.subheader("Required demand")
    st.write("Enter the quantity required for each item. Items are pulled from the uploaded supplier catalogs.")

    all_items = sorted({
        item
        for supplier in st.session_state.suppliers.values()
        for item in supplier["catalog"]
    })

    if not all_items:
        st.info("Add at least one supplier and buying manual in the sidebar.")
    else:
        default_rows = []
        for item in all_items:
            default_rows.append({
                "Item": item,
                "Required quantity": float(st.session_state.demand.get(item, 0)),
            })
        df = pd.DataFrame(default_rows)
        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Required quantity": st.column_config.NumberColumn(
                    min_value=0.0, step=1.0, format="%.2f"
                )
            },
            disabled=["Item"],
            key="demand_editor",
        )

        if st.button("Save demand", type="primary"):
            st.session_state.demand = {
                row["Item"]: float(row["Required quantity"])
                for _, row in edited.iterrows()
                if float(row["Required quantity"]) > 0
            }
            st.success(f"Saved {len(st.session_state.demand)} required items.")

with tab2:
    st.subheader("Optimization settings")
    admin_fee = st.number_input("Admin PO fee (R)", min_value=0.0, value=100.0, step=10.0)
    buffer_cap = st.number_input("Default buffer cap (units)", min_value=0.0, value=5.0, step=1.0)

    st.write("**Current required demand**")
    if st.session_state.demand:
        st.dataframe(
            pd.DataFrame(
                [{"Item": k, "Quantity": v} for k, v in st.session_state.demand.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No demand has been saved yet.")

    if st.button("🚀 Optimize purchasing basket", type="primary", use_container_width=True):
        if not st.session_state.suppliers:
            st.error("Add at least one supplier.")
        elif not st.session_state.demand:
            st.error("Enter and save at least one required item.")
        else:
            try:
                # Validate all required items before solving.
                missing = [
                    item for item in st.session_state.demand
                    if not any(item in s["catalog"] for s in st.session_state.suppliers.values())
                ]
                if missing:
                    st.error("Required items missing from all catalogs: " + ", ".join(missing))
                else:
                    result, pairs, pair_index, supplier_names, y_offset, z_offset, buffer_items = build_and_solve(
                        st.session_state.suppliers,
                        st.session_state.demand,
                        admin_po_fee=admin_fee,
                        buffer_caps={item: buffer_cap for item in buffer_items}
                        if False else None,
                    )

                    st.session_state.optimization = {
                        "result": result,
                        "pairs": pairs,
                        "pair_index": pair_index,
                        "supplier_names": supplier_names,
                        "y_offset": y_offset,
                        "z_offset": z_offset,
                        "buffer_items": buffer_items,
                        "admin_fee": admin_fee,
                    }
                    if result.success:
                        st.success(f"Optimization complete. Optimal landed cost: R{result.fun:,.2f}")
                    else:
                        st.error(f"Solver failed: {result.message}")
            except Exception as e:
                st.exception(e)

with tab3:
    st.subheader("Optimized purchasing basket")
    opt = st.session_state.get("optimization")

    if not opt:
        st.info("Run the optimizer to see results.")
    elif not opt["result"].success:
        st.error(opt["result"].message)
    else:
        result = opt["result"]
        suppliers = st.session_state.suppliers
        pairs = opt["pairs"]
        pair_index = opt["pair_index"]
        supplier_names = opt["supplier_names"]
        y_offset = opt["y_offset"]
        z_offset = opt["z_offset"]
        buffer_items = opt["buffer_items"]
        admin_fee = opt["admin_fee"]

        x = result.x
        total = float(result.fun)

        c1, c2, c3 = st.columns(3)
        c1.metric("Optimal landed cost", f"R{total:,.2f}")
        active = sum(1 for j in range(len(supplier_names)) if x[y_offset+j] >= 0.5)
        c2.metric("Suppliers used", active)
        c3.metric("Buffer-eligible items", len(buffer_items))

        rows = []
        for j, s in enumerate(supplier_names):
            if x[y_offset+j] < 0.5:
                continue
            spend = 0.0
            for (item, sup), idx in pair_index.items():
                if sup == s and x[idx] > 1e-6:
                    qty = float(x[idx])
                    price = float(suppliers[s]["catalog"][item]["price"])
                    spend += qty * price
                    rows.append({
                        "Supplier": s,
                        "Item": item,
                        "Type": "Buffer" if item in buffer_items else "Required",
                        "Quantity": qty,
                        "Unit price": price,
                        "Subtotal": qty * price,
                    })

            free_delivery = x[z_offset+j] >= 0.5
            delivery = 0.0 if free_delivery else float(suppliers[s]["delivery_fee"])
            po_total = spend + delivery + admin_fee
            st.markdown(f"### {s}")
            a, b, c, d = st.columns(4)
            a.metric("Product spend", f"R{spend:,.2f}")
            b.metric("MOQ", f"R{suppliers[s]['moq']:,.2f}")
            c.metric("Delivery", "FREE" if free_delivery else f"R{delivery:,.2f}")
            d.metric("PO total", f"R{po_total:,.2f}")

        if rows:
            result_df = pd.DataFrame(rows)
            st.dataframe(
                result_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Quantity": st.column_config.NumberColumn(format="%.2f"),
                    "Unit price": st.column_config.NumberColumn(format="R%.2f"),
                    "Subtotal": st.column_config.NumberColumn(format="R%.2f"),
                },
            )

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download purchase basket CSV",
                csv,
                "optimized_purchase_basket.csv",
                "text/csv",
            )

        st.markdown("### Buffer-eligible items")
        if buffer_items:
            st.write(", ".join(sorted(buffer_items)))
        else:
            st.write("None")

st.divider()
st.caption("Powered by the uploaded buying-manual ingestion and MILP optimization modules.")
