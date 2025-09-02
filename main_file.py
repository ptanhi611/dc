import streamlit as st
import pandas as pd
from itertools import combinations
import plotly.graph_objects as go
import base64

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="K-Map Logic Minimizer", page_icon="⚡")

# --- Custom Design ---
def local_css():
    # A subtle, self-contained SVG background for a professional "blueprint" feel.
    bg_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
        <defs>
            <pattern id="smallGrid" width="10" height="10" patternUnits="userSpaceOnUse">
                <path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(50, 100, 150, 0.1)" stroke-width="0.5"/>
            </pattern>
            <pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse">
                <rect width="100" height="100" fill="url(#smallGrid)"/>
                <path d="M 100 0 L 0 0 0 100" fill="none" stroke="rgba(50, 100, 150, 0.2)" stroke-width="1"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
    </svg>
    """
    bg_svg_b64 = base64.b64encode(bg_svg.encode('utf-8')).decode('utf-8')

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Roboto+Mono:wght@500&display=swap');
        
        .stApp {{
            background-color: #0f172a;
            background-image: url("data:image/svg+xml;base64,{bg_svg_b64}");
        }}
        
        .main .block-container {{ padding-top: 2rem; }}
        
        /* K-MAP BUTTONS: Made significantly larger and more impactful */
        .stButton>button {{
            width: 100%;
            height: 6em; /* SIGNIFICANTLY TALLER */
            font-size: 1.8em; /* MUCH LARGER FONT */
            font-weight: 500;
            font-family: 'Roboto Mono', monospace;
            border-radius: 12px;
            border: 2px solid #4a5568;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .stButton>button:hover {{
            border-color: #38bdf8;
            transform: translateY(-3px);
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.6);
        }}
        
        h1, h2, h3 {{
            color: #e2e8f0;
            text-align: center;
            font-family: 'Poppins', sans-serif;
        }}
        
        .axis-label {{
            font-size: 1.4em;
            font-weight: 600;
            font-family: 'Poppins', sans-serif;
            color: #38bdf8;
            text-align: center;
        }}
        .row-label {{
             font-size: 1.3em;
             font-weight: 600;
             font-family: 'Roboto Mono', monospace;
             text-align: right;
             padding-top: 2.3em;
        }}
        </style>
    """, unsafe_allow_html=True)

# --- CORE LOGIC FUNCTIONS ---
def gray_code(n):
    if n == 0: return ['']
    first_half = gray_code(n - 1)
    return ['0' + code for code in first_half] + ['1' + code for code in first_half[::-1]]

def dec_to_bin(dec, num_vars):
    return format(dec, f'0{num_vars}b')

def get_kmap_indices(num_vars):
    if num_vars == 2: return gray_code(1), gray_code(1)
    if num_vars == 3: return gray_code(1), gray_code(2)
    if num_vars == 4: return gray_code(2), gray_code(2)
    if num_vars == 5: return gray_code(2), gray_code(3)
    return [], []

def combine_terms(term1, term2):
    diff_count, new_term = 0, ""
    for c1, c2 in zip(term1, term2):
        if c1 != c2:
            diff_count += 1; new_term += '-'
        else: new_term += c1
    return new_term if diff_count == 1 else None

def find_prime_implicants(minterms_dc, num_vars):
    if not minterms_dc: return set()
    terms = {dec_to_bin(m, num_vars) for m in minterms_dc}
    prime_implicants = set()
    while True:
        new_terms, uncombined = set(), set(terms)
        for t1, t2 in combinations(terms, 2):
            combined = combine_terms(t1, t2)
            if combined:
                new_terms.add(combined)
                uncombined.discard(t1); uncombined.discard(t2)
        prime_implicants.update(uncombined)
        if not new_terms or terms == new_terms: break
        terms = new_terms
    return prime_implicants

def get_minimal_cover(prime_implicants, minterms, num_vars):
    if not minterms: return []
    if not prime_implicants: return []
    minterm_bins = {dec_to_bin(m, num_vars) for m in minterms}
    chart = {pi: [mt for mt in minterm_bins if all((p == m or p == '-') for p, m in zip(pi, mt))] for pi in prime_implicants}
    
    essential_pis, covered_minterms = set(), set()
    minterm_to_pis = {mt: [pi for pi, mts in chart.items() if mt in mts] for mt in minterm_bins}
    for mt, pis in minterm_to_pis.items():
        if len(pis) == 1:
            essential_pi = pis[0]
            essential_pis.add(essential_pi)
            covered_minterms.update(chart[essential_pi])

    uncovered_minterms = minterm_bins - covered_minterms
    remaining_pis = {pi for pi in prime_implicants if pi not in essential_pis}
    
    final_cover = set(essential_pis)
    while uncovered_minterms:
        best_pi = max(remaining_pis, key=lambda pi: len(set(chart[pi]) & uncovered_minterms), default=None)
        if not best_pi: break
        final_cover.add(best_pi)
        uncovered_minterms -= set(chart[best_pi])
        remaining_pis.remove(best_pi)
    return list(final_cover)

def term_to_expression(term, variables, is_pos):
    parts = []
    for i, char in enumerate(term):
        if char == '0': parts.append(variables[i] if is_pos else variables[i] + "'")
        elif char == '1': parts.append(variables[i] + "'" if is_pos else variables[i])
    joiner = " + " if is_pos else ""
    return f"({joiner.join(parts)})" if is_pos and parts else "".join(parts)

def generate_full_expression(cover, variables, is_pos):
    if not cover: return "0" if is_pos else "1"
    expr_parts = [term_to_expression(term, variables, is_pos) for term in sorted(cover)]
    return (" * ").join(expr_parts) if is_pos else (" + ").join(expr_parts)

# --- Verilog and Waveform Generation ---
def generate_verilog(expression, variables):
    verilog_expr = expression.replace("'", "~").replace("+", "|").replace("*", "&").replace(" ", "")
    if not verilog_expr or verilog_expr == "()": verilog_expr = "1'b0"

    lines = [f"// Synthesizable module for: F = {expression}",
        "`timescale 1ns/1ps", "module minimized_logic(",
        f"    input  {', '.join(variables)},", "    output F", ");", "",
        f"    assign F = {verilog_expr};", "", "endmodule"]
    return "\n".join(lines)

def generate_testbench(variables):
    num_vars = len(variables)
    dut_ports = ",\n        ".join([f".{v}(i[{num_vars-1-idx}])" for idx, v in enumerate(variables)])
    lines = ["// Testbench to verify the logic", "module tb_minimized_logic;", "",
        f"    reg [{num_vars-1}:0] i;", "    wire F;", "",
        "    minimized_logic dut (", f"        {dut_ports}", "    );", "",
        "    initial begin", "        i = 0;",
        "        $dumpfile(\"waveform.vcd\");", "        $dumpvars(0, tb_minimized_logic);",
        f"        for (i = 0; i < {2**num_vars}; i = i + 1) begin", "            #10;", "        end",
        "        $finish;", "    end", "endmodule"]
    return "\n".join(lines)

def generate_waveform(kmap_states, num_vars):
    traces = {var: [] for var in ['A', 'B', 'C', 'D', 'E'][:num_vars] + ['F']}
    
    for i in range(2**num_vars):
        input_bin = dec_to_bin(i, num_vars)
        for j, var_name in enumerate(traces.keys()):
            if var_name != 'F':
                traces[var_name].append(int(input_bin[j]))
        
        # Determine output based on the original K-map state
        cell_val = kmap_states.get(f'kmap_cell_{i}', '0')
        output = 1 if cell_val == '1' else 0 # Treat 'x' and '0' as 0 for waveform output
        traces['F'].append(output)

    fig = go.Figure()
    time_steps = list(range(2**num_vars))
    
    for i, (signal, values) in enumerate(traces.items()):
        y_pos = len(traces) - i
        # BUG FIX: Duplicate hover data to match duplicated coordinates
        x_coords = [t for t in time_steps for _ in (0, 1)]
        y_coords = [y_pos + 0.4 * val for val in values for _ in (0, 1)]
        hover_values = [v for v in values for _ in (0, 1)]
        
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode='lines', name=signal,
            line=dict(shape='hv', width=4 if signal == 'F' else 2),
            customdata=hover_values,
            hovertemplate='<b>Value</b>: %{customdata}<extra></extra>' # Display only 0 or 1
        ))

    fig.update_layout(
        title='Simulated Digital Waveform',
        yaxis=dict(tickvals=list(range(len(traces), 0, -1)), ticktext=list(traces.keys()), showgrid=True, zeroline=False, range=[0.5, len(traces) + 0.5]),
        xaxis=dict(title='Time (simulation steps)', showgrid=False, range=[-0.5, 2**num_vars - 0.5]),
        height=300 + 50 * num_vars,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#e2e8f0"
    )
    return fig

# --- Main Application UI ---
local_css()
st.title("⚡ K-Map Logic Minimizer")

with st.sidebar:
    st.header("⚙️ Controls")
    num_vars = st.selectbox("Number of Variables", options=[2, 3, 4, 5], index=2, key="num_vars_select")
    if st.button("Reset K-Map"):
        for i in range(2**num_vars): st.session_state[f'kmap_cell_{i}'] = '0'
        st.rerun()

for i in range(32):
    if f'kmap_cell_{i}' not in st.session_state: st.session_state[f'kmap_cell_{i}'] = '0'

st.header("Interactive Karnaugh Map")
variables = ['A', 'B', 'C', 'D', 'E'][:num_vars]

def draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars, offset=0):
    top_labels = st.columns([1.5, len(cols)], gap="small") 
    with top_labels[0]:
        st.markdown(f"<div class='axis-label' style='text-align:right; padding-right:1em;'>{row_vars} \\ {col_vars}</div>", unsafe_allow_html=True)
    with top_labels[1]:
        col_headers = st.columns(len(cols), gap="small")
        for i, col_label in enumerate(cols):
            col_headers[i].markdown(f"<div class='axis-label'>{col_label}</div>", unsafe_allow_html=True)
    
    for r_label in rows:
        row_ui_cols = st.columns([1.5] + [1] * len(cols), gap="small") # CLOSER BUTTONS
        row_ui_cols[0].markdown(f"<div class='row-label'>{r_label}</div>", unsafe_allow_html=True)
        
        for c_label in cols:
            prefix = '1' if offset == 16 else '0'
            term_bin = (prefix if num_vars == 5 else "") + r_label + c_label
            term_dec = int(term_bin, 2)
            cell_state = st.session_state.get(f'kmap_cell_{term_dec}', '0')
            button_key = f"btn_{term_dec}"
            
            st.markdown(f"""<style>
                div[data-testid*="stButton"] button[data-testid*="{button_key}"] {{
                    background-color: {'#4ade80' if cell_state == '1' else '#60a5fa' if cell_state == 'x' else '#334155'};
                    color: {'#052e16' if cell_state == '1' else '#1e3a8a' if cell_state == 'x' else '#94a3b8'};
                }}</style>""", unsafe_allow_html=True)

            if row_ui_cols[cols.index(c_label)+1].button(cell_state.upper(), key=button_key, help=f"Minterm {term_dec}"):
                st.session_state[f'kmap_cell_{term_dec}'] = {'0': '1', '1': 'x', 'x': '0'}[cell_state]
                st.rerun()

_, center_col, _ = st.columns([1, 5, 1]) # Wider center column
with center_col:
    rows, cols = get_kmap_indices(num_vars)
    if num_vars == 2: row_vars, col_vars = "A", "B"
    elif num_vars == 3: row_vars, col_vars = "A", "BC"
    elif num_vars == 4: row_vars, col_vars = "AB", "CD"
    else: row_vars, col_vars = "BC", "DE"

    if num_vars == 5:
        st.subheader(f"A = 0"); draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars, offset=0)
        st.subheader(f"A = 1"); draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars, offset=16)
    else: draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars)

st.write("---")

minterms = [i for i in range(2**num_vars) if st.session_state.get(f'kmap_cell_{i}') == '1']
dont_cares = [i for i in range(2**num_vars) if st.session_state.get(f'kmap_cell_{i}') == 'x']

if not minterms and not dont_cares:
    st.info("Click on the K-Map cells above to define the boolean function.")
else:
    st.header("Results")
    
    # --- SIMULTANEOUS SOP AND POS CALCULATION ---
    # SOP
    sop_pi = find_prime_implicants(set(minterms) | set(dont_cares), num_vars)
    sop_cover = get_minimal_cover(sop_pi, set(minterms), num_vars)
    sop_expression = generate_full_expression(sop_cover, variables, is_pos=False)
    
    # POS
    zeros = set(range(2**num_vars)) - set(minterms) - set(dont_cares)
    pos_pi = find_prime_implicants(zeros | set(dont_cares), num_vars)
    pos_cover = get_minimal_cover(pos_pi, zeros, num_vars)
    pos_expression = generate_full_expression(pos_cover, variables, is_pos=True)

    tab1, tab2, tab3 = st.tabs(["Minimized Expressions", "Optimized Verilog", "Waveform"])
    
    with tab1:
        st.subheader("Sum of Products (SOP)")
        st.latex(f"F_{{SOP}} = {sop_expression.replace('*', ' \\cdot ').replace('+', ' + ').replace('()', '0').replace('\'', '^{\\prime}')}")
        
        st.subheader("Product of Sums (POS)")
        st.latex(f"F_{{POS}} = {pos_expression.replace('*', ' \\cdot ').replace('+', ' + ').replace('()', '1').replace('\'', '^{\\prime}')}")
        
    with tab2:
        # CHOOSE SHORTER EXPRESSION FOR VERILOG
        use_pos_for_verilog = len(pos_expression) < len(sop_expression)
        chosen_expr = pos_expression if use_pos_for_verilog else sop_expression
        form_name = "Product of Sums (POS)" if use_pos_for_verilog else "Sum of Products (SOP)"
        
        st.info(f"💡 Generating Verilog based on the shorter expression: **{form_name}**")
        st.code(generate_verilog(chosen_expr, variables), language='verilog')
        st.code(generate_testbench(variables), language='verilog')

    with tab3:
        st.subheader("Simulated Waveform")
        try:
            fig = generate_waveform(st.session_state, num_vars)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e: 
            st.error(f"Could not generate waveform: {e}")

