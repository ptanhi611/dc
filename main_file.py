import streamlit as st
from itertools import combinations
import plotly.graph_objects as go
import base64

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="K-Map Logic Minimizer", page_icon="⚡", initial_sidebar_state="expanded")

# --- Custom Design ---
def local_css():
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Roboto+Mono:wght@700&display=swap');
        
        /* DARK THEME */
        .stApp {{
            background-color: #0f172a; /* Dark blue background */
        }}
        
        .main .block-container {{ padding-top: 2rem; padding-left: 2rem; padding-right: 2rem;}}
        
        /* K-MAP BUTTONS: TRULY HUMONGOUS AND BOLD */
        .stButton>button {{
            width: 200%;
            height: 18em; /* ENORMOUS HEIGHT */
            font-size: 7em; /* ABSOLUTELY MASSIVE FONT */
            font-weight: 700; /* BOLDER FONT */
            font-family: 'Roboto Mono', monospace;
            border-radius: 24px; /* Larger radius for larger button */
            border: 6px solid #475569; /* Much Thicker border */
            transition: all 0.25s ease-in-out;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .stButton>button:hover {{
            border-color: #7dd3fc; /* Lighter, more vibrant blue */
            transform: translateY(-8px) scale(1.04); /* More dramatic lift */
            box-shadow: 0 12px 35px rgba(125, 211, 252, 0.4);
        }}

        /* Make the Reset button smaller */
        [data-testid="stSidebar"] .stButton>button {{
             height: 3em;
             font-size: 1em;
             border-width: 2px;
        }}
        
        /* Headings and Text */
        h1, h2, h3, .stMarkdown {{
            color: #e2e8f0; /* Light text */
            font-family: 'Poppins', sans-serif;
        }}
        h1, h2, h3 {{ text-align: center; }}
        
        /* K-MAP AXIS LABELS: Rebuilt for perfect alignment */
        .label-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
        }}
        .axis-label {{
            font-size: 2em; font-weight: 600; font-family: 'Poppins', sans-serif;
            color: #7dd3fc;
        }}
        .gray-code-label {{
             font-size: 1.8em; font-weight: 700; font-family: 'Roboto Mono', monospace;
             color: #94a3b8;
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
        if c1 != c2: diff_count += 1; new_term += '-'
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
            if combined: new_terms.add(combined); uncombined.discard(t1); uncombined.discard(t2)
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
            essential_pi = pis[0]; essential_pis.add(essential_pi)
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
    if not cover: return "1" if is_pos else "0"
    expr_parts = [term_to_expression(term, variables, is_pos) for term in sorted(cover)]
    if not expr_parts: return "1" if is_pos else "0"
    return (" * ").join(expr_parts) if is_pos else (" + ").join(expr_parts)

# --- Verilog and Waveform Generation ---
def generate_verilog(expression, variables):
    verilog_expr = expression.replace("'", "~").replace("+", "|").replace("*", "&").replace(" ", "")
    if not verilog_expr or verilog_expr == "()": verilog_expr = "1'b0"

    lines = [f"// Synthesizable module for: F = {expression}", "`timescale 1ns/1ps",
        "module minimized_logic(", f"    input  {', '.join(variables)},", "    output F", ");", "",
        f"    assign F = {verilog_expr};", "", "endmodule"]
    return "\n".join(lines)

def generate_testbench(variables):
    num_vars = len(variables)
    dut_ports = ",\n        ".join([f".{v}(i[{num_vars-1-idx}])" for idx, v in enumerate(variables)])
    lines = ["// Testbench to verify the logic", "module tb_minimized_logic;", "",
        f"    reg [{num_vars-1}:0] i;", "    wire F;", "", "    minimized_logic dut (",
        f"        {dut_ports}", "    );", "", "    initial begin", "        i = 0;",
        "        $dumpfile(\"waveform.vcd\");", "        $dumpvars(0, tb_minimized_logic);",
        f"        for (i = 0; i < {2**num_vars}; i = i + 1) begin", "            #10;",
        "        end", "        $finish;", "    end", "endmodule"]
    return "\n".join(lines)

def generate_waveform(kmap_states, num_vars):
    traces = {var: [] for var in ['A', 'B', 'C', 'D', 'E'][:num_vars] + ['F']}
    for i in range(2**num_vars):
        input_bin = dec_to_bin(i, num_vars)
        for j, var_name in enumerate(traces.keys()):
            if var_name != 'F': traces[var_name].append(int(input_bin[j]))
        cell_val = kmap_states.get(f'kmap_cell_{i}', '0')
        output = 1 if cell_val == '1' else 0
        traces['F'].append(output)

    fig = go.Figure()
    time_steps = list(range(2**num_vars))
    for i, (signal, values) in enumerate(traces.items()):
        y_pos = len(traces) - i
        x_coords = [t for t in time_steps for _ in (0, 1)]; y_coords = [y_pos + 0.4 * val for val in values for _ in (0, 1)]
        hover_values = [v for v in values for _ in (0, 1)]
        fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines', name=signal,
            line=dict(shape='hv', width=4 if signal == 'F' else 2, color='#38bdf8' if signal != 'F' else '#f97316'),
            customdata=hover_values, hovertemplate='<b>Value</b>: %{customdata}<extra></extra>'))
    fig.update_layout(title='Simulated Digital Waveform',
        yaxis=dict(tickvals=list(range(len(traces), 0, -1)), ticktext=list(traces.keys()), showgrid=True, zeroline=False, range=[0.5, len(traces) + 0.5]),
        xaxis=dict(title='Time (simulation steps)', showgrid=False, range=[-0.5, 2**num_vars - 0.5]),
        height=300 + 50 * num_vars, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#e2e8f0")
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

# --- NEW, ROBUST GRID DRAWING FUNCTION ---
def draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars, offset=0):
    # This function uses more lines of code to be more explicit and guarantee alignment.
    # It creates a clear structure of containers for perfect positioning.

    # TOP LABELS ROW (Column variables and gray codes)
    # The first item is a spacer to align with the row-label column
    top_label_cols = st.columns([1.5] + [1] * len(cols), gap="small")
    
    with top_label_cols[0]:
        # Top-left corner label for variable names
        st.markdown(f"""
            <div class="label-container" style="justify-content: flex-end; padding-right: 1em;">
                <span class="axis-label">{row_vars} \ {col_vars}</span>
            </div>
        """, unsafe_allow_html=True)

    for i, col_label in enumerate(cols):
        with top_label_cols[i + 1]:
            st.markdown(f"<div class='label-container'><span class='gray-code-label'>{col_label}</span></div>", unsafe_allow_html=True)

    # K-MAP ROWS (Row gray codes and buttons)
    for r_label in rows:
        grid_row_cols = st.columns([1.5] + [1] * len(cols), gap="small")
        
        # First column is for the row gray code label
        with grid_row_cols[0]:
            st.markdown(f"<div class='label-container' style='justify-content: flex-end; padding-right: 1em;'><span class='gray-code-label'>{r_label}</span></div>", unsafe_allow_html=True)
        
        # Subsequent columns are the clickable buttons
        for i, c_label in enumerate(cols):
            with grid_row_cols[i + 1]:
                prefix = '1' if offset == 16 else '0'
                term_bin = (prefix if num_vars == 5 else "") + r_label + c_label
                term_dec = int(term_bin, 2)
                cell_state = st.session_state.get(f'kmap_cell_{term_dec}', '0')
                button_key = f"btn_{term_dec}"
                
                st.markdown(f"""<style>
                    div[data-testid*="stButton"] button[data-testid*="{button_key}"] {{
                        background-color: {'#22c55e' if cell_state == '1' else '#f59e0b' if cell_state == 'x' else '#475569'};
                        color: {'#ffffff' if cell_state == '1' else '#ffffff' if cell_state == 'x' else '#cbd5e1'};
                    }}</style>""", unsafe_allow_html=True)

                if st.button(cell_state.upper(), key=button_key, help=f"Minterm {term_dec}"):
                    st.session_state[f'kmap_cell_{term_dec}'] = {'0': '1', '1': 'x', 'x': '0'}[cell_state]
                    st.rerun()

_, center_col, _ = st.columns([1, 10, 1]) # Even wider center column for the massive buttons
with center_col:
    rows, cols = get_kmap_indices(num_vars)
    if num_vars == 2: row_vars, col_vars = "A", "B"
    elif num_vars == 3: row_vars, col_vars = "A", "BC"
    elif num_vars == 4: row_vars, col_vars = "AB", "CD"
    else: row_vars, col_vars = "BC", "DE" # A is the 5th variable

    if num_vars == 5:
        st.subheader(f"A = 0"); draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars, offset=0)
        st.markdown("<br>", unsafe_allow_html=True) # Add space between maps
        st.subheader(f"A = 1"); draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars, offset=16)
    else: draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars)

st.write("---")
minterms = [i for i in range(2**num_vars) if st.session_state.get(f'kmap_cell_{i}') == '1']
dont_cares = [i for i in range(2**num_vars) if st.session_state.get(f'kmap_cell_{i}') == 'x']

if not minterms and not dont_cares:
    st.info("Click on the K-Map cells above to define the boolean function.")
else:
    st.header("Results")
    
    sop_pi = find_prime_implicants(set(minterms) | set(dont_cares), num_vars)
    sop_cover = get_minimal_cover(sop_pi, set(minterms), num_vars)
    sop_expression = generate_full_expression(sop_cover, variables, is_pos=False)
    
    zeros = set(range(2**num_vars)) - set(minterms) - set(dont_cares)
    pos_pi = find_prime_implicants(zeros | set(dont_cares), num_vars)
    pos_cover = get_minimal_cover(pos_pi, zeros, num_vars)
    pos_expression = generate_full_expression(pos_cover, variables, is_pos=True)

    tab1, tab2, tab3 = st.tabs(["Minimized Expressions", "Optimized Verilog", "Waveform"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sum of Products (SOP)")
            st.latex(f"F = {sop_expression.replace('*', ' \\cdot ').replace('+', ' + ').replace('()', '0').replace('\'', '^{\\prime}')}")
        with col2:
            st.subheader("Product of Sums (POS)")
            st.latex(f"F = {pos_expression.replace('*', ' \\cdot ').replace('+', ' + ').replace('()', '1').replace('\'', '^{\\prime}')}")
        
    with tab2:
        use_pos = len(pos_expression.replace(" ", "")) < len(sop_expression.replace(" ", ""))
        chosen_expr = pos_expression if use_pos else sop_expression
        form_name = "Product of Sums (POS)" if use_pos else "Sum of Products (SOP)"
        st.info(f"💡 Generating Verilog based on the shorter expression: **{form_name}**")
        st.code(generate_verilog(chosen_expr, variables), language='verilog')
        st.code(generate_testbench(variables), language='verilog')

    with tab3:
        st.subheader("Simulated Waveform")
        try:
            fig = generate_waveform(st.session_state, num_vars)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e: st.error(f"Could not generate waveform: {e}")

