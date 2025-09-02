import streamlit as st
import pandas as pd
from itertools import combinations, product
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Interactive K-Map Solver", page_icon="⚡")

# --- Custom CSS for Styling ---
def local_css():
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            height: 4em; /* Made buttons slightly shorter for a compact grid */
            font-size: 1.3em; /* Increased font size for bigger look */
            font-weight: bold;
            border-radius: 10px;
            border: 2px solid #555;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            border-color: #00bfff;
            transform: scale(1.05);
        }
        .kmap-cell-0 {
            background-color: #e0e0e0;
            color: #333;
        }
        .kmap-cell-1 {
            background-color: #90ee90; /* Light Green */
            color: #006400; /* Dark Green */
        }
        .kmap-cell-x {
            background-color: #add8e6; /* Light Blue */
            color: #00008b; /* Dark Blue */
        }
        .main-container {
            padding: 0 4rem; /* Add padding to the sides */
        }
        h1, h2, h3 {
            color: #eee;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
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

# --- Quine-McCluskey Minimization Algorithm (with Don't Cares) ---
def combine_terms(term1, term2):
    diff_count, new_term = 0, ""
    for c1, c2 in zip(term1, term2):
        if c1 != c2:
            diff_count += 1
            new_term += '-'
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
                uncombined.discard(t1)
                uncombined.discard(t2)
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
    return f"({joiner.join(parts)})" if is_pos else "".join(parts)

# --- Verilog and Waveform Generation ---
def generate_verilog(expression, variables, is_pos):
    """Generates the synthesizable Verilog module."""
    variables_str = ", ".join(variables)
    verilog_expr = expression.replace("'", "~").replace("+", "|").replace("*", "&").replace(" ", "")
    if not verilog_expr:
        verilog_expr = "1'b0" if is_pos else "1'b1"

    lines = [
        "`timescale 1ns/1ps",
        "module minimized_logic(",
        f"    input  {variables_str},",
        "    output F",
        ");",
        "",
        f"    assign F = {verilog_expr};",
        "",
        "endmodule"
    ]
    return "\n".join(lines)

def generate_testbench(variables):
    """Generates the Verilog testbench."""
    num_vars = len(variables)
    port_connections = [f".{variables[0]}(i[{num_vars-1}])"]
    port_connections.extend([f".{v}(i[{num_vars-1-idx}])" for idx, v in enumerate(variables[1:])])
    dut_ports = ",\n        ".join(port_connections)

    lines = [
        "module tb_minimized_logic;",
        "",
        "    // Inputs",
        f"    reg [{num_vars-1}:0] i;",
        "",
        "    // Outputs",
        "    wire F;",
        "",
        "    // Instantiate the Unit Under Test (UUT)",
        "    minimized_logic dut (",
        f"        {dut_ports}",
        "    );",
        "",
        "    initial begin",
        "        // Initialize Inputs",
        "        i = 0;",
        "",
        "        // Setup waveform dump",
        "        $dumpfile(\"waveform.vcd\");",
        "        $dumpvars(0, tb_minimized_logic);",
        "        ",
        "        // Loop through all possible input combinations",
        f"        for (i = 0; i < {2**num_vars}; i = i + 1) begin",
        "            #10; // Wait 10 time units per combination",
        "        end",
        "",
        "        // Finish simulation",
        "        $finish;",
        "    end",
        "endmodule"
    ]
    return "\n".join(lines)

def generate_waveform(minimized_cover, variables, is_pos):
    num_vars = len(variables)
    traces = {var: [] for var in variables + ['F']}
    
    for i in range(2**num_vars):
        input_bin = dec_to_bin(i, num_vars)
        for j, var in enumerate(variables):
            traces[var].append(int(input_bin[j]))
        
        output_val = 0
        if is_pos: # Product of Sums
            output_val = 1
            if minimized_cover:
                for term in minimized_cover:
                    term_val = 0 
                    for k, char in enumerate(term):
                        if (char == '0' and input_bin[k] == '0') or \
                           (char == '1' and input_bin[k] == '1'):
                            term_val = 1
                            break
                    if term_val == 0:
                        output_val = 0
                        break
        else: # Sum of Products
            output_val = 0
            if minimized_cover:
                for term in minimized_cover:
                    term_val = 1
                    for k, char in enumerate(term):
                        if (char == '0' and input_bin[k] == '1') or \
                           (char == '1' and input_bin[k] == '0'):
                            term_val = 0
                            break
                    if term_val == 1:
                        output_val = 1
                        break
        
        traces['F'].append(output_val)

    fig = go.Figure()
    time_steps = list(range(2**num_vars + 1))
    
    for i, (signal, values) in enumerate(traces.items()):
        y_pos = len(traces) - i
        x_coords = [t for t in time_steps for _ in (0, 1)][1:-1]
        y_coords = [y_pos + 0.4 * val for val in values for _ in (0, 1)]
        fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines', name=signal,
                                 line=dict(shape='hv', width=2)))

    fig.update_layout(
        title='Digital Waveform',
        yaxis=dict(
            tickvals=list(range(len(traces), 0, -1)),
            ticktext=list(traces.keys()),
            showgrid=True,
            zeroline=False,
            range=[0.5, len(traces) + 0.5]
        ),
        xaxis=dict(title='Time (simulation steps)'),
        height=300 + 50 * len(variables),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Main Application UI ---
local_css()
st.title("⚡ Interactive K-Map Solver & Verilog Generator")

with st.sidebar:
    st.header("⚙️ Controls")
    num_vars = st.selectbox(
        "Number of Variables", 
        options=[2, 3, 4, 5], 
        index=2, 
        key="num_vars_select"
    )
    solve_for = st.radio("Solve for", ('Sum of Products (SOP)', 'Product of Sums (POS)'), key="solve_for")
    
    if st.button("Reset K-Map"):
        for i in range(2**num_vars):
            st.session_state[f'kmap_cell_{i}'] = '0'
        st.rerun()

for i in range(32):
    if f'kmap_cell_{i}' not in st.session_state:
        st.session_state[f'kmap_cell_{i}'] = '0'

st.header("Interactive Karnaugh Map")

is_pos = (solve_for == 'POS')
variables = ['A', 'B', 'C', 'D', 'E'][:num_vars]

def draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars, offset=0):
    # --- Grid Labels ---
    label_cols = st.columns([1, len(cols)])
    with label_cols[0]:
        st.markdown(f"<div style='text-align:right; padding-top:2.5em; font-weight:bold; color: #00bfff;'>{row_vars} \\</div>", unsafe_allow_html=True)
    with label_cols[1]:
        st.markdown(f"<div style='text-align:center; font-weight:bold; color: #00bfff;'>{col_vars}</div>", unsafe_allow_html=True)
    
    # --- Grid Body ---
    for r_idx, r_label in enumerate(rows):
        row_cols = st.columns([1] + [1] * len(cols))
        row_cols[0].markdown(f"<div style='text-align: right; margin-top: 1.5em; font-weight:bold;'>{r_label}</div>", unsafe_allow_html=True)
        
        for c_idx, c_label in enumerate(cols):
            if num_vars == 5:
                prefix = '1' if offset == 16 else '0'
                term_bin = prefix + r_label + c_label
            else:
                term_bin = r_label + c_label
            term_dec = int(term_bin, 2)
            
            cell_state = st.session_state.get(f'kmap_cell_{term_dec}', '0')
            
            button_key = f"btn_{term_dec}"
            st.markdown(f"""
            <style>
                div[data-testid*="stHorizontalBlock"] div[data-testid*="stButton"] button[data-testid*="{button_key}"] {{
                    background-color: {'#90ee90' if cell_state == '1' else '#add8e6' if cell_state == 'x' else '#e0e0e0'};
                    color: {'#006400' if cell_state == '1' else '#00008b' if cell_state == 'x' else '#333'};
                }}
            </style>
            """, unsafe_allow_html=True)

            if row_cols[c_idx+1].button(cell_state.upper(), key=button_key, help=f"Minterm {term_dec}"):
                current_state = st.session_state[f'kmap_cell_{term_dec}']
                next_state = {'0': '1', '1': 'x', 'x': '0'}[current_state]
                st.session_state[f'kmap_cell_{term_dec}'] = next_state
                st.rerun()

# --- Centered Layout for K-Map ---
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    rows, cols = get_kmap_indices(num_vars)
    if num_vars == 2: row_vars, col_vars = "A", "B"
    elif num_vars == 3: row_vars, col_vars = "A", "BC"
    elif num_vars == 4: row_vars, col_vars = "AB", "CD"
    else: row_vars, col_vars = "BC", "DE"

    if num_vars == 5:
        st.subheader(f"A = 0")
        draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars, offset=0)
        st.subheader(f"A = 1")
        draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars, offset=16)
    else:
        draw_kmap_grid(rows, cols, row_vars, col_vars, num_vars)

st.write("---")

minterms = [i for i in range(2**num_vars) if st.session_state.get(f'kmap_cell_{i}') == '1']
dont_cares = [i for i in range(2**num_vars) if st.session_state.get(f'kmap_cell_{i}') == 'x']

if not minterms and not dont_cares:
    st.info("Click on the K-Map cells above to define the boolean function.")
else:
    st.header("Results")
    terms_to_solve = set(minterms)
    terms_for_pi = set(minterms) | set(dont_cares)
    
    if is_pos:
        all_terms = set(range(2**num_vars))
        zeros = all_terms - set(minterms) - set(dont_cares)
        terms_to_solve = zeros
        terms_for_pi = zeros | set(dont_cares)
    
    result_expression = ""
    minimized_cover = []
    
    if not terms_to_solve:
         result_expression = "0" if is_pos else "1"
    else:
        prime_implicants = find_prime_implicants(terms_for_pi, num_vars)
        minimized_cover = get_minimal_cover(prime_implicants, terms_to_solve, num_vars)
        
        if not minimized_cover:
            result_expression = "1" if is_pos else "0"
        else:
            expr_parts = [term_to_expression(term, variables, is_pos) for term in sorted(minimized_cover)]
            joiner = " * " if is_pos else " + "
            result_expression = joiner.join(expr_parts)
    
    tab1, tab2, tab3 = st.tabs(["Minimized Expression", "Verilog Code", "Waveform"])

    with tab1:
        st.subheader("Minimized Boolean Expression")
        latex_expr = result_expression.replace('*', ' \\cdot ').replace("'", "^{\\prime}")
        st.latex(f"F({', '.join(variables)}) = {latex_expr}")

    with tab2:
        st.subheader("Generated Verilog")
        module_code = generate_verilog(result_expression, variables, is_pos)
        tb_code = generate_testbench(variables)
        
        st.code(module_code, language='verilog')
        st.code(tb_code, language='verilog')

    with tab3:
        st.subheader("Simulated Waveform")
        try:
            fig = generate_waveform(minimized_cover, variables, is_pos)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate waveform: {e}")

