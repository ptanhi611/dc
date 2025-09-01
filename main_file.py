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
            height: 5em;
            font-size: 1.2em;
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
        .css-1d391kg { /* Main content area */
            padding-top: 2rem;
        }
        h1, h2, h3 {
            color: #eee;
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
    # Convert Python boolean expression to Verilog syntax
    verilog_expr = expression.replace("'", "~").replace("+", "|").replace("*", "&").replace(" ", "")
    if not verilog_expr:
        verilog_expr = "1'b0" if is_pos else "1'b1"


    # Define the Verilog module using an f-string
    module_code = f"""
`timescale 1ns/1ps
module minimized_logic(
    input  {variables_str},
    output F
);

    assign F = {verilog_expr};

endmodule
"""
    return module_code

def generate_testbench(variables):
    """Generates the Verilog testbench."""
    num_vars = len(variables)
    
    # Create the DUT port connections dynamically for better readability
    port_connections = [f".{variables[0]}(i[{num_vars-1}])"]
    port_connections.extend([f".{v}(i[{num_vars-1-idx}])" for idx, v in enumerate(variables[1:])])
    dut_ports = ",\n        ".join(port_connections)

    # Define the testbench module using an f-string
    tb_code = f"""
module tb_minimized_logic;

    // Inputs
    reg [{num_vars-1}:0] i;

    // Outputs
    wire F;

    // Instantiate the Unit Under Test (UUT)
    minimized_logic dut (
        {dut_ports}
    );

    initial begin
        // Initialize Inputs
        i = 0;

        // Setup waveform dump
        $dumpfile("waveform.vcd");
        $dumpvars(0, tb_minimized_logic);
        
        // Loop through all possible input combinations
        for (i = 0; i < {2**num_vars}; i = i + 1) begin
            #10; // Wait 10 time units per combination
        end

        // Finish simulation
        $finish;
    end
      
endmodule
"""
    return tb_code

def generate_waveform(minimized_cover, variables, is_pos):
    num_vars = len(variables)
    traces = {var: [] for var in variables + ['F']}
    
    for i in range(2**num_vars):
        input_bin = dec_to_bin(i, num_vars)
        for j, var in enumerate(variables):
            traces[var].append(int(input_bin[j]))
        
        # Calculate output F
        output_val = 1 if is_pos else 0
        if not minimized_cover:
             output_val = 1 if is_pos else 0
        else:
            if is_pos: # Product of Sums
                output_val = 1
                for term in minimized_cover:
                    # An OR term is 0 only if all its literals are 0
                    term_val = 0 
                    for k, char in enumerate(term):
                        if (char == '0' and input_bin[k] == '0') or \
                           (char == '1' and input_bin[k] == '1'):
                            term_val = 1 # if any part of the sum term is 1, the sum is 1
                            break
                    if term_val == 0: # if any sum term is 0, the final product is 0
                        output_val = 0
                        break
            else: # Sum of Products
                output_val = 0
                for term in minimized_cover:
                    # An AND term is 1 only if all its literals are 1
                    term_val = 1
                    for k, char in enumerate(term):
                        if (char == '0' and input_bin[k] == '1') or \
                           (char == '1' and input_bin[k] == '0'):
                            term_val = 0
                            break
                    if term_val == 1: # if any product term is 1, the final sum is 1
                        output_val = 1
                        break
        
        traces['F'].append(output_val)

    fig = go.Figure()
    time_steps = list(range(2**num_vars + 1))
    
    for i, (signal, values) in enumerate(traces.items()):
        y_pos = len(traces) - i
        # Create stepped line plot
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

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    num_vars = st.slider("Number of Variables", 2, 5, 4, key="num_vars_slider")
    solve_for = st.radio("Solve for", ('Sum of Products (SOP)', 'Product of Sums (POS)'), key="solve_for")
    
    if st.button("Reset K-Map"):
        for i in range(2**num_vars):
            st.session_state[f'kmap_cell_{i}'] = '0'
        st.experimental_rerun()

# Initialize session state for K-map cells
for i in range(32): # Max cells needed is for 5 variables (2**5)
    if f'kmap_cell_{i}' not in st.session_state:
        st.session_state[f'kmap_cell_{i}'] = '0'

# --- Interactive K-Map ---
st.header("Interactive Karnaugh Map")
is_pos = (solve_for == 'POS')
variables = ['A', 'B', 'C', 'D', 'E'][:num_vars]

def draw_kmap_grid(rows, cols, offset=0):
    st.write("---")
    header_cols = st.columns([1] + [1] * len(cols))
    header_cols[0].write("")
    for i, col_label in enumerate(cols):
        header_cols[i+1].markdown(f"<h4 style='text-align: center;'>{col_label}</h4>", unsafe_allow_html=True)

    for r_idx, r_label in enumerate(rows):
        row_cols = st.columns([1] + [1] * len(cols))
        row_cols[0].markdown(f"<h4 style='text-align: center; padding-top: 2em;'>{r_label}</h4>", unsafe_allow_html=True)
        for c_idx, c_label in enumerate(cols):
            if num_vars == 5:
                prefix = '1' if offset == 16 else '0'
                term_bin = prefix + r_label + c_label
            else:
                term_bin = r_label + c_label
            term_dec = int(term_bin, 2)
            
            cell_state = st.session_state.get(f'kmap_cell_{term_dec}', '0')
            
            button_key = f"btn_{term_dec}"
            # This is a hack to apply class to button
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


# --- Display K-Map(s) ---
rows, cols = get_kmap_indices(num_vars)
if num_vars == 5:
    st.subheader(f"{variables[0]} = 0")
    draw_kmap_grid(rows, cols, offset=0)
    st.subheader(f"{variables[0]} = 1")
    draw_kmap_grid(rows, cols, offset=16)
else:
    draw_kmap_grid(rows, cols)

st.write("---")

# --- Process and Display Results ---
minterms = [i for i in range(2**num_vars) if st.session_state.get(f'kmap_cell_{i}') == '1']
dont_cares = [i for i in range(2**num_vars) if st.session_state.get(f'kmap_cell_{i}') == 'x']

if not minterms and not dont_cares:
    st.info("Click on the K-Map cells above to define the boolean function.")
else:
    terms_to_solve = set(minterms)
    terms_for_pi = set(minterms) | set(dont_cares)
    
    if is_pos:  # For POS, we find the SOP of the OFF-set (0s)
        all_terms = set(range(2**num_vars))
        zeros = all_terms - set(minterms) - set(dont_cares)
        terms_to_solve = zeros
        terms_for_pi = zeros | set(dont_cares)
    
    result_expression = ""
    minimized_cover = []
    
    if not terms_to_solve:  # Handle edge case (all 1s/dontcares for SOP, or all 0s/dontcares for POS)
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
    
    # --- Output Tabs ---
    st.header("Results")
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
        if minimized_cover or (not terms_to_solve and (result_expression in ["0", "1"] or not result_expression)):
            fig = generate_waveform(minimized_cover, variables, is_pos)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Cannot generate waveform for an empty expression.")
