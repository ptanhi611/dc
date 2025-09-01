import streamlit as st
import pandas as pd
from itertools import combinations

# --- Helper Functions ---

def gray_code(n):
    """Generates Gray code for n bits."""
    if n == 0:
        return ['']
    first_half = gray_code(n - 1)
    second_half = first_half[::-1]
    return ['0' + code for code in first_half] + ['1' + code for code in second_half]

def get_kmap_coords(num_vars):
    """Generates row and column labels for the K-map display."""
    if num_vars == 2:
        rows, cols = gray_code(1), gray_code(1)
    elif num_vars == 3:
        rows, cols = gray_code(1), gray_code(2)
    elif num_vars == 4:
        rows, cols = gray_code(2), gray_code(2)
    elif num_vars == 5:
        rows, cols = gray_code(2), gray_code(3)
    else:
        raise ValueError("Number of variables must be between 2 and 5")
    return rows, cols

def bin_to_gray(binary):
    """Converts a binary string to a Gray code string."""
    gray = binary[0]
    for i in range(1, len(binary)):
        gray += str(int(binary[i-1]) ^ int(binary[i]))
    return gray

def dec_to_bin(dec, num_vars):
    """Converts a decimal number to a binary string of length num_vars."""
    return format(dec, f'0{num_vars}b')

# --- Quine-McCluskey Minimization Algorithm ---

def combine_terms(term1, term2):
    """Combines two minterms if they differ by exactly one bit."""
    diff_count = 0
    new_term = ""
    for c1, c2 in zip(term1, term2):
        if c1 != c2:
            diff_count += 1
            new_term += '-'
        else:
            new_term += c1
    return new_term if diff_count == 1 else None

def find_prime_implicants(minterms, num_vars):
    """Finds all prime implicants for a given set of minterms."""
    if not minterms:
        return set()

    # Convert minterms to binary representation
    terms = {dec_to_bin(m, num_vars) for m in minterms}
    prime_implicants = set()
    
    while True:
        new_terms = set()
        uncombined = set(terms)
        
        for t1, t2 in combinations(terms, 2):
            combined = combine_terms(t1, t2)
            if combined:
                new_terms.add(combined)
                uncombined.discard(t1)
                uncombined.discard(t2)
        
        prime_implicants.update(uncombined)
        
        if not new_terms or terms == new_terms:
            break
        terms = new_terms
        
    return prime_implicants

def get_essential_prime_implicants(prime_implicants, minterms, num_vars):
    """Finds essential prime implicants and a minimal cover using a prime implicant chart."""
    if not minterms:
        return []
    if not prime_implicants:
        return []

    minterm_bins = {dec_to_bin(m, num_vars) for m in minterms}
    
    # Create the prime implicant chart
    chart = {pi: [] for pi in prime_implicants}
    for pi in prime_implicants:
        for mt in minterm_bins:
            # Check if minterm is covered by the prime implicant
            is_covered = all((p == m or p == '-') for p, m in zip(pi, mt))
            if is_covered:
                chart[pi].append(mt)
    
    essential_pis = set()
    covered_minterms = set()
    
    # Identify essential PIs
    minterm_to_pis = {mt: [] for mt in minterm_bins}
    for pi, mts in chart.items():
        for mt in mts:
            minterm_to_pis[mt].append(pi)
            
    for mt, pis in minterm_to_pis.items():
        if len(pis) == 1:
            essential_pi = pis[0]
            essential_pis.add(essential_pi)
            covered_minterms.update(chart[essential_pi])

    # Cover remaining minterms with a minimal set of PIs
    uncovered_minterms = minterm_bins - covered_minterms
    remaining_pis = {pi for pi in prime_implicants if pi not in essential_pis}
    
    # Greedy approach to cover remaining minterms
    final_cover = set(essential_pis)
    while uncovered_minterms:
        best_pi = max(remaining_pis, key=lambda pi: len(set(chart[pi]) & uncovered_minterms), default=None)
        if not best_pi:
            break
        final_cover.add(best_pi)
        uncovered_minterms -= set(chart[best_pi])
        remaining_pis.remove(best_pi)
        
    return list(final_cover)

def term_to_expression(term, variables, is_pos):
    """Converts a binary term to a boolean expression part."""
    parts = []
    for i, char in enumerate(term):
        if char == '0':
            parts.append(variables[i] if is_pos else variables[i] + "'")
        elif char == '1':
            parts.append(variables[i] + "'" if is_pos else variables[i])
    
    if is_pos:
        return f"({' + '.join(parts)})"
    return ''.join(parts)

# --- UI and Main Application Logic ---

st.set_page_config(layout="wide", page_title="K-Map Solver", page_icon="🔢")

st.title("🔢 Karnaugh Map (K-Map) Solver")
st.markdown("An interactive tool to minimize boolean functions, visualize K-maps, and generate Verilog code.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    num_vars = st.slider("Number of Variables", 2, 5, 4, key="num_vars")
    solve_for = st.radio("Solve for", ('Sum of Products (SOP)', 'Product of Sums (POS)'), key="solve_for")
    
    is_pos = (solve_for == 'Product of Sums (POS)')
    term_label = "Maxterms" if is_pos else "Minterms"
    
    st.markdown(f"---")
    st.subheader(f"Enter {term_label}")
    input_terms_str = st.text_area(
        f"Enter {term_label} separated by commas (e.g., 0, 1, 4, 5)",
        key="input_terms_str"
    )
    
    solve_button = st.button("✨ Solve")

# --- Main Content Area ---
st.header("Results")

if solve_button:
    try:
        # 1. Parse Input
        if not input_terms_str:
            st.warning(f"Please enter the {term_label}.")
            st.stop()
            
        input_terms = sorted(list(set(int(i.strip()) for i in input_terms_str.split(','))))
        max_term = 2**num_vars - 1
        if any(t > max_term for t in input_terms):
            st.error(f"Invalid input: Terms must be between 0 and {max_term} for {num_vars} variables.")
            st.stop()
        
        # Define terms to solve for based on SOP/POS
        total_terms = set(range(2**num_vars))
        terms_to_solve = set(input_terms)
        if is_pos: # For POS, we find the SOP of the OFF-set (0s)
            terms_to_solve = total_terms - terms_to_solve
        
        col1, col2 = st.columns(2)

        with col1:
            # 2. Display K-Map
            st.subheader("Karnaugh Map")
            
            rows, cols = get_kmap_coords(num_vars)
            
            if num_vars <= 4:
                df = pd.DataFrame(index=rows, columns=cols)
                for r_idx, r in enumerate(rows):
                    for c_idx, c in enumerate(cols):
                        if num_vars == 2:
                            term_bin = r + c
                        elif num_vars == 3:
                            term_bin = r + c
                        else: # 4 vars
                             term_bin = r + c
                        
                        term_dec = int(term_bin, 2)
                        value = '1' if term_dec in input_terms else '0'
                        df.iloc[r_idx, c_idx] = value
                
                st.table(df)

            else: # 5 variables
                st.markdown("**For V=0**")
                df0 = pd.DataFrame(index=gray_code(2), columns=gray_code(3))
                st.markdown("**For V=1**")
                df1 = pd.DataFrame(index=gray_code(2), columns=gray_code(3))

                for term in range(32):
                    term_bin = dec_to_bin(term, 5)
                    v, r, c = term_bin[0], term_bin[1:3], term_bin[3:5]
                    r_idx, c_idx = gray_code(2).index(r), gray_code(2).index(c) # Re-adjusting for 4x4 display per variable
                    
                    if len(gray_code(3)) > 4: # Handle 5-var column layout
                         c_idx = gray_code(3).index(term_bin[2:])
                         r_idx = gray_code(2).index(term_bin[0:2])

                    value = '1' if term in input_terms else '0'
                    if v == '0':
                         df0.iloc[r_idx, c_idx] = value
                    else:
                         df1.iloc[r_idx, c_idx] = value
                
                st.markdown("**Map for A=0**")
                st.table(df0)
                st.markdown("**Map for A=1**")
                st.table(df1)


        with col2:
            # 3. Minimize and Display Expression
            st.subheader("Minimized Expression")
            
            if not terms_to_solve: # Handle edge cases (all 0s or all 1s)
                result_expression = "0" if not is_pos else "1"
            else:
                prime_implicants = find_prime_implicants(terms_to_solve, num_vars)
                minimal_cover = get_essential_prime_implicants(prime_implicants, terms_to_solve, num_vars)
                
                variables = ['A', 'B', 'C', 'D', 'E'][:num_vars]
                
                if not minimal_cover:
                     result_expression = "1" if not is_pos else "0"
                else:
                    expression_parts = [term_to_expression(term, variables, is_pos) for term in sorted(minimal_cover)]
                    joiner = " * " if is_pos else " + "
                    result_expression = joiner.join(expression_parts)

            st.latex(f"F = {result_expression.replace('*', ' \\cdot ').replace('+', ' + ').replace('\'', '^{\'}')}")


            # 4. Generate and Display Verilog Code
            st.subheader("Verilog Code")
            variables_str = ", ".join(variables)
            verilog_expression = result_expression.replace("'", "~").replace("+", "|").replace("*", "&").replace(" ", "")
            
            verilog_code = f"""
module minimized_logic(
    input  {variables_str},
    output F
);

assign F = {verilog_expression};

endmodule
"""
            st.code(verilog_code, language='verilog')

    except ValueError:
        st.error("Invalid input. Please ensure you enter comma-separated numbers only.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
