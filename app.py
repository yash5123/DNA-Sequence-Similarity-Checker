import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
import base64
import re

# Set page configuration for a centered layout
st.set_page_config(page_title="LCS Visualizer", layout="centered")

def compute_lcs_dp(X: str, Y: str):
    """Computes the Dynamic Programming table, LCS length, and the backtracking path."""
    m, n = len(X), len(Y)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # DP table filling
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    
    # Backtracking to find LCS sequence and path
    i, j = m, n
    lcs_chars = []
    path = []
    while i > 0 and j > 0:
        path.append((i, j))
        if X[i - 1] == Y[j - 1]:
            lcs_chars.append(X[i - 1])
            i -= 1
            j -= 1
        else:
            if dp[i - 1, j] >= dp[i, j - 1]:
                i -= 1
            else:
                j -= 1
    path.append((0, 0))
    lcs = ''.join(reversed(lcs_chars))
    return dp, lcs, list(reversed(path))

def create_styled_dataframe(dp_table, X, Y, highlight_path=None):
    """
    Creates a styled Pandas DataFrame from the DP table, ensuring unique labels 
    to avoid the KeyError: '`Styler.apply` and `.map` are not compatible...'
    """
    # Create unique labels for display: character + its 1-based index (e.g., 'A[1]', 'T[2]')
    unique_row_labels = ['ε'] + [f"{ch}[{i+1}]" for i, ch in enumerate(X)]
    unique_col_labels = ['ε'] + [f"{ch}[{j+1}]" for j, ch in enumerate(Y)]
    
    dp_df = pd.DataFrame(dp_table)
    
    # Apply unique labels to satisfy pandas Styler's internal checks
    dp_df.index = unique_row_labels
    dp_df.columns = unique_col_labels
    
    # Create the style application DataFrame (all empty strings initially)
    style_df = pd.DataFrame('', index=unique_row_labels, columns=unique_col_labels)
    
    # Apply highlighting to the style_df
    if highlight_path:
        for i, j in highlight_path:
            if i > 0 and j > 0:
                # NEW COLOR: Light Purple/Lavender (#e0b0ff) for LCS path highlight
                style_df.iloc[i, j] = 'background-color: #e0b0ff' 
            
    # Apply styling using the unique labels
    styled_df = dp_df.style.apply(lambda x: style_df, axis=None).set_properties(**{'text-align': 'center'})
    
    return styled_df
    
# --- Matplotlib utility functions for Animation ---

def draw_dp_table_frame(dp, X, Y, current_cell=None, highlight_path=None, figsize=(6,6)):
    m, n = len(X), len(Y)
    if m > 10 or n > 10:
        figsize=(8,8) 

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    cell_text = []
    for i in range(0, m+1):
        row = []
        for j in range(0, n+1):
            row.append(str(int(dp[i,j])))
        cell_text.append(row)

    table = ax.table(cellText=cell_text, cellLoc='center', loc='center', bbox=[0.1, 0.1, 0.9, 0.9])
    
    for (row, col), cell in table.get_celld().items():
        cell.set_height(1.0 / (m+2))
        cell.set_width(1.0 / (n+2))
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.get_text().set_ha("center")
        cell.get_text().set_va("center")

    if current_cell is not None:
        i, j = current_cell
        table[i, j].set_facecolor('#fff59d') # Yellow highlight for current processing cell

    if highlight_path:
        for i, j in highlight_path:
            if i > 0 and j > 0:
                # NEW COLOR: Light Purple/Lavender (#e0b0ff) for LCS path highlight
                table[i, j].set_facecolor('#e0b0ff') 

    for i in range(m + 1):
        char = 'ε' if i == 0 else X[i - 1]
        y_pos = 0.9 - i * (0.8 / (m + 1))
        ax.text(0.0, y_pos, str(char), ha='left', va='center', fontsize=12, weight='bold', transform=ax.transAxes)

    for j in range(n + 1):
        char = 'ε' if j == 0 else Y[j - 1]
        x_pos = 0.15 + j * (0.8 / (n + 1))
        ax.text(x_pos, 0.95, str(char), ha='center', va='bottom', fontsize=12, weight='bold', transform=ax.transAxes)

    ax.set_title(f"DP table (X='{X}', Y='{Y}')", pad=12)
    plt.tight_layout()
    return fig

def create_lcs_animation(X, Y, speed_ms=300, max_frames=1000):
    m, n = len(X), len(Y)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    frames = []

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
            
            fig = draw_dp_table_frame(dp, X, Y, current_cell=(i, j), figsize=(6,6)) 
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            frames.append(imageio.v2.imread(buf))
            if len(frames) >= max_frames:
                break
        if len(frames) >= max_frames:
            break
            
    dp_full, lcs, full_path = compute_lcs_dp(X, Y)
    
    # Add frames for backtracking path highlighting
    for k in range(len(full_path)):
        subpath = full_path[:k+1] 
        fig = draw_dp_table_frame(dp_full, X, Y, highlight_path=subpath, figsize=(6,6))
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))
        if len(frames) >= max_frames:
            break
            
    if len(frames) == 0:
        return None, lcs, dp_full
        
    gif_buf = BytesIO()
    duration = max(0.05, speed_ms / 1000.0)
    imageio.mimsave(gif_buf, frames, format='GIF', duration=duration)
    gif_buf.seek(0)
    return gif_buf.getvalue(), lcs, dp_full
 

# --- Streamlit App UI Begins Here ---
st.title("🧬 DNA Sequence Similarity Checker")
st.write("This tool computes the Longest Common Subsequence (LCS) for two strings made of DNA/RNA bases (A, T, G, C) and visualizes the Dynamic Programming (DP) process.")

# Sample data for easy testing
sample_pairs = pd.DataFrame({
    "a": ["ATGCGTAG", "GATTACA", "AGTCAG"],
    "b": ["GTACGTA", "GTCTGA", "GTC"]
})

st.subheader("Input Strings")

st.markdown("**Choose a sample pair** or provide your own DNA/RNA sequences below.")
col1, col2 = st.columns([1, 2])
with col1:
    sample_choice = st.selectbox("Pick sample index (or Manual Input)", ["Manual"] + [f"{i}: {r['a']} | {r['b']}" for i,r in sample_pairs.iterrows()])

if sample_choice == "Manual":
    X_input = st.text_input("String A (DNA 1, only A, T, G, C)", value="ATGCGTAG").upper()
    Y_input = st.text_input("String B (DNA 2, only A, T, G, C)", value="GTACGTA").upper()
    X = X_input
    Y = Y_input
else:
    sel = int(sample_choice.split(":")[0])
    X = sample_pairs.at[sel, "a"]
    Y = sample_pairs.at[sel, "b"]
    st.markdown(f"Selected sample **{sel}**: A = `{X}` , B = `{Y}`")

valid_chars = set(['A', 'T', 'G', 'C'])
is_valid = True
if not all(c in valid_chars for c in X):
    st.error("String A contains invalid characters. Please use only A, T, G, or C.")
    is_valid = False
if not all(c in valid_chars for c in Y):
    st.error("String B contains invalid characters. Please use only A, T, G, or C.")
    is_valid = False

if X and Y and is_valid:
    st.write("---")
    st.subheader("LCS Analysis: Step-by-Step")
    
    # Run the main computation
    dp_table, lcs, lcs_path = compute_lcs_dp(X, Y)
    lcs_length = dp_table[-1,-1]
    
    # --- Step 1: Initial Matrix Setup (with theory) ---
    st.markdown("### Step 1: Initial Matrix Setup (The Base Case)")
    
    dp_empty = np.zeros((len(X) + 1, len(Y) + 1), dtype=int)
    st.dataframe(create_styled_dataframe(dp_empty, X, Y), use_container_width=True)
    
    st.markdown(
        """
        The Dynamic Programming approach requires solving the **smallest subproblems** first. 
        The first row ($i=0$) and first column ($j=0$) represent the LCS when one string is the **empty string** ($\epsilon$). 
        The LCS between any string and an empty string is always **0**, establishing the base case for the recurrence.
        """
    )

    # --- Step 2: Value Filling (with theory) ---
    st.markdown("### Step 2: DP Table Value Filling (The Recursive Step)")
    
    st.markdown(
        """
        Each cell $L[i, j]$ is calculated based on the solutions to smaller subproblems (cells to the left, above, and diagonally up-left). This process uses the principle of **Optimal Substructure**.
        The value in the table represents the length of the LCS for the prefixes $X[1..i]$ and $Y[1..j]$.
        """
    )
    st.markdown("The cells are filled based on the recurrence relation:")
    st.latex(r'''
    L[i, j] = 
    \begin{cases}
    0 & \text{if } i=0 \text{ or } j=0 \\
    L[i-1, j-1] + 1 & \text{if } X[i-1] = Y[j-1] \\
    \max(L[i-1, j], L[i, j-1]) & \text{if } X[i-1] \neq Y[j-1]
    \end{cases}
    ''')
    
    st.dataframe(create_styled_dataframe(dp_table, X, Y), use_container_width=True)

    # --- Step 3: Backtracking (with theory) ---
    st.markdown("### Step 3: Backtracking for LCS (Solution Reconstruction)")
    
    st.markdown(
        """
        After the entire table is filled, the value in the bottom-right cell $L[m, n]$ is the length of the LCS. 
        To reconstruct the actual sequence, we **b  acktrack** from $L[m, n]$ to $L[0, 0]$:
        * **Diagonal Move (Match):** If the value $L[i, j]$ comes from $L[i-1, j-1] + 1$, it means a match occurred, and the character $X[i-1]$ belongs to the LCS. We add it to the sequence and move diagonally.
        * **Up or Left Move (Mismatch):** If the value comes from $\max(L[i-1, j], L[i, j-1])$, it means a mismatch occurred, and we move to the cell (up or left) with the **larger value** to trace the optimal path.
        """
    )
    st.dataframe(create_styled_dataframe(dp_table, X, Y, highlight_path=lcs_path), use_container_width=True)
    
    # --- SIMPLE Summary Section (Text-only as requested) ---
    st.write("---")
    
    len_X = len(X)
    len_Y = len(Y)
    avg_length = (len_X + len_Y) / 2
    percent_matched = (lcs_length / avg_length) * 100 if avg_length > 0 else 0.0

    st.subheader("LCS Results Summary")
    
    st.markdown(f"DNA A (String A): **{X}** (Length: **{len_X}**)")
    st.markdown(f"DNA B (String B): **{Y}** (Length: **{len_Y}**)")
    st.markdown(f"LCS Length: **{lcs_length}**")
    st.markdown(f"Longest Common Sequence: **{lcs}**")
    st.markdown(f"Percentage Matched: **{percent_matched:.2f}%** (LCS Length / Average DNA Length)")

    st.write("---")
    
    # --- Final Step: Animation ---
    
    st.sidebar.header("Animation settings")
    speed_ms = st.sidebar.slider("Frame duration (ms)", min_value=50, max_value=1000, value=250, step=25)
    max_anim_length = st.sidebar.slider("Max string length for animation", min_value=5, max_value=20, value=12, step=1)
    max_frames = st.sidebar.number_input("Max frames", min_value=50, max_value=2000, value=500, step=50)

    st.subheader("Final Step: Full Animated Explanation")
    
    if len(X) > max_anim_length or len(Y) > max_anim_length:
        st.warning(f"One string is longer than {max_anim_length}. Animation is resource-intensive and is currently disabled. Please use shorter strings for animation.")
    else:
        if st.button("Generate Animation"):
            with st.spinner("Generating animation (may take a moment for longer strings)..."):
                try:
                    gif_bytes, lcs_from_anim, dp_full_anim = create_lcs_animation(X, Y, speed_ms=speed_ms, max_frames=int(max_frames))
                    
                    if gif_bytes is None:
                        st.error("No frames generated for animation.")
                    else:
                        st.image(gif_bytes)
                        b64 = base64.b64encode(gif_bytes).decode()
                        href = f'<a href="data:image/gif;base64,{b64}" download="lcs_animation.gif">Download GIF</a>'
                        st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Failed to generate animation: {e}")

st.write("---")
st.caption("Built for educational purposes.")