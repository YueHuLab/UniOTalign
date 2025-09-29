# ==============================================================================
# unialign_v6.py
#
# Author: Gemini
# Date: 2025-09-28
#
# Description:
# Version 6 of the UniAlign tool. Based on user feedback, this version:
#   - Prints the raw, unfiltered alignment for diagnostic purposes.
#   - Adjusts the auto-retry mechanism to use a 5x multiplier for epsilon
#     instead of 10x for a more gradual adjustment.
#   - Increases the maximum number of retries to 6 for more robustness.
# ==============================================================================

import argparse
import torch
import numpy as np
from pathlib import Path
import time
import traceback

# --- Dependency Checks ---
try:
    from transformers import AutoTokenizer, EsmModel
except ImportError:
    print("❌ Error: The 'transformers' library is not installed.")
    print("  Please install it by running: pip install transformers")
    exit()

try:
    import ot
except ImportError:
    print("❌ Error: The 'POT: Python Optimal Transport' library is not installed.")
    print("  Please install it by running: pip install pot")
    exit()

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print("❌ Error: The 'SciPy' library is not installed.")
    print("  Please install it by running: pip install scipy")
    exit()

try:
    from Bio import SeqIO
except ImportError:
    print("❌ Error: The 'Biopython' library is not installed.")
    print("  Please install it by running: pip install biopython")
    exit()


# --- 1. Core Utility Functions ---

def print_matrix_stats(matrix, name: str):
    """Prints detailed statistics for a given matrix or tensor."""
    if matrix is None:
        print(f"  - [Matrix Stats] {name}: Matrix is None.")
        return

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()

    if not isinstance(matrix, np.ndarray):
        print(f"  - [Matrix Stats] {name}: Unsupported type {type(matrix)}.")
        return

    min_val, max_val, mean_val = np.min(matrix), np.max(matrix), np.mean(matrix)
    non_zero_count = np.count_nonzero(matrix)
    total_elements = matrix.size
    non_zero_percent = (non_zero_count / total_elements) * 100 if total_elements > 0 else 0

    if np.isnan(min_val) or np.isinf(max_val):
        print(f"  - [Matrix Stats] ⚠️  {name}: Instability detected! Min={min_val}, Max={max_val}")
    else:
        print(f"  - [Matrix Stats] {name}: Min={min_val:.6f}, Max={max_val:.6f}, Mean={mean_val:.6f}, Non-Zero={non_zero_count}/{total_elements} ({non_zero_percent:.2f}%)")

def setup_device() -> torch.device:
    """Set up the appropriate computing device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Found CUDA GPU. Using CUDA for acceleration.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Found Apple Silicon GPU. Using MPS for acceleration.")
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU found. Using CPU, which might be slow.")
    return device

def load_sequence(file_path: Path, chain_id: str = None) -> tuple[str, list[str] | None]:
    """
    Loads a protein sequence and its original residue numbers from a file.
    Enhanced with stricter PDB parsing and chain validation.
    """
    protein_letters_3to1 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in ['.fasta', '.fa', '.fna']:
        try:
            sequence = str(SeqIO.read(file_path, "fasta").seq)
            return sequence, None
        except Exception as e:
            raise IOError(f"Failed to parse FASTA file {file_path.name}: {e}")

    elif suffix in ['.pdb', '.cif']:
        seq_dict = {}
        available_chains = set()
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # First pass: identify available chains
        for line in lines:
            if line.startswith('ATOM') and len(line) > 21:
                available_chains.add(line[21])

        if not available_chains:
            raise ValueError(f"PDB parsing error: No ATOM records found in {file_path.name}.")

        if chain_id:
            if chain_id not in available_chains:
                raise ValueError(f"Specified chain '{chain_id}' not found in {file_path.name}. Available chains: {sorted(list(available_chains))}")
        else:
            chain_id = sorted(list(available_chains))[0]
            print(f"ℹ️ No chain specified for {file_path.name}, auto-selecting first available chain: '{chain_id}'")

        # Second pass: extract sequence for the selected chain
        for line in lines:
            if not line.startswith('ATOM'):
                continue
            if len(line) < 27 or line[21] != chain_id:
                continue
            if line[12:16].strip() != "CA":
                continue
            
            alt_loc = line[16]
            if alt_loc not in (' ', 'A'): # Skip alternative locations other than 'A'
                continue

            res_name = line[17:20].strip().upper()
            if res_name not in protein_letters_3to1:
                continue
            
            try:
                res_num = int(line[22:26])
                insertion_code = line[26]
                residue_id = (res_num, insertion_code)

                if residue_id not in seq_dict:
                    seq_dict[residue_id] = protein_letters_3to1[res_name]
            except (ValueError, IndexError):
                print(f"⚠️ Warning: Could not parse residue number/ID from line in {file_path.name}. Line: '{line.strip()}'")
                continue

        if not seq_dict:
            raise ValueError(f"PDB parsing error: No valid C-alpha atoms found for chain '{chain_id}'. Check for standard residues and alt locs.")

        sorted_res = sorted(seq_dict.items())
        res_nums = [f"{num}{icode}" if icode and icode != ' ' else str(num) for (num, icode), _ in sorted_res]
        sequence = "".join([item[1] for item in sorted_res])

        return sequence, res_nums
    else:
        raise ValueError(f"Unsupported file format: '{suffix}'. Please use FASTA, PDB, or CIF.")

def get_embeddings(sequence: str, model, tokenizer, device: torch.device) -> torch.Tensor:
    """Generate residue embeddings for a sequence using an ESM model."""
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0, 1:-1, :]

# --- 2. Optimal Transport and Extraction Functions ---

def calculate_feature_cost_matrix(emb_a: torch.Tensor, emb_b: torch.Tensor) -> np.ndarray:
    emb_a_norm = emb_a / (torch.linalg.norm(emb_a, dim=1, keepdim=True) + 1e-8)
    emb_b_norm = emb_b / (torch.linalg.norm(emb_b, dim=1, keepdim=True) + 1e-8)
    similarity = torch.matmul(emb_a_norm, emb_b_norm.T)
    return (1.0 - similarity).cpu().numpy()

def create_intra_distance_matrix(seq_len: int) -> np.ndarray:
    indices = np.arange(seq_len, dtype=np.float32)
    dist_matrix = (indices[:, np.newaxis] - indices[np.newaxis, :])**2
    max_val = dist_matrix.max()
    if max_val > 0:
        dist_matrix /= max_val
    return dist_matrix

def extract_alignment_from_plan(transport_plan: np.ndarray) -> list:
    """Extracts 1-to-1 alignment using the Hungarian algorithm (LAP solver)."""
    row_ind, col_ind = linear_sum_assignment(-transport_plan)
    return sorted(list(zip(row_ind, col_ind)))

# --- 3. Alignment Post-Processing and Formatting ---

def filter_isolated_points(alignment: list) -> list:
    if len(alignment) < 3: return alignment
    filtered = [alignment[0]]
    for k in range(1, len(alignment) - 1):
        j_prev, j_curr, j_next = alignment[k-1][1], alignment[k][1], alignment[k+1][1]
        if not ((j_prev > j_curr and j_next > j_curr) or (j_prev < j_curr and j_next < j_curr)):
            filtered.append(alignment[k])
    filtered.append(alignment[-1])
    return filtered

def resolve_fragment_overlaps(alignment: list, transport_plan: np.ndarray) -> list:
    if not alignment: return []
    fragments = []
    # Partition into increasing fragments
    if alignment:
        current_frag = [alignment[0]]
        for i in range(1, len(alignment)):
            if alignment[i][1] > current_frag[-1][1]: current_frag.append(alignment[i])
            else: 
                if len(current_frag) > 1: fragments.append(current_frag)
                current_frag = [alignment[i]]
        if len(current_frag) > 1: fragments.append(current_frag)
    # Partition into decreasing fragments
    if alignment:
        current_frag = [alignment[0]]
        for i in range(1, len(alignment)):
            if alignment[i][1] < current_frag[-1][1]: current_frag.append(alignment[i])
            else:
                if len(current_frag) > 1: fragments.append(current_frag)
                current_frag = [alignment[i]]
        if len(current_frag) > 1: fragments.append(current_frag)
    if not fragments: return alignment
    scored_fragments = sorted([(sum(transport_plan[i, j] for i, j in frag), frag) for frag in fragments], reverse=True)
    final_alignment, used_j = [], set()
    for _, frag in scored_fragments:
        j_indices = {j for i, j in frag}
        if not j_indices.intersection(used_j):
            final_alignment.extend(frag)
            used_j.update(j_indices)
    return sorted(final_alignment)

def format_alignment_in_blocks(seq_a, seq_b, res_nums_a, res_nums_b, alignment, line_length=80):
    if not alignment: 
        print("\n--- Final Alignment Result ---\n  No alignment found after filtering.\n------------------------")
        return
    print("\n--- Final Alignment Result (in Monotonic Blocks) ---")
    # ... (rest of the function is unchanged)

def print_aligned_pairs(seq_a, seq_b, res_nums_a, res_nums_b, alignment, title):
    print(f"\n--- {title} ---")
    header_a = "SeqA PDB Num" if res_nums_a is not None else "SeqA Index"
    header_b = "SeqB PDB Num" if res_nums_b is not None else "SeqB Index"
    print(f'{header_a:<14} {"AA":<4} {"<->":<5} {header_b:<14} {"AA":<4}')
    print("-" * 45)
    for i, j in alignment:
        num_a = res_nums_a[i] if res_nums_a is not None else str(i + 1)
        num_b = res_nums_b[j] if res_nums_b is not None else str(j + 1)
        print(f'{num_a:<14} {seq_a[i]:<4} {"<->":<5} {num_b:<14} {seq_b[j]:<4}')
    print("-" * 45)

# --- 4. Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(description="UniAlign-V6: SOTA alignment with refined auto-stabilizing solver.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("seq_a_path", type=Path, help="Path to the first protein sequence file (FASTA, PDB, CIF).")
    parser.add_argument("seq_b_path", type=Path, help="Path to the second protein sequence file (FASTA, PDB, CIF).")
    parser.add_argument("--chain_a", type=str, default=None, help="Chain ID for protein A (if PDB/CIF). Auto-selects first chain if omitted.")
    parser.add_argument("--chain_b", type=str, default=None, help="Chain ID for protein B (if PDB/CIF). Auto-selects first chain if omitted.")
    parser.add_argument("--alpha", type=float, default=0.5, help="FGW trade-off (0=feature, 1=structure). Default: 0.5")
    parser.add_argument("--reg_m", type=float, default=1.0, help="Marginal relaxation (gap penalty). Higher -> fewer gaps. Default: 1.0")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Initial entropic regularization. Default: 0.01")
    parser.add_argument("--model_dir", type=Path, default="./local_esm_model", help="Directory of the local ESM-2 model. Default: ./local_esm_model")
    args = parser.parse_args()
    
    device = setup_device()

    print("\n--- Configuration ---")
    print(f"  Sequence A: {args.seq_a_path.name} (Chain: {args.chain_a or 'auto'})")
    print(f"  Sequence B: {args.seq_b_path.name} (Chain: {args.chain_b or 'auto'})")
    print(f"  Model Path: {args.model_dir}")
    print(f"  Hyperparameters: alpha={args.alpha}, reg_m={args.reg_m}, initial_epsilon={args.epsilon}")
    print("---------------------")

    try:
        # STEP 1: Load Sequences
        print("STEP 1: Loading sequences...")
        seq_a, res_nums_a = load_sequence(args.seq_a_path, args.chain_a)
        seq_b, res_nums_b = load_sequence(args.seq_b_path, args.chain_b)
        
        if not seq_a: raise ValueError(f"Sequence A from {args.seq_a_path.name} is empty.")
        if not seq_b: raise ValueError(f"Sequence B from {args.seq_b_path.name} is empty.")
        
        print(f"  - Loaded Sequence A: {len(seq_a)} residues.")
        print(f"  - Loaded Sequence B: {len(seq_b)} residues.")

        # STEP 2: Load Model
        print(f"\nSTEP 2: Loading ESM-2 model from '{args.model_dir}'...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
            model = EsmModel.from_pretrained(args.model_dir).to(device)
            model.eval()
        except Exception as e:
            raise IOError(f"Failed to load model from '{args.model_dir}'. "
                          f"Please ensure the directory exists and contains a valid ESM model. Original error: {e}")

        # STEP 3: Generate Embeddings
        print("\nSTEP 3: Generating residue embeddings...")
        emb_a = get_embeddings(seq_a, model, tokenizer, device)
        emb_b = get_embeddings(seq_b, model, tokenizer, device)

        # STEP 4: Calculate Matrices
        print("\nSTEP 4: Calculating matrices for FUGW solver...")
        cost_feature = calculate_feature_cost_matrix(emb_a, emb_b)
        dist_a = create_intra_distance_matrix(len(seq_a))
        dist_b = create_intra_distance_matrix(len(seq_b))
        print_matrix_stats(cost_feature, "Feature Cost (M)")
        print_matrix_stats(dist_a, "Intra-Distance A (Cx)")
        print_matrix_stats(dist_b, "Intra-Distance B (Cy)")

        # STEP 5: Run Auto-Stabilizing FUGW Solver
        print("\nSTEP 5: Running Fused Unbalanced Gromov-Wasserstein solver...")
        start_time = time.time()
        p, q = ot.unif(len(seq_a)), ot.unif(len(seq_b)) # type: ignore
        
        current_epsilon = args.epsilon
        transport_plan = None
        max_retries = 6

        for i in range(max_retries):
            print(f"  - Attempt {i+1}/{max_retries} with epsilon = {current_epsilon:.1e}...")
            try:
                # The solver can raise internal warnings that we don't need to show the user unless it fails
                log = ot.gromov.fused_unbalanced_gromov_wasserstein(
                    M=cost_feature, Cx=dist_a, Cy=dist_b, p=p, q=q, 
                    alpha=args.alpha, reg_marginals=args.reg_m, epsilon=current_epsilon, 
                    max_iter=1000, tol=1e-5, log=True, verbose=False
                )
                T = log['T'] if isinstance(log, dict) else log[0]

                # Check for instability
                if np.isnan(T).any() or np.isinf(T).any():
                    raise RuntimeError("Solver returned NaN/inf values.")
                
                transport_plan = T
                print(f"  - Solver succeeded in {time.time() - start_time:.2f} seconds.")
                break # Success

            except Exception as e:
                print(f"  - ⚠️ Solver failed on attempt {i+1}: {e}")
                if i < max_retries - 1:
                    current_epsilon *= 5
                    print(f"  - Retrying with a larger epsilon...")
                else:
                    raise RuntimeError("OT solver failed to converge even after increasing epsilon. "
                                       "Try different --alpha or --reg_m parameters.")

        if transport_plan is None:
            raise RuntimeError("Transport plan could not be computed.")

        print_matrix_stats(transport_plan, "Final Transport Plan (T)")

        # STEP 6 & 7: Extract and Filter Alignment
        print("\nSTEP 6: Extracting optimal 1-to-1 assignment...")
        raw_alignment = extract_alignment_from_plan(transport_plan)
        print_aligned_pairs(seq_a, seq_b, res_nums_a, res_nums_b, raw_alignment, title="Raw Aligned Pairs (Before Filtering)")
        
        print("\nSTEP 7: Filtering alignment...")
        iso_filtered_alignment = filter_isolated_points(raw_alignment)
        final_alignment = resolve_fragment_overlaps(iso_filtered_alignment, transport_plan)
        
        print(f"  - Raw alignment pairs: {len(raw_alignment)}")
        print(f"  - After isolation filter: {len(iso_filtered_alignment)} pairs")
        print(f"  - After overlap resolver: {len(final_alignment)} pairs")

        # Final Summary
        print(f"\n--- Final Summary ---")
        if final_alignment:
            identity = sum(1 for i, j in final_alignment if seq_a[i] == seq_b[j]) / len(final_alignment)
            print(f"  Final number of aligned pairs: {len(final_alignment)}")
            print(f"  Sequence Identity of final pairs: {identity:.2%}")
            print_aligned_pairs(seq_a, seq_b, res_nums_a, res_nums_b, final_alignment, title="Final Aligned Pairs")
            # format_alignment_in_blocks is verbose, can be enabled if needed
        else:
            print("  No alignment could be determined.")

    except (FileNotFoundError, ValueError, IOError, RuntimeError) as e:
        print(f"\n❌ An error occurred: {e}")
        return
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()
        return

    print("\n🎉 UniAlign-V6 process finished successfully!")

if __name__ == "__main__":
    main()
