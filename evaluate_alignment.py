import re
from pathlib import Path
import pandas as pd

# 3-letter to 1-letter amino acid code mapping
PROTEIN_LETTERS_3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

def parse_reference_file(file_path):
    """
    Parses the reference alignment file to extract ground truth alignments,
    including amino acid types.
    """
    reference_alignments = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_pair = None
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            # Updated regex to better handle various naming conventions
            pair_match = re.match(r'#([a-zA-Z0-9_\.]+)-([a-zA-Z0-9_\.]+)', line)
            if pair_match:
                p1 = pair_match.group(1).lower()
                p2 = pair_match.group(2).lower()
                current_pair = tuple(sorted((p1, p2)))
                reference_alignments[current_pair] = []
        elif line and current_pair:
            parts = line.split('\t')
            if len(parts) == 2:
                res1_str, res2_str = parts
                res1_parts = res1_str.split('.')
                res2_parts = res2_str.split('.')

                if len(res1_parts) >= 2 and len(res2_parts) >= 2:
                    aa1 = res1_parts[0]
                    res1_num = res1_parts[1]
                    # Handle insertion codes like '77S'
                    res1_icode = res1_parts[2] if len(res1_parts) > 2 and res1_parts[2] != '_' else ''
                    
                    aa2 = res2_parts[0]
                    res2_num = res2_parts[1]
                    res2_icode = res2_parts[2] if len(res2_parts) > 2 and res2_parts[2] != '_' else ''

                    ref_res1_id = f"{res1_num}{res1_icode}"
                    ref_res2_id = f"{res2_num}{res2_icode}"
                    
                    # Convert to 1-letter code for comparison
                    aa1_one_letter = PROTEIN_LETTERS_3TO1.get(aa1, '?')
                    aa2_one_letter = PROTEIN_LETTERS_3TO1.get(aa2, '?')

                    reference_alignments[current_pair].append((ref_res1_id, aa1_one_letter, ref_res2_id, aa2_one_letter))
    return reference_alignments

def parse_log_file(log_path):
    """
    Parses a single alignment log file to extract protein names and the raw alignment.
    """
    with open(log_path, 'r') as f:
        content = f.read()

    # Extract base protein names, removing .pdb suffix
    seq_a_match = re.search(r"Sequence A: (\S+?)(?:\.pdb)?\s*\(", content)
    seq_b_match = re.search(r"Sequence B: (\S+?)(?:\.pdb)?\s*\(", content)

    if not seq_a_match or not seq_b_match:
        return None, None

    p1 = seq_a_match.group(1).lower()
    p2 = seq_b_match.group(1).lower()
    pair_key = tuple(sorted((p1, p2)))

    aligned_pairs = []

    # Regex to find the raw alignment data block
    section_finder = re.compile(
        r"---\s*Raw\s*Aligned\s*Pairs\s*\(Before\s*Filtering\)\s*---\n"
        r".*?\n"  # Skips the "SeqA PDB Num..." header line
        r"-+\n"   # Skips the "------------------\n" separator line
        r"(.+?)"  # The actual alignment data (non-greedy capture)
        r"\n-+",    # Stop at the next line that contains dashes
        re.DOTALL
    )
    
    match = section_finder.search(content)
    if match:
        alignment_data = match.group(1)
        lines = alignment_data.strip().split('\n')
        for line in lines:
            pair_match = re.match(r"^\s*(\S+)\s+([A-Z])\s+<->\s+(\S+)\s+([A-Z])", line)
            if pair_match:
                res1_id, aa1, res2_id, aa2 = pair_match.groups()
                aligned_pairs.append((res1_id, aa1, res2_id, aa2))
            
    return pair_key, aligned_pairs

def main():
    log_dir = Path('output/alignment_logs')
    reference_file = Path('reference_RPIC.txt')

    if not reference_file.exists():
        print(f"Error: Reference file not found at {reference_file}")
        return

    if not log_dir.exists() or not log_dir.is_dir():
        print(f"Error: Log directory not found at {log_dir}")
        return

    reference_data = parse_reference_file(reference_file)
    results = []
    log_files = list(log_dir.glob('*.log'))

    if not log_files:
        print(f"No .log files found in {log_dir}")
        return

    for log_file in log_files:
        pair_key, predicted_pairs_raw = parse_log_file(log_file)
        
        if pair_key is None:
            print(f"Warning: Could not parse protein pair from {log_file.name}")
            continue

        # The protein names in the reference file might have a trailing underscore
        # Let's try to match with and without it.
        pair_key_alt = tuple(sorted([p.strip('_') for p in pair_key]))

        if pair_key in reference_data or pair_key_alt in reference_data:
            ref_pairs_key = pair_key if pair_key in reference_data else pair_key_alt
            ref_pairs = set(reference_data[ref_pairs_key])
            
            # Predicted pairs as (res1, aa1, res2, aa2)
            pred_pairs = set(predicted_pairs_raw)
            # Swapped predicted pairs as (res2, aa2, res1, aa1) to account for order ambiguity
            pred_pairs_swapped = set([(p[2], p[3], p[0], p[1]) for p in pred_pairs])

            correct_count = len(ref_pairs.intersection(pred_pairs))
            correct_count_swapped = len(ref_pairs.intersection(pred_pairs_swapped))
            
            # The correct alignment could be A->B or B->A. We take the one with more matches.
            final_correct = max(correct_count, correct_count_swapped)

            total_ref = len(ref_pairs)
            total_pred = len(pred_pairs)
            precision = (final_correct / total_pred) * 100 if total_pred > 0 else 0
            recall = (final_correct / total_ref) * 100 if total_ref > 0 else 0
            
            results.append({
                "Protein Pair": f"{ref_pairs_key[0]}_vs_{ref_pairs_key[1]}",
                "Correct": final_correct,
                "Reference": total_ref,
                "Predicted": total_pred,
                "Recall (%)": recall,
                "Precision (%)": precision
            })
        else:
            print(f"Info: Pair {pair_key} (and {pair_key_alt}) from log {log_file.name} not found in reference file.")


    if not results:
        print("No matching pairs found between any log files and the reference file.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="Recall (%)", ascending=False)
    
    print("\n--- Alignment Evaluation Summary ---")
    print(df.to_string(index=False))

    if not df.empty:
        avg_recall = df["Recall (%)"].mean()
        avg_precision = df["Precision (%)"].mean()
        print(f"\n--- Overall Average Recall: {avg_recall:.2f}% ---")
        print(f"--- Overall Average Precision: {avg_precision:.2f}% ---")

if __name__ == "__main__":
    main()
