"""
Dataset Generator for Insect COI Mitochondrial Barcoding & Phylogeny
Generates realistic 658 bp Folmer region sequences across 6 major Insecta orders:
Coleoptera, Diptera, Lepidoptera, Hymenoptera, Hemiptera, Odonata.
"""

import os
import sys
import random
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Set fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

# Taxonomy database: Orders -> Species
INSECT_TAXONOMY = {
    "Coleoptera": [
        ("Tribolium castaneum", "Red flour beetle", 0.36),
        ("Harmonia axyridis", "Harlequin ladybird", 0.37),
        ("Coccinella septempunctata", "Seven-spot ladybird", 0.35),
        ("Carabus violaceus", "Violet ground beetle", 0.38),
        ("Tenebrio molitor", "Mealworm beetle", 0.36),
        ("Leptinotarsa decemlineata", "Colorado potato beetle", 0.34),
        ("Lucanus cervus", "Stag beetle", 0.39),
    ],
    "Diptera": [
        ("Drosophila melanogaster", "Fruit fly", 0.30),
        ("Aedes aegypti", "Yellow fever mosquito", 0.28),
        ("Anopheles gambiae", "African malaria mosquito", 0.27),
        ("Musca domestica", "Common housefly", 0.30),
        ("Lucilia sericata", "Green bottle fly", 0.31),
        ("Culex quinquefasciatus", "Southern house mosquito", 0.29),
        ("Ceratitis capitata", "Mediterranean fruit fly", 0.31),
    ],
    "Lepidoptera": [
        ("Bombyx mori", "Domestic silkmoth", 0.32),
        ("Danaus plexippus", "Monarch butterfly", 0.30),
        ("Heliconius melpomene", "Postman butterfly", 0.31),
        ("Manduca sexta", "Tobacco hornworm", 0.33),
        ("Pieris rapae", "Small cabbage white", 0.29),
        ("Papilio machaon", "Old World swallowtail", 0.32),
        ("Spodoptera frugiperda", "Fall armyworm", 0.30),
    ],
    "Hymenoptera": [
        ("Apis mellifera", "Western honeybee", 0.22),
        ("Bombus terrestris", "Buff-tailed bumblebee", 0.23),
        ("Formica rufa", "Red wood ant", 0.25),
        ("Camponotus floridanus", "Carpenter ant", 0.24),
        ("Polistes dominula", "European paper wasp", 0.26),
        ("Vespa crabro", "European hornet", 0.25),
        ("Lasius niger", "Black garden ant", 0.24),
    ],
    "Hemiptera": [
        ("Acyrthosiphon pisum", "Pea aphid", 0.33),
        ("Halyomorpha halys", "Brown marmorated stink bug", 0.35),
        ("Cimex lectularius", "Common bed bug", 0.32),
        ("Bemisia tabaci", "Silverleaf whitefly", 0.31),
        ("Nilaparvata lugens", "Brown planthopper", 0.34),
        ("Oncopeltus fasciatus", "Large milkweed bug", 0.36),
        ("Rhodnius prolixus", "Kissing bug", 0.35),
    ],
    "Odonata": [
        ("Anax junius", "Common green darner", 0.40),
        ("Libellula quadrimaculata", "Four-spotted chaser", 0.42),
        ("Ischnura elegans", "Common blue damselfly", 0.39),
        ("Enallagma cyathigerum", "Common blue damsel", 0.40),
        ("Sympetrum striolatum", "Common darter", 0.41),
        ("Calopteryx splendens", "Banded demoiselle", 0.43),
        ("Aeshna cyanea", "Southern hawker", 0.41),
    ]
}

# Conserved COI blocks (Folmer barcode region, ~658 bp total)
CONSERVED_START = "TCTACTAATCATAAAGATATTGGTACTTTATATTTTATTTTTGGAATTTGAGCGGGAATA"
CONSERVED_MID = "CTTGTAAGATCATTAATTCGATTAGAATTAGGTAATCCTGGATCATTAATTGGAGATGATCAA"
CONSERVED_CATALYTIC = "CATTTTCATTATGTATTATCAATAGGAGCAGTATTTGCTATTATAGCAGGATTTATTCAT"
CONSERVED_END = "TTTGACCCAGCCGGAGGAGGTGACCCAATTCTTTACCAACACTTATTTTGATTTTTTGGTCAT"

# Order-specific diagnostic signature motifs
ORDER_SIGNATURES = {
    "Coleoptera": [
        "ACAGGTACTTTATATC", "GGAGCACCTGATATAG", "TATTATTTGAGCCTAC", 
        "GCATTACTTGGAATTC", "CGTACCTTAATACTTG", "TTTACCTCTTTTAGCA"
    ],
    "Diptera": [
        "AACAGGATGAACAGTT", "TTCCTTTATTTGTATG", "TTAGGTGCACCAGATA", 
        "GATATAGCATTTCCAC", "GTAAATTTATTTACAG", "TTTCTCTTTATTTATC"
    ],
    "Lepidoptera": [
        "GGAACTTCTTTGAGTT", "TTATTATTTTCAGGTA", "GGATCTCCTGATATAG", 
        "ATAGCATATCATGGAA", "TTATTAGGAATTTCTA", "AATTTCATTATTTTTA"
    ],
    "Hymenoptera": [
        "AATAAATTTTAATATT", "TTAAATTTTTTAATTA", "TTTAATTTTATTAATA", 
        "TAATTATTTTAATTAT", "ATTAATTTATTTAATT", "TATTTTAAATAAATTT"
    ],
    "Hemiptera": [
        "AGCGGGTATAGTAGGA", "ACAGGTTCATCAATTT", "GGTGCAGGGACAGGAT", 
        "TCATTCTTAGCTGGAA", "CGTACTCTAATATTAG", "TGTTCCTCTTTTTGCA"
    ],
    "Odonata": [
        "CCCGGAGCTCTAATTG", "CACCGGACATAGCATT", "GCCCTACTTGGAGACG", 
        "CGTTACTTCGTCTATC", "GGTCCCTTGATACTAG", "CGTCCCTCTTTTCGCA"
    ]
}

def generate_coi_sequence(order, target_gc, length=658):
    """
    Synthesize a biologically valid 658 bp insect COI barcoding sequence.
    """
    gc_half = target_gc / 2.0
    at_half = (1.0 - target_gc) / 2.0
    p_a = at_half * 0.94
    p_t = at_half * 1.06
    p_c = gc_half * 0.95
    p_g = gc_half * 1.05
    probs = [p_a, p_c, p_g, p_t]
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    bases = ['A', 'C', 'G', 'T']

    seq_chars = list(np.random.choice(bases, size=length, p=probs))

    # Splice conserved blocks
    # 1. 5' start
    for i, ch in enumerate(CONSERVED_START):
        if random.random() < 0.95:
            seq_chars[i] = ch

    # 2. Middle catalytic domain
    mid_start = 220
    for i, ch in enumerate(CONSERVED_MID):
        if random.random() < 0.95:
            seq_chars[mid_start + i] = ch

    # 3. Heme binding domain
    cat_start = 380
    for i, ch in enumerate(CONSERVED_CATALYTIC):
        if random.random() < 0.95:
            seq_chars[cat_start + i] = ch

    # 4. 3' terminal domain
    end_start = 590
    for i, ch in enumerate(CONSERVED_END):
        if random.random() < 0.94:
            seq_chars[end_start + i] = ch

    # 5. Insert order-specific diagnostic motifs
    signatures = ORDER_SIGNATURES[order]
    insert_positions = [75, 140, 290, 350, 470, 530]
    for pos, sig in zip(insert_positions, signatures):
        for i, ch in enumerate(sig):
            if random.random() < 0.92:
                seq_chars[pos + i] = ch

    return "".join(seq_chars)

def calculate_gc(seq):
    gc_count = seq.count('G') + seq.count('C')
    return round((gc_count / len(seq)) * 100, 2)

def generate_dataset(n_samples_per_order=50, output_dir="."):
    records = []
    fasta_lines = []
    sample_id = 1

    for order, species_list in INSECT_TAXONOMY.items():
        for i in range(n_samples_per_order):
            species_info = species_list[i % len(species_list)]
            species_name = species_info[0]
            common_name = species_info[1]
            base_gc = species_info[2]

            target_gc = float(np.clip(np.random.normal(base_gc, 0.008), 0.18, 0.48))
            seq_len = int(np.random.choice([655, 657, 658, 659, 660], p=[0.1, 0.1, 0.6, 0.1, 0.1]))

            sequence = generate_coi_sequence(order, target_gc, length=seq_len)
            calculated_gc = calculate_gc(sequence)
            accession = f"BOLD-INS{sample_id:04d}"

            records.append({
                "sample_id": accession,
                "order": order,
                "species": species_name,
                "common_name": common_name,
                "gene": "COI (cox1)",
                "length_bp": len(sequence),
                "gc_percent": calculated_gc,
                "sequence": sequence
            })

            fasta_header = f">{accession}|{order}|{species_name.replace(' ', '_')}|{calculated_gc}%GC|len={len(sequence)}"
            fasta_lines.append(fasta_header)
            fasta_lines.append(sequence)

            sample_id += 1

    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "insect_phylogeny_data.csv")
    fasta_path = os.path.join(output_dir, "insect_coi_sequences.fasta")

    df.to_csv(csv_path, index=False)
    with open(fasta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(fasta_lines))

    print(f"[+] Generated {len(df)} sequences across {len(INSECT_TAXONOMY)} insect orders.")
    print(f"[+] CSV: {csv_path}")
    print(f"[+] FASTA: {fasta_path}")
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    generate_dataset(n_samples_per_order=50, output_dir=data_dir)
