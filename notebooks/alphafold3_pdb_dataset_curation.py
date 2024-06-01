# %% [markdown]
# # Curating AlphaFold 3 PDB Dataset
#
# For training AlphaFold 3, we follow the training procedure outlined in Abramson et al (2024).
#
# Filtering of targets:
# 1. The structure must have been released to the PDB before the cutoff date of 2021-09-30.
# 2. The structure must have a reported resolution of 9 Å or less.
# 3. The maximum number of polymer chains in a considered structure is 300 for training and 1000 for evaluation.
# 4. Any polymer chain containing fewer than 4 resolved molecules is filtered out.
#
# Filtering of bioassemblies:
# 1. Hydrogens are removed.
# 2. Polymer chains with all unknown molecules are removed.
# 3. Clashing chains are removed. Clashing chains are defined as those with >30% of atoms within 1.7 Å of an atom
# in another chain. If two chains are clashing with each other, the chain with the greater percentage of clashing
# atoms will be removed. If the same fraction of atoms are clashing, the chain with fewer total atoms is removed.
# If the chains have the same number of atoms, then the chain with the larger chain id is removed.
# 4. For molecules or small molecules with CCD codes, atoms outside of the CCD code’s defined set of atom names are
# removed.
# 5. Leaving atoms (ligand atom or groups of atoms that detach when bonds form) for covalent ligands are filtered
# out.
# 6. Protein chains with consecutive Cα atoms >10 Å apart are filtered out.
# 7. For bioassemblies with greater than 20 chains, we select a random interface token (with a centre atom <15 Å to
# the centre atom of a token in another chain) and select the closest 20 chains to this token based on minimum
# distance between any tokens centre atom.
# 8. Crystallization aids are removed if the mmCIF method information indicates that crystallography was used (see
# Table 9).
#

# %%
import argparse
import glob
import os
from typing import Dict, Set

import pandas as pd
from Bio.PDB import PDBIO, MMCIFParser, PDBParser
from pdbeccdutils.core import ccd_reader, clc_reader


# Function to load CCD atoms
def load_ccd_atoms(ccd_file_path):
    ccd_reader_result = ccd_reader.read_pdb_components_file(ccd_file_path)
    ccd_atoms = {id: ccd_reader_result[id].component.atoms_ids for id in ccd_reader_result}
    return ccd_atoms


# Function to load covalent ligands
def load_covalent_ligands(ccd_file_path):
    clc_reader_result = clc_reader.read_clc_components_file(ccd_file_path)
    clc_atoms = {id: clc_reader_result[id].component.atoms_ids for id in clc_reader_result}
    return clc_atoms


# Function to parse structures based on file type
def parse_structure(file_path):
    if file_path.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif file_path.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError("Unsupported file format")
    structure_id = os.path.splitext(os.path.basename(file_path))[0]
    structure = parser.get_structure(structure_id, file_path)
    return structure


# Function to filter based on PDB deposition date
def filter_pdb_deposition_date(
    structure, cutoff_date: pd.Timestamp = pd.to_datetime("2021-09-30")
):
    if (
        "deposition_date" in structure.header
        and pd.to_datetime(structure.header["deposition_date"]) <= cutoff_date
    ):
        return True
    return False


# Function to filter based on resolution
def filter_resolution(structure, max_resolution=9.0):
    if "resolution" in structure.header and structure.header["resolution"] <= max_resolution:
        return True
    return False


# Function to filter based on number of polymer chains
def filter_polymer_chains(structure, max_chains=1000, for_training=False):
    count = sum(1 for chain in structure.get_chains() if chain.id[0] == " ")
    return count <= (300 if for_training else max_chains)


# Function to filter polymer chains based on resolved residues
def filter_resolved_chains(structure):
    for chain in structure.get_chains():
        if len([res for res in chain if res.id[0] == " "]) < 4:
            structure[0].detach_child(chain.id)
    return structure if list(structure.get_chains()) else None


# Function to determine if a target passes all target filters
def filter_target(structure):
    target_passes_prefilters = (
        filter_pdb_deposition_date(structure)
        and filter_resolution(structure)
        and filter_polymer_chains(structure)
    )
    return filter_resolved_chains(structure) if target_passes_prefilters else None


# Function to remove hydrogens
def remove_hydrogens(structure):
    for chain in structure.get_chains():
        for res in chain:
            for atom in res.get_atoms():
                if atom.element == "H":
                    res.detach_child(atom.name)
            if not list(res.get_atoms()):
                chain.detach_child(res.id)
        if not list(chain.get_residues()):
            structure[0].detach_child(chain.id)
    return structure


# Function to remove polymer chains with all unknown residues
def remove_all_unknown_residue_chains(structure, standard_residues):
    for chain in structure.get_chains():
        if not any(res.resname in standard_residues for res in chain):
            structure[0].detach_child(chain.id)
    return structure


# Function to remove clashing chains
def remove_clashing_chains(structure, clash_threshold: float = 1.7, clash_percentage: float = 0.3):
    chains = list(structure.get_chains())
    clashing_chains = []

    for i, chain1 in enumerate(chains):
        for chain2 in chains[i + 1 :]:
            clash_count = sum(
                1
                for atom1 in chain1.get_atoms()
                for atom2 in chain2.get_atoms()
                if (atom1 - atom2) < clash_threshold
            )
            if (
                clash_count / len(list(chain1.get_atoms())) > clash_percentage
                or clash_count / len(list(chain2.get_atoms())) > clash_percentage
            ):
                clashing_chains.append((chain1, chain2, clash_count))

    for chain1, chain2, clash_count in clashing_chains:
        if clash_count / len(list(chain1.get_atoms())) > clash_count / len(
            list(chain2.get_atoms())
        ):
            structure[0].detach_child(chain1.id)
        elif clash_count / len(list(chain2.get_atoms())) > clash_count / len(
            list(chain1.get_atoms())
        ):
            structure[0].detach_child(chain2.id)
        else:
            if len(list(chain1.get_atoms())) < len(list(chain2.get_atoms())):
                structure[0].detach_child(chain1.id)
            elif len(list(chain2.get_atoms())) < len(list(chain1.get_atoms())):
                structure[0].detach_child(chain2.id)
            else:
                if chain1.id > chain2.id:
                    structure[0].detach_child(chain1.id)
                else:
                    structure[0].detach_child(chain2.id)
    return structure


# Function to remove excluded ligands
def remove_excluded_ligands(structure, ligand_exclusion_list):
    for chain in structure.get_chains():
        for res in chain:
            if res.resname in ligand_exclusion_list:
                chain.detach_child(res.id)
        if not list(chain.get_residues()):
            structure[0].detach_child(chain.id)
    return structure


# Function to remove atoms not in CCD code set
def remove_non_ccd_atoms(structure, ccd_atoms):
    for chain in structure.get_chains():
        for res in chain:
            if res.resname in ccd_atoms:
                for atom in res.get_atoms():
                    if atom.name not in ccd_atoms.get(res.resname, {}):
                        res.detach_child(atom.name)
            if not list(res.get_atoms()):
                chain.detach_child(res.id)
        if not list(chain.get_residues()):
            structure[0].detach_child(chain.id)
    return structure


# Function to remove leaving atoms in covalent ligands
def remove_leaving_atoms(structure, covalent_ligands):
    for chain in structure.get_chains():
        for res in chain:
            if res.resname in covalent_ligands:
                for atom in res.get_atoms():
                    if atom.name in covalent_ligands[res.resname]:
                        res.detach_child(atom.name)
            if not list(res.get_atoms()):
                chain.detach_child(res.id)
        if not list(chain.get_residues()):
            structure[0].detach_child(chain.id)
    return structure


# Function to filter chains with large Cα distances
def filter_large_ca_distances(structure):
    for chain in structure.get_chains():
        ca_atoms = [res["CA"] for res in chain if "CA" in res]
        for i, ca1 in enumerate(ca_atoms[:-1]):
            ca2 = ca_atoms[i + 1]
            if (ca1, ca2) > 10:
                structure[0].detach_child(chain.id)
                break
    return structure


# TODO: Function to select closest 20 chains in large bioassemblies
def select_closest_chains(structure, max_chains=20):
    if len(structure.get_chains()) > max_chains:
        chains = list(structure.get_chains())
        import random

        token_chain = random.choice(chains)
        token_atom = random.choice(list(token_chain.get_atoms()))
        chain_distances = []
        for chain in chains:
            min_distance = min(token_atom - atom for atom in chain.get_atoms())
            chain_distances.append((chain, min_distance))
        chain_distances.sort(key=lambda x: x[1])
        for chain, _ in chain_distances[max_chains:]:
            structure[0].detach_child(chain.id)
    return structure


# Function to remove crystallization aids
def remove_crystallization_aids(structure, crystallography_methods):
    if structure.header["structure_method"] in crystallography_methods:
        for chain in structure.get_chains():
            for res in chain:
                if res.resname in crystallography_methods[structure.header["structure_method"]]:
                    chain.detach_child(res.id)
            if not list(chain.get_residues()):
                structure[0].detach_child(chain.id)
    return structure


# Example main function to process a list of mmCIF files
def process_structures(
    file_paths,
    standard_residues: Set[str],
    ligand_exclusion_list: Set[str],
    ccd_atoms: Dict[str, Set[str]],
    covalent_ligands: Dict[str, Set[str]],
    crystallography_methods: Dict[str, Set[str]],
):
    processed_structures = []
    for file_path in file_paths:
        structure = parse_structure(file_path)
        # Filtering of targets
        structure = filter_target(structure)
        if structure is not None:
            # Filtering of bioassemblies
            structure = remove_hydrogens(structure)
            structure = remove_all_unknown_residue_chains(structure, standard_residues)
            structure = remove_clashing_chains(structure)
            structure = remove_excluded_ligands(structure, ligand_exclusion_list)
            structure = remove_non_ccd_atoms(structure, ccd_atoms)
            structure = remove_leaving_atoms(structure, covalent_ligands)
            structure = filter_large_ca_distances(structure)
            structure = select_closest_chains(structure)
            structure = remove_crystallization_aids(structure, crystallography_methods)
            if list(structure.get_chains()):
                processed_structures.append(structure)
    return processed_structures


# Parse command-line arguments #

parser = argparse.ArgumentParser(
    description="Process mmCIF files to curate the AlphaFold 3 PDB dataset."
)
parser.add_argument(
    "--mmcif_dir",
    type=str,
    default=os.path.join("data", "mmCIF"),
    help="Path to the input directory containing mmCIF files to process.",
)
parser.add_argument(
    "--ccd_dir",
    type=str,
    default=os.path.join("data", "CCD"),
    help="Path to the directory containing CCD files to reference during data processing.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=os.path.join("data", "PDB_set"),
    help="Path to the output directory in which to store processed mmCIF dataset files.",
)
args = parser.parse_args("")

assert os.path.exists(args.mmcif_dir), f"Input directory {args.mmcif_dir} does not exist."
assert os.path.exists(args.ccd_dir), f"CCD directory {args.ccd_dir} does not exist."
assert os.path.exists(
    os.path.join(args.ccd_dir, "chem_comp_model.cif")
), f"CCD ligands file not found in {args.ccd_dir}."
assert os.path.exists(
    os.path.join(args.ccd_dir, "components.cif")
), f"CCD components file not found in {args.ccd_dir}."
os.makedirs(args.output_dir, exist_ok=True)

# Define constants #

# TODO: Section 2.5.4 of the AlphaFold 3 supplement
ccd_atoms = load_ccd_atoms(os.path.join(args.ccd_dir, "components.cif"))
covalent_ligands = load_covalent_ligands(os.path.join(args.ccd_dir, "components.cif"))

# Table 9 of the AlphaFold 3 supplement
crystallization_aids = set(
    "SO4, GOL, EDO, PO4, ACT, PEG, DMS, TRS, PGE, PG4, FMT, EPE, MPD, MES, CD, IOD".split(", ")
)
crystallography_methods = {"X-RAY DIFFRACTION": crystallization_aids}

# Table 10 of the AlphaFold 3 supplement
ligand_exclusion_list = set(
    "144, 15P, 1PE, 2F2, 2JC, 3HR, 3SY, 7N5, 7PE, 9JE, AAE, ABA, ACE, ACN, ACT, ACY, AZI, BAM, BCN, BCT, BDN, BEN, BME, BO3, BTB, BTC, BU1, C8E, CAD, CAQ, CBM, CCN, CIT, CL, CLR, CM, CMO, CO3, CPT, CXS, D10, DEP, DIO, DMS, DN, DOD, DOX, EDO, EEE, EGL, EOH, EOX, EPE, ETF, FCY, FJO, FLC, FMT, FW5, GOL, GSH, GTT, GYF, HED, IHP, IHS, IMD, IOD, IPA, IPH, LDA, MB3, MEG, MES, MLA, MLI, MOH, MPD, MRD, MSE, MYR, N, NA, NH2, NH4, NHE, NO3, O4B, OHE, OLA, OLC, OMB, OME, OXA, P6G, PE3, PE4, PEG, PEO, PEP, PG0, PG4, PGE, PGR, PLM, PO4, POL, POP, PVO, SAR, SCN, SEO, SEP, SIN, SO4, SPD, SPM, SR, STE, STO, STU, TAR, TBU, TME, TPO, TRS, UNK, UNL, UNX, UPL, URE".split(
        ", "
    )
)

# Table 11 of the AlphaFold 3 supplement
ccd_codes_defining_glycans = set(
    "045, 05L, 07E, 07Y, 08U, 09X, 0BD, 0H0, 0HX, 0LP, 0MK, 0NZ, 0UB, 0V4, 0WK, 0XY, 0YT, 10M, 12E, 145, 147, 149, 14T, 15L, 16F, 16G, 16O, 17T, 18D, 18O, 1CF, 1FT, 1GL, 1GN, 1LL, 1S3, 1S4, 1SD, 1X4, 20S, 20X, 22O, 22S, 23V, 24S, 25E, 26O, 27C, 289, 291, 293, 2DG, 2DR, 2F8, 2FG, 2FL, 2GL, 2GS, 2H5, 2HA, 2M4, 2M5, 2M8, 2OS, 2WP, 2WS, 32O, 34V, 38J, 3BU, 3DO, 3DY, 3FM, 3GR, 3HD, 3J3, 3J4, 3LJ, 3LR, 3MG, 3MK, 3R3, 3S6, 3SA, 3YW, 40J, 42D, 445, 44S, 46D, 46Z, 475, 48Z, 491, 49A, 49S, 49T, 49V, 4AM, 4CQ, 4GC, 4GL, 4GP, 4JA, 4N2, 4NN, 4QY, 4R1, 4RS, 4SG, 4UZ, 4V5, 50A, 51N, 56N, 57S, 5GF, 5GO, 5II, 5KQ, 5KS, 5KT, 5KV, 5L3, 5LS, 5LT, 5MM, 5N6, 5QP, 5SP, 5TH, 5TJ, 5TK, 5TM, 61J, 62I, 64K, 66O, 6BG, 6C2, 6DM, 6GB, 6GP, 6GR, 6K3, 6KH, 6KL, 6KS, 6KU, 6KW, 6LA, 6LS, 6LW, 6MJ, 6MN, 6PZ, 6S2, 6UD, 6YR, 6ZC, 73E, 79J, 7CV, 7D1, 7GP, 7JZ, 7K2, 7K3, 7NU, 83Y, 89Y, 8B7, 8B9, 8EX, 8GA, 8GG, 8GP, 8I4, 8LR, 8OQ, 8PK, 8S0, 8YV, 95Z, 96O, 98U, 9AM, 9C1, 9CD, 9GP, 9KJ, 9MR, 9OK, 9PG, 9QG, 9S7, 9SG, 9SJ, 9SM, 9SP, 9T1, 9T7, 9VP, 9WJ, 9WN, 9WZ, 9YW, A0K, A1Q, A2G, A5C, A6P, AAL, ABD, ABE, ABF, ABL, AC1, ACR, ACX, ADA, AF1, AFD, AFO, AFP, AGL, AH2, AH8, AHG, AHM, AHR, AIG, ALL, ALX, AMG, AMN, AMU, AMV, ANA, AOG, AQA, ARA, ARB, ARI, ARW, ASC, ASG, ASO, AXP, AXR, AY9, AZC, B0D, B16, B1H, B1N, B2G, B4G, B6D, B7G, B8D, B9D, BBK, BBV, BCD, BDF, BDG, BDP, BDR, BEM, BFN, BG6, BG8, BGC, BGL, BGN, BGP, BGS, BHG, BM3, BM7, BMA, BMX, BND, BNG, BNX, BO1, BOG, BQY, BS7, BTG, BTU, BW3, BWG, BXF, BXP, BXX, BXY, BZD, C3B, C3G, C3X, C4B, C4W, C5X, CBF, CBI, CBK, CDR, CE5, CE6, CE8, CEG, CEZ, CGF, CJB, CKB, CKP, CNP, CR1, CR6, CRA, CT3, CTO, CTR, CTT, D1M, D5E, D6G, DAF, DAG, DAN, DDA, DDL, DEG, DEL, DFR, DFX, DG0, DGO, DGS, DGU, DJB, DJE, DK4, DKX, DKZ, DL6, DLD, DLF, DLG, DNO, DO8, DOM, DPC, DQR, DR2, DR3, DR5, DRI, DSR, DT6, DVC, DYM, E3M, E5G, EAG, EBG, EBQ, EEN, EEQ, EGA, EMP, EMZ, EPG, EQP, EQV, ERE, ERI, ETT, EUS, F1P, F1X, F55, F58, F6P, F8X, FBP, FCA, FCB, FCT, FDP, FDQ, FFC, FFX, FIF, FK9, FKD, FMF, FMO, FNG, FNY, FRU, FSA, FSI, FSM, FSW, FUB, FUC, FUD, FUF, FUL, FUY, FVQ, FX1, FYJ, G0S, G16, G1P, G20, G28, G2F, G3F, G3I, G4D, G4S, G6D, G6P, G6S, G7P, G8Z, GAA, GAC, GAD, GAF, GAL, GAT, GBH, GC1, GC4, GC9, GCB, GCD, GCN, GCO, GCS, GCT, GCU, GCV, GCW, GDA, GDL, GE1, GE3, GFP, GIV, GL0, GL1, GL2, GL4, GL5, GL6, GL7, GL9, GLA, GLC, GLD, GLF, GLG, GLO, GLP, GLS, GLT, GM0, GMB, GMH, GMT, GMZ, GN1, GN4, GNS, GNX, GP0, GP1, GP4, GPH, GPK, GPM, GPO, GPQ, GPU, GPV, GPW, GQ1, GRF, GRX, GS1, GS9, GTK, GTM, GTR, GU0, GU1, GU2, GU3, GU4, GU5, GU6, GU8, GU9, GUF, GUL, GUP, GUZ, GXL, GXV, GYE, GYG, GYP, GYU, GYV, GZL, H1M, H1S, H2P, H3S, H53, H6Q, H6Z, HBZ, HD4, HNV, HNW, HSG, HSH, HSJ, HSQ, HSX, HSY, HTG, HTM, HVC, IAB, IDC, IDF, IDG, IDR, IDS, IDU, IDX, IDY, IEM, IN1, IPT, ISD, ISL, ISX, IXD, J5B, JFZ, JHM, JLT, JRV, JSV, JV4, JVA, JVS, JZR, K5B, K99, KBA, KBG, KD5, KDA, KDB, KDD, KDE, KDF, KDM, KDN, KDO, KDR, KFN, KG1, KGM, KHP, KME, KO1, KO2, KOT, KTU, L0W, L1L, L6S, L6T, LAG, LAH, LAI, LAK, LAO, LAT, LB2, LBS, LBT, LCN, LDY, LEC, LER, LFC, LFR, LGC, LGU, LKA, LKS, LM2, LMO, LNV, LOG, LOX, LRH, LTG, LVO, LVZ, LXB, LXC, LXZ, LZ0, M1F, M1P, M2F, M3M, M3N, M55, M6D, M6P, M7B, M7P, M8C, MA1, MA2, MA3, MA8, MAB, MAF, MAG, MAL, MAN, MAT, MAV, MAW, MBE, MBF, MBG, MCU, MDA, MDP, MFB, MFU, MG5, MGC, MGL, MGS, MJJ, MLB, MLR, MMA, MN0, MNA, MQG, MQT, MRH, MRP, MSX, MTT, MUB, MUR, MVP, MXY, MXZ, MYG, N1L, N3U, N9S, NA1, NAA, NAG, NBG, NBX, NBY, NDG, NFG, NG1, NG6, NGA, NGC, NGE, NGK, NGR, NGS, NGY, NGZ, NHF, NLC, NM6, NM9, NNG, NPF, NSQ, NT1, NTF, NTO, NTP, NXD, NYT, OAK, OI7, OPM, OSU, OTG, OTN, OTU, OX2, P53, P6P, P8E, PA1, PAV, PDX, PH5, PKM, PNA, PNG, PNJ, PNW, PPC, PRP, PSG, PSV, PTQ, PUF, PZU, QDK, QIF, QKH, QPS, QV4, R1P, R1X, R2B, R2G, RAE, RAF, RAM, RAO, RB5, RBL, RCD, RER, RF5, RG1, RGG, RHA, RHC, RI2, RIB, RIP, RM4, RP3, RP5, RP6, RR7, RRJ, RRY, RST, RTG, RTV, RUG, RUU, RV7, RVG, RVM, RWI, RY7, RZM, S7P, S81, SA0, SCG, SCR, SDY, SEJ, SF6, SF9, SFU, SG4, SG5, SG6, SG7, SGA, SGC, SGD, SGN, SHB, SHD, SHG, SIA, SID, SIO, SIZ, SLB, SLM, SLT, SMD, SN5, SNG, SOE, SOG, SOL, SOR, SR1, SSG, SSH, STW, STZ, SUC, SUP, SUS, SWE, SZZ, T68, T6D, T6P, T6T, TA6, TAG, TCB, TDG, TEU, TF0, TFU, TGA, TGK, TGR, TGY, TH1, TM5, TM6, TMR, TMX, TNX, TOA, TOC, TQY, TRE, TRV, TS8, TT7, TTV, TU4, TUG, TUJ, TUP, TUR, TVD, TVG, TVM, TVS, TVV, TVY, TW7, TWA, TWD, TWG, TWJ, TWY, TXB, TYV, U1Y, U2A, U2D, U63, U8V, U97, U9A, U9D, U9G, U9J, U9M, UAP, UBH, UBO, UDC, UEA, V3M, V3P, V71, VG1, VJ1, VJ4, VKN, VTB, W9T, WIA, WOO, WUN, WZ1, WZ2, X0X, X1P, X1X, X2F, X2Y, X34, X6X, X6Y, XDX, XGP, XIL, XKJ, XLF, XLS, XMM, XS2, XXM, XXR, XXX, XYF, XYL, XYP, XYS, XYT, XYZ, YDR, YIO, YJM, YKR, YO5, YX0, YX1, YYB, YYH, YYJ, YYK, YYM, YYQ, YZ0, Z0F, Z15, Z16, Z2D, Z2T, Z3K, Z3L, Z3Q, Z3U, Z4K, Z4R, Z4S, Z4U, Z4V, Z4W, Z4Y, Z57, Z5J, Z5L, Z61, Z6H, Z6J, Z6W, Z8H, Z8T, Z9D, Z9E, Z9H, Z9K, Z9L, Z9M, Z9N, Z9W, ZB0, ZB1, ZB2, ZB3, ZCD, ZCZ, ZD0, ZDC, ZDO, ZEE, ZEL, ZGE, ZMR".split(
        ", "
    )
)

# Table 12 of the AlphaFold 3 supplement
ions = set(
    "118, 119, 1AL, 1CU, 2FK, 2HP, 2OF, 3CO, 3MT, 3NI, 3OF, 4MO, 4PU, 4TI, 543, 6MO, AG, AL, ALF, AM, ATH, AU, AU3, AUC, BA, BEF, BF4, BO4, BR, BS3, BSY, CA, CAC, CD, CD1, CD3, CD5, CE, CF, CHT, CO, CO5, CON, CR, CS, CSB, CU, CU1, CU2, CU3, CUA, CUZ, CYN, DME, DMI, DSC, DTI, DY, E4N, EDR, EMC, ER3, EU, EU3, F, FE, FE2, FPO, GA, GD3, GEP, HAI, HG, HGC, HO3, IN, IR, IR3, IRI, IUM, K, KO4, LA, LCO, LCP, LI, LU, MAC, MG, MH2, MH3, MMC, MN, MN3, MN5, MN6, MO, MO1, MO2, MO3, MO4, MO5, MO6, MOO, MOS, MOW, MW1, MW2, MW3, NA2, NA5, NA6, NAO, NAW, NET, NI, NI1, NI2, NI3, NO2, NRU, O4M, OAA, OC1, OC2, OC3, OC4, OC5, OC6, OC7, OC8, OCL, OCM, OCN, OCO, OF1, OF2, OF3, OH, OS, OS4, OXL, PB, PBM, PD, PER, PI, PO3, PR, PT, PT4, PTN, RB, RH3, RHD, RU, SB, SE4, SEK, SM, SMO, SO3, T1A, TB, TBA, TCN, TEA, TH, THE, TL, TMA, TRA, V, VN3, VO4, W, WO5, Y1, YB, YB2, YH, YT3, ZCM, ZN, ZN2, ZN3, ZNO, ZO3, ZR".split(
        ", "
    )
)

# Table 13 of the AlphaFold 3 supplement
standard_residues = set(
    "ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL, UNK, A, G, C, U, DA, DG, DC, DT, N, DN".split(
        ", "
    )
)

# Table 14 of the AlphaFold 3 supplement
recent_pdb_test_set_with_nucleic_acid_complexes = set(
    "7B0C, 7BCA, 7BJQ, 7EDS, 7EOF, 7F3J, 7F8Z, 7F9H, 7M4L, 7MKT, 7MWH, 7MZ0, 7MZ1, 7MZ2, 7N5U, 7N5V, 7N5W, 7NQF, 7NRP, 7OGS, 7OOO, 7OOS, 7OOT, 7OUE, 7OWF, 7OY7, 7OZZ, 7P0W, 7P3F, 7P8L, 7P9J, 7P9Z, 7PSX, 7PTQ, 7PZA, 7PZB, 7Q3O, 7Q4N, 7Q94, 7QAZ, 7QP2, 7R6R, 7R6T, 7R8G, 7R8H, 7R8I, 7RCC, 7RCD, 7RCE, 7RCF, 7RCG, 7RCU, 7RGU, 7RSR, 7RSS, 7S03, 7S68, 7S9J, 7S9K, 7S9L, 7S9M, 7S9N, 7S9O, 7S9P, 7S9Q, 7SOP, 7SOS, 7SOT, 7SOU, 7SOV, 7SOW, 7SUM, 7SUV, 7SVB, 7SX5, 7SXE, 7T18, 7T19, 7T1A, 7T1B, 7T8K, 7TDW, 7TDX, 7TEA, 7TEC, 7TO1, 7TO2, 7TQW, 7TUV, 7TXC, 7TZ1, 7TZR, 7TZS, 7TZT, 7TZU, 7TZV, 7U76, 7U79, 7U7A, 7U7B, 7U7C, 7U7F, 7U7G, 7U7I, 7U7J, 7U7K, 7U7L, 7UBL, 7UBU, 7UCR, 7UPZ, 7UQ6, 7UR5, 7URI, 7URM, 7UU4, 7UXD, 7UZ0, 7V2Z, 7VE5, 7VFT, 7VG8, 7VKI, 7VKL, 7VN2, 7VNV, 7VNW, 7VO9, 7VOU, 7VOV, 7VOX, 7VP1, 7VP2, 7VP3, 7VP4, 7VP5, 7VP7, 7VSJ, 7VTI, 7WM3, 7WQ5, 7X5E, 7X5F, 7X5G, 7X5L, 7X5M, 7XHV, 7XI3, 7XQ5, 7XRC, 7XS4, 7YHO, 7YZE, 7YZF, 7YZG, 7Z0U, 7Z5A, 7ZHH, 7ZVN, 7ZVX, 8A1C, 8A4I, 8AMG, 8AMI, 8AMJ, 8AMK, 8AML, 8AMM, 8AMN, 8B0R, 8CSH, 8CTZ, 8CU0, 8CZQ, 8D28, 8D2A, 8D2B, 8D5L, 8D5O, 8DVP, 8DVR, 8DVS, 8DVU, 8DVY, 8DW0, 8DW1, 8DW4, 8DW8, 8DWM, 8DZK, 8E2P, 8E2Q, 8EDJ, 8EF9, 8EFC, 8EFK, 8GMS, 8GMT, 8GMU".split(
        ", "
    )
)

# Table 15 of the AlphaFold 3 supplement
posebusters_v2_common_natural_ligands = set(
    "2BA, 5AD, A3P, ACP, ADP, AKG, ANP, APC, APR, ATP, BCN, BDP, BGC, C5P, CDP, CTP, DGL, DSG, F15, FAD, FDA, FMN, GSH, GSP, GTP, H4B, IPE, MFU, MTA, MTE, NAD, NAI, NCA, NGA, OGA, PGA, PHO, PJ8, PLG, PLP, PRP, SAH, SFG, SIN, SLB, TPP, UD1, UDP, UPG, URI".split(
        ", "
    )
)

# Process all structures #
file_paths = glob.glob(os.path.join(args.mmcif_dir, "*", "*.cif"))
processed_structures = process_structures(
    file_paths,
    standard_residues,
    ligand_exclusion_list,
    ccd_atoms,
    covalent_ligands,
    crystallography_methods,
)

# Save processed structures #
io = PDBIO()
for structure in processed_structures:
    io.set_structure(structure)
    io.save(os.path.join(args.output_dir, f"{structure.id}.cif"))
