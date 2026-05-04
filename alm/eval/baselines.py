"""Cited baseline numbers for paper headline tables.

Sourced from each benchmark's published paper. Do NOT rerun these — cite via
table footnotes. Schema: BASELINES[benchmark][model][task][metric] = value.

Aggregator emits the ALM column even when baselines are missing; missing
entries render as "—" in the LaTeX table.
"""

BASELINES = {
    # LLM4Mat-Bench (Rubungo et al., arxiv.org/abs/2411.00177, Tables 4-5 + Appendix E).
    # Metric: MAD:MAE ratio (regression). Higher is better; >5 = "good model".
    # Note: paper does not evaluate Llama-3-8B-Instruct directly — Llama-2-7b-chat
    # is the only Llama variant in their table. Composition input throughout.
    "llm4mat": {
        "CGCNN": {  # CIF input
            ("mp", "formation_energy_per_atom"):       {"mad_mae": 5.32},
            ("mp", "band_gap"):                         {"mad_mae": 3.26},
            ("mp", "energy_per_atom"):                  {"mad_mae": 7.22},
            ("mp", "energy_above_hull"):                {"mad_mae": 3.87},
            ("mp", "density"):                          {"mad_mae": 8.77},
            ("jarvis_dft", "formation_energy_peratom"): {"mad_mae": 13.62},
            ("jarvis_dft", "optb88vdw_bandgap"):        {"mad_mae": 4.80},
            ("jarvis_dft", "ehull"):                    {"mad_mae": 1.57},
            ("oqmd", "e_form"):                         {"mad_mae": 22.29},
            ("oqmd", "bandgap"):                        {"mad_mae": 6.70},
            ("gnome", "Formation_Energy_Per_Atom"):     {"mad_mae": 34.57},
            ("gnome", "Bandgap"):                       {"mad_mae": 8.55},
        },
        "MatBERT-109.5M": {  # composition input
            ("mp", "formation_energy_per_atom"):       {"mad_mae": 5.32},
            ("mp", "band_gap"):                         {"mad_mae": 2.97},
            ("mp", "energy_per_atom"):                  {"mad_mae": 9.32},
            ("mp", "energy_above_hull"):                {"mad_mae": 2.58},
            ("mp", "density"):                          {"mad_mae": 7.63},
            ("jarvis_dft", "formation_energy_peratom"): {"mad_mae": 6.81},
            ("jarvis_dft", "optb88vdw_bandgap"):        {"mad_mae": 4.08},
            ("jarvis_dft", "ehull"):                    {"mad_mae": 2.79},
            ("oqmd", "e_form"):                         {"mad_mae": 7.66},
            ("oqmd", "bandgap"):                        {"mad_mae": 3.88},
            ("gnome", "Formation_Energy_Per_Atom"):     {"mad_mae": 30.25},
            ("gnome", "Bandgap"):                       {"mad_mae": 4.69},
        },
        "LLM-Prop-35M": {  # composition input
            ("mp", "formation_energy_per_atom"):       {"mad_mae": 4.39},
            ("mp", "band_gap"):                         {"mad_mae": 2.35},
            ("mp", "energy_per_atom"):                  {"mad_mae": 7.44},
            ("mp", "energy_above_hull"):                {"mad_mae": 2.01},
            ("mp", "density"):                          {"mad_mae": 6.68},
            ("jarvis_dft", "formation_energy_peratom"): {"mad_mae": 4.77},
            ("jarvis_dft", "optb88vdw_bandgap"):        {"mad_mae": 2.62},
            ("jarvis_dft", "ehull"):                    {"mad_mae": 2.07},
            ("oqmd", "e_form"):                         {"mad_mae": 9.20},
            ("oqmd", "bandgap"):                        {"mad_mae": 2.85},
            ("gnome", "Formation_Energy_Per_Atom"):     {"mad_mae": 25.47},
            ("gnome", "Bandgap"):                       {"mad_mae": 3.74},
        },
        "Llama-2-7b-chat-0S": {
            ("mp", "formation_energy_per_atom"):       {"mad_mae": 0.39},
            ("mp", "band_gap"):                         {"mad_mae": 0.62},
            ("mp", "density"):                          {"mad_mae": 0.97},
        },
        "Llama-2-7b-chat-5S": {
            ("mp", "formation_energy_per_atom"):       {"mad_mae": 0.63},
            ("mp", "band_gap"):                         {"mad_mae": 1.22},
            ("mp", "density"):                          {"mad_mae": 0.90},
        },
        "Mistral-7b-Instruct-v0.1-5S": {
            ("mp", "formation_energy_per_atom"):       {"mad_mae": 0.109},
            ("mp", "band_gap"):                         {"mad_mae": 0.898},
            ("mp", "energy_per_atom"):                  {"mad_mae": 0.345},
            ("mp", "energy_above_hull"):                {"mad_mae": 0.164},
            ("mp", "density"):                          {"mad_mae": 0.255},
            ("jarvis_dft", "formation_energy_peratom"): {"mad_mae": 0.128},
            ("jarvis_dft", "optb88vdw_bandgap"):        {"mad_mae": 0.011},
            ("jarvis_dft", "mbj_bandgap"):              {"mad_mae": 0.513},
            ("jarvis_dft", "ehull"):                    {"mad_mae": 0.171},
            ("oqmd", "e_form"):                         {"mad_mae": 0.723},
            ("oqmd", "bandgap"):                        {"mad_mae": 0.479},
            ("gnome", "Formation_Energy_Per_Atom"):     {"mad_mae": 0.192},
            ("gnome", "Bandgap"):                       {"mad_mae": 0.242},
            ("gnome", "Decomposition_Energy_Per_Atom"): {"mad_mae": 0.022},
            ("gnome", "Density"):                       {"mad_mae": 0.186},
            ("snumat", "Band_gap_HSE"):                 {"mad_mae": 0.468},
            ("snumat", "Band_gap_GGA"):                 {"mad_mae": 0.794},
            ("hmof", "max_co2_adsp"):                   {"mad_mae": 0.451},
            ("hmof", "void_fraction"):                  {"mad_mae": 0.033},
            ("hmof", "surface_area_m2g"):               {"mad_mae": 0.567},
            ("hmof", "lcd"):                            {"mad_mae": 0.458},
            ("hmof", "pld"):                            {"mad_mae": 0.735},
            ("cantor_hea", "Ef_per_atom"):              {"mad_mae": 0.387},
            ("cantor_hea", "e_above_hull"):             {"mad_mae": 0.220},
            ("cantor_hea", "volume_per_atom"):          {"mad_mae": 0.134},
            ("jarvis_qetb", "energy_per_atom"):         {"mad_mae": 0.878},
            ("jarvis_qetb", "indir_gap"):               {"mad_mae": 0.394},
            ("jarvis_qetb", "f_enp"):                   {"mad_mae": 0.693},
            ("omdb", "bandgap"):                        {"mad_mae": 0.222},
            ("qmof", "bandgap"):                        {"mad_mae": 0.236},
            ("qmof", "lcd"):                            {"mad_mae": 0.807},
            ("qmof", "pld"):                            {"mad_mae": 0.254},
        },
        "Gemma-2-9b-it-5S": {
            ("mp", "formation_energy_per_atom"):       {"mad_mae": 0.519},
            ("mp", "band_gap"):                         {"mad_mae": 0.999},
            ("mp", "energy_per_atom"):                  {"mad_mae": 0.482},
            ("mp", "energy_above_hull"):                {"mad_mae": 0.088},
            ("mp", "density"):                          {"mad_mae": 1.368},
            ("jarvis_dft", "formation_energy_peratom"): {"mad_mae": 0.150},
            ("jarvis_dft", "optb88vdw_bandgap"):        {"mad_mae": 0.011},
            ("jarvis_dft", "mbj_bandgap"):              {"mad_mae": 0.973},
            ("jarvis_dft", "ehull"):                    {"mad_mae": 0.601},
            ("oqmd", "e_form"):                         {"mad_mae": 0.682},
            ("oqmd", "bandgap"):                        {"mad_mae": 0.613},
            ("gnome", "Formation_Energy_Per_Atom"):     {"mad_mae": 0.182},
            ("gnome", "Bandgap"):                       {"mad_mae": 0.186},
            ("gnome", "Decomposition_Energy_Per_Atom"): {"mad_mae": 0.046},
            ("gnome", "Density"):                       {"mad_mae": 1.07},
            ("snumat", "Band_gap_HSE"):                 {"mad_mae": 1.168},
            ("snumat", "Band_gap_GGA"):                 {"mad_mae": 0.741},
            ("hmof", "max_co2_adsp"):                   {"mad_mae": 0.679},
            ("hmof", "void_fraction"):                  {"mad_mae": 0.765},
            ("hmof", "surface_area_m2g"):               {"mad_mae": 0.567},
            ("hmof", "lcd"):                            {"mad_mae": 0.971},
            ("hmof", "pld"):                            {"mad_mae": 0.973},
            ("cantor_hea", "Ef_per_atom"):              {"mad_mae": 0.148},
            ("cantor_hea", "e_above_hull"):             {"mad_mae": 0.964},
            ("cantor_hea", "volume_per_atom"):          {"mad_mae": 0.268},
            ("jarvis_qetb", "energy_per_atom"):         {"mad_mae": 0.989},
            ("jarvis_qetb", "indir_gap"):               {"mad_mae": 0.397},
            ("jarvis_qetb", "f_enp"):                   {"mad_mae": 0.608},
            ("omdb", "bandgap"):                        {"mad_mae": 0.967},
            ("qmof", "bandgap"):                        {"mad_mae": 1.053},
            ("qmof", "lcd"):                            {"mad_mae": 0.913},
            ("qmof", "pld"):                            {"mad_mae": 1.048},
        },
    },

    # MatterChat MP held-out (Tang et al., arxiv.org/abs/2502.13107, Table 1).
    # Six classification (acc) + three regression (RMSE) tasks. Paper does NOT
    # publish numbers for GPT-4 / Gemini / DeepSeek on these tasks (Figure 5
    # text says LLM errors were "excessively large", excluded from table).
    "matterchat": {
        "Simple-Adapter+LoRA": {
            "metallic":         {"acc": 0.6373},
            "direct_bandgap":   {"acc": 0.8629},
            "stability":        {"acc": 0.7418},
            "exp_observed":     {"acc": 0.7171},
            "is_magnetic":      {"acc": 0.8339},
            "magnetic_order":   {"acc": 0.7759},
            "formation_energy": {"rmse": 0.4105},
            "energy_above_hull":{"rmse": 0.4415},
            "bandgap":          {"rmse": 1.2516},
        },
        "LoRA-LLM-only": {
            "metallic":         {"acc": 0.6864},
            "direct_bandgap":   {"acc": 0.7839},
            "stability":        {"acc": 0.7944},
            "exp_observed":     {"acc": 0.6549},
            "is_magnetic":      {"acc": 0.6833},
            "magnetic_order":   {"acc": 0.4238},
            "formation_energy": {"rmse": 1.8059},
            "energy_above_hull":{"rmse": 0.4051},
            "bandgap":          {"rmse": 1.4725},
        },
        "MatterChat": {
            "metallic":         {"acc": 0.8683},
            "direct_bandgap":   {"acc": 0.8753},
            "stability":        {"acc": 0.8515},
            "exp_observed":     {"acc": 0.8504},
            "is_magnetic":      {"acc": 0.9368},
            "magnetic_order":   {"acc": 0.8570},
            "formation_energy": {"rmse": 0.1500},
            "energy_above_hull":{"rmse": 0.1053},
            "bandgap":          {"rmse": 0.5590},
        },
        "MatterChat+RAG": {
            "metallic":         {"acc": 0.8873},
            "direct_bandgap":   {"acc": 0.8797},
            "stability":        {"acc": 0.8573},
            "exp_observed":     {"acc": 0.8570},
            "is_magnetic":      {"acc": 0.9333},
            "magnetic_order":   {"acc": 0.8535},
            "formation_energy": {"rmse": 0.1212},
            "energy_above_hull":{"rmse": 0.0964},
            "bandgap":          {"rmse": 0.5058},
        },
    },

    # GNoME formation-energy held-out (MatterChat Figure 2(b)). Paper reports only
    # qualitative comparison; no MAE table for Gemini/GPT-4o/DeepSeek.
    "gnome_fe": {},

    # MatText (Alampara et al., arxiv.org/abs/2406.17295, Table 2). MAE.
    # Perovskites in eV; KVRH/GVRH in log10(modulus).
    "mattext": {
        "BERT-Composition":     {"perovskites": {"mae": 0.099}, "kvrh": {"mae": 0.149}, "gvrh": {"mae": 0.144}},
        "BERT-SLICES":          {"perovskites": {"mae": 0.099}, "kvrh": {"mae": 0.149}, "gvrh": {"mae": 0.144}},
        "BERT-CIF-P1":          {"perovskites": {"mae": 0.095}, "kvrh": {"mae": 0.154}, "gvrh": {"mae": 0.136}},
        "BERT-CIF-Symmetrized": {"perovskites": {"mae": 0.109}, "kvrh": {"mae": 0.153}, "gvrh": {"mae": 0.153}},
        "BERT-Z-Matrix":        {"perovskites": {"mae": 0.095}, "kvrh": {"mae": 0.152}, "gvrh": {"mae": 0.154}},
        "BERT-Local-Env":       {"perovskites": {"mae": 0.098}, "kvrh": {"mae": 0.154}},
        "Llama-Composition":     {"perovskites": {"mae": 0.294}, "kvrh": {"mae": 0.460}, "gvrh": {"mae": 0.288}},
        "Llama-SLICES":          {"perovskites": {"mae": 0.294}, "kvrh": {"mae": 0.460}, "gvrh": {"mae": 0.288}},
        "Llama-CIF-P1":          {"perovskites": {"mae": 0.181}, "kvrh": {"mae": 0.315}, "gvrh": {"mae": 0.343}},
        "Llama-CIF-Symmetrized": {"perovskites": {"mae": 0.225}, "kvrh": {"mae": 0.402}, "gvrh": {"mae": 0.342}},
        "Llama-Z-Matrix":        {"perovskites": {"mae": 0.286}, "kvrh": {"mae": 0.219}, "gvrh": {"mae": 0.382}},
        "Llama-Local-Env":       {"perovskites": {"mae": 0.410}, "kvrh": {"mae": 0.480}, "gvrh": {"mae": 0.329}},
    },

    # Park et al. Mat2Props (Sci Data 2024, doi 10.1038/s41597-024-03886-w, Table 3).
    # Property prediction MAE on the 10% Materials Project held-out. MAD column gives
    # the dataset spread baseline (0.93 eV/at for Ef, 1.35 eV for Eg).
    # MEGNet's Eg cell is em-dashed in the paper.
    "mat2props": {
        "CFID":    {"formation_energy_per_atom": {"mae": 0.104}, "band_gap": {"mae": 0.434}},
        "CGCNN":   {"formation_energy_per_atom": {"mae": 0.039}, "band_gap": {"mae": 0.388}},
        "MEGNet":  {"formation_energy_per_atom": {"mae": 0.028}, "band_gap": {"mae": 0.330}},
        "SchNet":  {"formation_energy_per_atom": {"mae": 0.035}},
        "ALIGNN":  {"formation_energy_per_atom": {"mae": 0.022}, "band_gap": {"mae": 0.218}},
        "GPT-3.5": {"formation_energy_per_atom": {"mae": 1.897}, "band_gap": {"mae": 1.309}},
    },

    # Park et al. Mat2MCQ (Fig 5d). Paper presents results in a bar chart, not a
    # table; numerical values not extractable from the PDF text. Skip baselines
    # here — comparison will rely on the ALM column alone.
    "mat2mcq": {},

    # MaScQA (Zaki et al., arxiv.org/abs/2308.09115, Digital Discovery 2024 D3DD00188A,
    # Table 2). 650-question benchmark. Numbers in % accuracy.
    "mascqa": {
        "GPT-4":            {"all": {"acc": 0.6138}, "MCQ": {"acc": 0.7465},
                             "MATCH": {"acc": 0.8857}, "MCQN": {"acc": 0.5882},
                             "NUM": {"acc": 0.3728}},
        "GPT-4-CoT":        {"all": {"acc": 0.6262}, "MCQ": {"acc": 0.7711},
                             "MATCH": {"acc": 0.9286}, "MCQN": {"acc": 0.5000},
                             "NUM": {"acc": 0.3904}},
        "GPT-3.5":          {"all": {"acc": 0.3831}, "MCQ": {"acc": 0.5669},
                             "MATCH": {"acc": 0.4000}, "MCQN": {"acc": 0.3529},
                             "NUM": {"acc": 0.1579}},
        "GPT-3.5-CoT":      {"all": {"acc": 0.3785}, "MCQ": {"acc": 0.5704},
                             "MATCH": {"acc": 0.3857}, "MCQN": {"acc": 0.3382},
                             "NUM": {"acc": 0.1491}},
        "Llama-2-70B-CoT":  {"all": {"acc": 0.2400}, "MCQ": {"acc": 0.4120},
                             "MATCH": {"acc": 0.2286}, "MCQN": {"acc": 0.2059},
                             "NUM": {"acc": 0.0395}},
    },

    # Language retention — no published baselines; ALM and Qwen3-8B base both
    # run through eval_language_retention.py for clean comparison.
    "language_retention": {},

    # ───────────────────────────────────────────────────────────────────────
    # Stage 3b crystal-generation baselines.
    # Verbatim numbers from each paper's headline table; cite via footnote.
    # ───────────────────────────────────────────────────────────────────────

    # CSP eval — MP-20 (CDVAE/CrystaLLM split, n=1 and n=20 generations per
    # composition, OrderedStructureMatcher with ltol=0.3, stol=0.5, angle_tol=10).
    # Source: CrystaLLM Table 3 (Antunes et al., Nat Commun 15, 10570 (2024));
    # OMatG Table on MP-20 CSP (arXiv 2502.02582);
    # DiffCSP (Jiao et al., NeurIPS 2023).
    "stage3b_csp_mp_20_n1_g10": {
        "CrystaLLM-large": {"csp": {"match_rate": 0.5870, "rmse": 0.0408}},
        "OMatG-Linear-ODE": {"csp": {"match_rate": 0.6375, "rmse": None}},
    },
    "stage3b_csp_mp_20_n20_g10": {
        "CrystaLLM-large": {"csp": {"match_rate": 0.7397, "rmse": 0.0349}},
        "DiffCSP":          {"csp": {"match_rate": 0.7793, "rmse": 0.0492}},
        "OMatG-Linear-ODE": {"csp": {"match_rate": 0.6983, "rmse": None}},
    },

    # CSP eval — MPTS-52 (DiffCSP split). Same matcher convention.
    "stage3b_csp_mpts_52_n1_g10": {
        "CrystaLLM-large": {"csp": {"match_rate": 0.1921, "rmse": 0.1110}},
        "OMatG-Linear-ODE": {"csp": {"match_rate": 0.2515, "rmse": None}},
    },
    "stage3b_csp_mpts_52_n20_g10": {
        "CrystaLLM-large": {"csp": {"match_rate": 0.3375, "rmse": 0.1059}},
        "DiffCSP":          {"csp": {"match_rate": 0.3402, "rmse": 0.1749}},
        "OMatG-Linear-ODE": {"csp": {"match_rate": 0.2738, "rmse": None}},
    },

    # De-novo (DNG) eval — MatterGen-style headline. Sources:
    #   MatterGen (Zeni et al., Nature 2025) — % stable / % metastable / S.U.N. on
    #     Alex-MP-ICSD hull. We report MP-hull-equivalent figures where given.
    #   Crystal-text-LLM Table 1 (LLaMA-2-70B): metastable via M3GNet, stable via DFT.
    #   OMatG (Linear SDE+γ) Table on MP-20 DNG.
    # Numbers reported as fractions in [0, 1].
    "stage3b_dng_g00": {
        "MatterGen":         {"dng": {"sun": 0.3857, "metastable": 0.78, "stable": 0.13,
                                       "rmsd_to_relaxed_min": 0.021}},
        "Crystal-text-LLM-70B": {"dng": {"validity_geom": 0.996, "metastable": 0.498,
                                          "stable": 0.106}},
        "OMatG-LinearSDE-gamma": {"dng": {"sun": 0.2248, "stability": 0.4618,
                                           "novelty": 0.7331,
                                           "rmsd_to_relaxed_min": 0.6357}},
        "DiffCSP":           {"dng": {"sun": 0.1595, "stability": 0.4343,
                                       "rmsd_to_relaxed_min": 0.3861}},
        "MatterGen-MP":      {"dng": {"sun": 0.2030, "stability": 0.4479,
                                       "rmsd_to_relaxed_min": 0.1038}},
    },

    # Text-conditional eval. No prior published numbers — Chemeleon reports
    # composition-matching only on a structurally-conditioned setup that's not
    # directly comparable. We emit the ALM column alone here; the
    # `mclm-Stage3b` row will appear when aggregate_results joins the run.
    "stage3b_text_cond_g10": {
        "Chemeleon": {"text_cond": {"composition_match_ratio": 0.6752}},
    },
}


# Crystal-generation papers' citation strings (for paper bibliography).
CRYSTAL_GEN_CITATIONS = {
    "CrystaLLM": "Antunes et al., Nat Commun 15, 10570 (2024). arXiv:2307.04340",
    "MatterGen": "Zeni et al., Nature 639, 624 (2025). arXiv:2312.03687",
    "Crystal-text-LLM": "Gruver et al., ICLR 2024. arXiv:2402.04379",
    "Chemeleon": "Park et al., Nat Commun 2025. doi:10.1038/s41467-025-59636-y",
    "OMatG": "Hassan et al., 2025. arXiv:2502.02582",
    "CDVAE": "Xie et al., ICLR 2022.",
    "DiffCSP": "Jiao et al., NeurIPS 2023.",
}
