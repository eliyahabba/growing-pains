"""
Model Matcher - Interactive tool for matching models between different data sources.

This app helps map model names between different sources (pickle vs parquet)
that use different naming conventions for the same models.

Mappings are saved PER FILE PAIR - each (source_file, target_file) combination
has its own separate mapping.

Usage:
    streamlit run model_matcher.py
"""

import pickle
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# Configuration
# =============================================================================

# File is at: src/experiments/utils/model_matcher.py
# Need to go up 3 levels to reach project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TINYBENCHMARKS_DIR = PROJECT_ROOT / "aggregated_data/tinybenchmarks"
AGGREGATED_DIR = PROJECT_ROOT / "aggregated_data/aggregated"
MAPPINGS_DIR = PROJECT_ROOT / "src/experiments/model_mappings"


# =============================================================================
# Data Loading
# =============================================================================

@st.cache_data(show_spinner="Loading data sources...")
def load_all_sources():
    """Load all model sources from both pickle and parquet files.
    
    Pickle files (tinybenchmarks/) - used in current experiments:
    - lb.pickle: 395 models (Open LLM Leaderboard)
    - helm_lite.pickle: 30 models
    - mmlu_fields.pickle: 428 models
    
    Parquet files (aggregated/) - for future use:
    - helm_lite_aggregated.parquet: 91 models (more than pickle!)
    - helm_classic_aggregated.parquet: 70 models, 59 datasets
    - helm_mmlu_aggregated.parquet: 79 models
    
    NOTE: Model naming conventions differ between sources!
    - Pickle: uses underscore (e.g., '01-ai_yi-34b')
    - Parquet: uses slash (e.g., '01-ai/yi-34b')
    """
    sources = {}
    
    # Pickle files used in run_all_balanced.sh experiments
    RELEVANT_PICKLES = ['lb.pickle', 'helm_lite.pickle', 'mmlu_fields.pickle']
    
    # Load pickles
    for p in sorted(TINYBENCHMARKS_DIR.glob('*.pickle')):
        if p.name not in RELEVANT_PICKLES:
            continue
        with open(p, 'rb') as f:
            data = pickle.load(f)
        models = list(np.array(data.get('models', [])).flatten())
        models_sorted = sorted([str(m) for m in models])
        sources[p.name] = {'type': 'pickle', 'models': models_sorted, 'count': len(models_sorted)}
    
    # Load all parquet files (for potential future use)
    for p in sorted(AGGREGATED_DIR.glob('*.parquet')):
        df = pd.read_parquet(p)
        models = list(df['model_name'].unique())
        models_sorted = sorted(models)
        sources[p.name] = {'type': 'parquet', 'models': models_sorted, 'count': len(models_sorted)}
    
    return sources


def get_mapping_file(source_file: str, target_file: str) -> Path:
    """Get the mapping file path for a specific file pair."""
    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    # Create a clean filename from the pair
    safe_source = source_file.replace('.', '_').replace('/', '_')
    safe_target = target_file.replace('.', '_').replace('/', '_')
    return MAPPINGS_DIR / f"{safe_source}__TO__{safe_target}.json"


def load_mappings(source_file: str, target_file: str) -> dict:
    """Load existing mappings for a specific file pair."""
    mapping_file = get_mapping_file(source_file, target_file)
    if mapping_file.exists():
        with open(mapping_file) as f:
            return json.load(f)
    return {}


def auto_match_models(source_models: list, target_models: list, threshold: float = 1.0) -> dict:
    """Automatically match models that have similarity >= threshold.
    
    Returns dict of {source_model: target_model} for matches.
    """
    matches = {}
    
    # Build normalized lookup for targets
    target_lookup = {}
    for t in target_models:
        norm = normalize_model_name(t)
        target_lookup[norm] = t
    
    for source in source_models:
        source_norm = normalize_model_name(source)
        
        # Check for exact normalized match (100%)
        if source_norm in target_lookup:
            matches[source] = target_lookup[source_norm]
            continue
        
        # If threshold < 1.0, also check similarity scores
        if threshold < 1.0:
            best_match = None
            best_score = 0
            for target_norm, target_orig in target_lookup.items():
                score = SequenceMatcher(None, source_norm, target_norm).ratio()
                if score >= threshold and score > best_score:
                    best_score = score
                    best_match = target_orig
            if best_match:
                matches[source] = best_match
    
    return matches


def save_mappings(mappings: dict, source_file: str, target_file: str):
    """Save mappings for a specific file pair."""
    mapping_file = get_mapping_file(source_file, target_file)
    with open(mapping_file, 'w') as f:
        json.dump(mappings, f, indent=2, ensure_ascii=False)


# =============================================================================
# Matching Logic
# =============================================================================

def clean_model_name_for_display(model_name: str) -> str:
    """Clean model name for display by removing common prefixes.
    
    Handles both pickle naming (underscore) and parquet naming (slash).
    
    Example transformations:
    - 'open-llm-leaderboard/details_mistralai__Mixtral-8x7B-v0.1' 
      → 'mistralai/Mixtral-8x7B-v0.1'
    """
    name = str(model_name)
    
    # Remove common prefixes that add clutter
    prefixes_to_remove = [
        'open-llm-leaderboard/details_',
        'open-llm-leaderboard/',
    ]
    
    for prefix in prefixes_to_remove:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break  # Only remove one prefix
    
    # Convert double underscore to slash (LB naming convention)
    # 'mistralai__Mixtral-8x7B-v0.1' → 'mistralai/Mixtral-8x7B-v0.1'
    if '__' in name:
        name = name.replace('__', '/')
    
    return name


def normalize_model_name(model_name: str) -> str:
    """Normalize model name for matching (convert to common format).
    
    Handles different naming conventions:
    - LB: 'open-llm-leaderboard/details_mistralai__Mixtral-8x7B-v0.1'
    - HELM: 'mistralai/mixtral-8x7b-32kseqlen'
    
    Normalizes to: 'mistralai_mixtral-8x7b-v0.1' (lowercase, underscores)
    """
    name = str(model_name).lower()
    
    # Remove common prefixes first
    prefixes = ['open-llm-leaderboard/details_', 'open-llm-leaderboard_details_', 'open-llm-leaderboard_', 'open-llm-leaderboard/']
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    
    # Convert double underscore to single (LB convention: org__model)
    name = name.replace('__', '_')
    
    # Replace slashes with underscores for uniform comparison
    name = name.replace('/', '_')
    
    return name


def extract_keywords(model_name: str) -> set:
    """Extract keywords from a model name for matching."""
    name = str(model_name).lower()
    
    # Remove common prefixes
    prefixes = [
        'open-llm-leaderboard/details_',
        'openai/', 'anthropic/', 'google/', 'meta/', 'mistralai/',
        'ai21/', 'amazon/', 'cohere/', 'qwen/', 'allenai/',
        'microsoft/', 'deepseek-ai/', 'writer/',
    ]
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
    
    # Split on common separators
    parts = re.split(r'[-_/:]', name)
    
    # Filter out very short parts and common suffixes
    keywords = set()
    for part in parts:
        part = part.strip()
        if len(part) >= 2:
            # Remove version numbers at the end
            part = re.sub(r'v?\d+(\.\d+)*$', '', part)
            if part and len(part) >= 2:
                keywords.add(part)
    
    return keywords


def similarity_score(name1: str, name2: str) -> float:
    """Calculate similarity between two model names.
    
    Handles different naming conventions (pickle vs parquet).
    """
    # First check for exact match after normalization
    norm1 = normalize_model_name(name1)
    norm2 = normalize_model_name(name2)
    
    if norm1 == norm2:
        return 1.0  # Perfect match despite different formatting
    
    # Keyword overlap
    kw1 = extract_keywords(name1)
    kw2 = extract_keywords(name2)
    
    if not kw1 or not kw2:
        return 0.0
    
    overlap = len(kw1 & kw2)
    union = len(kw1 | kw2)
    jaccard = overlap / union if union > 0 else 0.0
    
    # Sequence matching on normalized names
    seq_ratio = SequenceMatcher(None, norm1, norm2).ratio()
    
    # Combined score
    return 0.5 * jaccard + 0.5 * seq_ratio


def find_candidates(source_model: str, target_models: list, top_k: int = 10) -> list:
    """Find top-k candidate matches from target models."""
    scores = []
    for target in target_models:
        score = similarity_score(source_model, target)
        if score > 0.1:  # Minimum threshold
            scores.append((target, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def get_model_family(model_name: str) -> str:
    """Identify the model family/company."""
    name = str(model_name).lower()
    
    families = {
        'OpenAI': ['openai', 'gpt-3', 'gpt-4', 'gpt4', 'gpt3', 'davinci', 'curie', 'babbage'],
        'Anthropic': ['anthropic', 'claude'],
        'Meta': ['meta/', 'llama', 'meta_'],
        'Google': ['google', 'gemini', 'palm', 'bison', 'unicorn', 'gemma'],
        'Mistral': ['mistral', 'mixtral'],
        'Microsoft': ['microsoft', 'phi-'],
        'Amazon': ['amazon', 'nova'],
        'AI21': ['ai21', 'jamba', 'j1-', 'j2-'],
        'Cohere': ['cohere', 'command'],
        '01.AI': ['01-ai', 'yi-', 'yi_'],
        'AlephAlpha': ['alephalpha', 'luminous'],
        'DeepSeek': ['deepseek'],
        'Qwen': ['qwen'],
        'AllenAI': ['allenai', 'olmo'],
    }
    
    for family, keywords in families.items():
        for kw in keywords:
            if kw in name:
                return family
    
    return 'Other'


# =============================================================================
# Streamlit App
# =============================================================================

def main():
    st.set_page_config(
        page_title="Model Matcher",
        page_icon="🔗",
        layout="wide",
    )
    
    st.title("🔗 Model Matcher")
    st.markdown("Match models between different data sources. **Mappings are saved per file pair.**")
    
    # Load data
    sources = load_all_sources()
    
    # Sidebar - Source selection
    st.sidebar.header("📁 Source Selection")
    
    source_names = list(sources.keys())
    
    if not source_names:
        st.error("No data sources found in aggregated_data folders!")
        st.info(f"Checked: \n- {TINYBENCHMARKS_DIR}\n- {AGGREGATED_DIR}")
        return
    
    # Source (smaller group)
    source_file = st.sidebar.selectbox(
        "Source (to match FROM)",
        source_names,
        index=source_names.index('helm_lite_aggregated.parquet') if 'helm_lite_aggregated.parquet' in source_names else 0,
        key="source_file_select"
    )
    
    # Target (larger group)
    target_options = [n for n in source_names if n != source_file]
    if not target_options:
        st.warning("Need at least two data sources to perform matching.")
        return

    target_file = st.sidebar.selectbox(
        "Target (to match TO)",
        target_options,
        index=0,
        key="target_file_select"
    )
    
    if not source_file or not target_file:
        st.error("Please select both source and target files.")
        st.stop()
    
    # Load mappings for this specific file pair
    pair_key = f"{source_file}|{target_file}"
    if 'current_pair' not in st.session_state or st.session_state.current_pair != pair_key:
        st.session_state.current_pair = pair_key
        st.session_state.mappings = load_mappings(source_file, target_file)
        st.session_state.current_idx = 0
    
    source_models = sources[source_file]['models']
    target_models = sources[target_file]['models']
    
    # Debug: verify counts match
    expected_source_count = sources[source_file].get('count', len(source_models))
    expected_target_count = sources[target_file].get('count', len(target_models))
    
    st.sidebar.markdown(f"""
    **Source**: {source_file}  
    → {len(source_models)} models (expected: {expected_source_count})  
    **Target**: {target_file}  
    → {len(target_models)} models (expected: {expected_target_count})  
    **Mapping file**: `{get_mapping_file(source_file, target_file).name}`
    """)
    
    # Verify data integrity
    if len(source_models) != expected_source_count or len(target_models) != expected_target_count:
        st.sidebar.error("⚠️ Data mismatch! Try refreshing the page.")
    
    # Filter by family
    st.sidebar.header("🔍 Filter")
    
    families = set(get_model_family(m) for m in source_models)
    selected_family = st.sidebar.selectbox(
        "Filter by family",
        ['All'] + sorted(families),
    )
    
    if selected_family != 'All':
        source_models = [m for m in source_models if get_model_family(m) == selected_family]
    
    # Show only unmatched
    show_unmatched = st.sidebar.checkbox("Show only unmatched", value=True)
    
    if show_unmatched:
        matched_sources = set(st.session_state.mappings.keys())
        source_models = [m for m in source_models if m not in matched_sources]
    
    st.sidebar.markdown(f"**Showing**: {len(source_models)} models")
    
    # Auto-match section
    st.sidebar.header("⚡ Auto-Match")
    
    # Get all unmatched source models (not just filtered ones)
    all_source_models = sources[source_file]['models']
    unmatched_models = [m for m in all_source_models if m not in st.session_state.mappings]
    
    # Find potential auto-matches
    potential_matches = auto_match_models(unmatched_models, target_models, threshold=1.0)
    
    if potential_matches:
        st.sidebar.success(f"🎯 Found {len(potential_matches)} exact matches!")
        
        if st.sidebar.button(f"✅ Accept all {len(potential_matches)} matches", type="primary", use_container_width=True):
            # Add all matches to mappings
            st.session_state.mappings.update(potential_matches)
            save_mappings(st.session_state.mappings, source_file, target_file)
            st.rerun()
        
        # Show preview of matches
        with st.sidebar.expander(f"Preview matches ({len(potential_matches)})", expanded=False):
            for src, tgt in list(potential_matches.items())[:10]:
                src_clean = clean_model_name_for_display(src)
                st.caption(f"`{src_clean[:30]}` → `{tgt[:30]}`")
            if len(potential_matches) > 10:
                st.caption(f"... and {len(potential_matches) - 10} more")
    else:
        st.sidebar.info("No exact matches found for unmatched models.")
    
    st.sidebar.markdown("---")
    
    # Show all source models in sidebar expander - ALL of them in a text area
    with st.sidebar.expander(f"📋 All Source Models ({len(source_models)})", expanded=False):
        # Selectbox for quick jump
        if source_models:
            jump_model = st.selectbox(
                "Jump to model:",
                options=["-- Select --"] + source_models,
                key="jump_source"
            )
            if jump_model != "-- Select --" and jump_model in source_models:
                idx = source_models.index(jump_model)
                if st.button("Go to this model", key="btn_jump"):
                    st.session_state.current_idx = idx
                    st.rerun()
        
        # Text area with all models (sorted)
        all_source_text = "\n".join(source_models)
        st.text_area("All source models:", value=all_source_text, height=300, 
                    key="all_source_textarea")
    
    # Model selection
    if not source_models:
        st.success("✅ All models have been matched for this file pair!")
        
        # Show summary
        total = len(st.session_state.mappings)
        matched = sum(1 for v in st.session_state.mappings.values() if v is not None)
        st.metric("Total mappings", f"{matched} matched, {total - matched} no-match")
        return
    
    # Initialize current index in session state
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    
    # Ensure index is valid for current filtered list
    if st.session_state.current_idx >= len(source_models):
        st.session_state.current_idx = 0
    
    current_idx = st.session_state.current_idx
    
    # Show index selector in sidebar
    new_idx = st.sidebar.number_input(
        "Model index",
        min_value=0,
        max_value=len(source_models) - 1,
        value=current_idx,
        key="idx_input",
    )
    if new_idx != current_idx:
        st.session_state.current_idx = new_idx
        st.rerun()
    
    current_model = source_models[current_idx]
    
    # Clean name for display
    display_name = clean_model_name_for_display(current_model)
    
    # Main area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📌 Source Model")
        st.markdown(f"**{current_idx + 1} / {len(source_models)}**")
        st.code(display_name, language=None)
        if display_name != current_model:
            st.caption(f"Full: `{current_model}`")
        
        family = get_model_family(current_model)
        st.markdown(f"**Family**: {family}")
        
        keywords = extract_keywords(current_model)
        st.markdown(f"**Keywords**: {', '.join(sorted(keywords))}")
        
        # Navigation - 3 buttons
        st.markdown("---")
        
        def go_prev():
            if st.session_state.current_idx > 0:
                st.session_state.current_idx -= 1
        
        def go_next():
            st.session_state.current_idx += 1
        
        col_prev, col_skip, col_next = st.columns(3)
        with col_prev:
            st.button("⬅️ Prev", disabled=current_idx == 0, use_container_width=True, 
                     on_click=go_prev, key="btn_prev")
        with col_skip:
            st.button("⏭️ Skip", use_container_width=True, type="secondary",
                     on_click=go_next, key="btn_skip")
        with col_next:
            st.button("➡️ Next", disabled=current_idx >= len(source_models) - 1, 
                     use_container_width=True, on_click=go_next, key="btn_next")
        
        # Quick actions
        st.markdown("---")
        
        def mark_no_match_and_next():
            st.session_state.mappings[current_model] = None
            save_mappings(st.session_state.mappings, source_file, target_file)
            st.session_state.current_idx += 1
        
        st.button("❌ No Match & Next", use_container_width=True, type="primary",
                 on_click=mark_no_match_and_next, key="btn_no_match")
    
    with col2:
        st.header("🎯 Candidate Matches")
        
        # Find candidates
        candidates = find_candidates(current_model, target_models, top_k=15)
        
        if not candidates:
            st.warning("No candidates found with similarity > 0.1")
        else:
            # Show candidates with buttons
            for i, (candidate, score) in enumerate(candidates):
                col_score, col_name, col_btn = st.columns([1, 5, 1])
                
                with col_score:
                    score_pct = int(score * 100)
                    if score >= 0.5:
                        st.success(f"{score_pct}%")
                    elif score >= 0.3:
                        st.warning(f"{score_pct}%")
                    else:
                        st.info(f"{score_pct}%")
                
                with col_name:
                    candidate_family = get_model_family(candidate)
                    candidate_display = clean_model_name_for_display(candidate)
                    st.markdown(f"`{candidate_display}`")
                    if candidate_display != candidate:
                        st.caption(f"Family: {candidate_family} | Full: {candidate[:50]}...")
                    else:
                        st.caption(f"Family: {candidate_family}")
                
                with col_btn:
                    # Use closure to capture candidate value
                    def make_match_callback(cand):
                        def callback():
                            st.session_state.mappings[current_model] = cand
                            save_mappings(st.session_state.mappings, source_file, target_file)
                            st.session_state.current_idx += 1
                        return callback
                    
                    st.button("✅", key=f"match_{i}", on_click=make_match_callback(candidate))
        
        # Manual input
        st.markdown("---")
        st.subheader("🔧 Manual Match")
        manual_match = st.text_input("Enter target model name:", key="manual_input")
        col_manual, col_search = st.columns(2)
        with col_manual:
            def set_manual_match():
                if st.session_state.manual_input:
                    st.session_state.mappings[current_model] = st.session_state.manual_input
                    save_mappings(st.session_state.mappings, source_file, target_file)
                    st.session_state.current_idx += 1
            
            st.button("🔗 Set Manual Match", use_container_width=True, 
                     on_click=set_manual_match, key="btn_manual")
        with col_search:
            if st.button("🔍 Search in targets", use_container_width=True):
                if manual_match:
                    # Search for the text in target models
                    found = [t for t in target_models if manual_match.lower() in t.lower()]
                    if found:
                        st.info(f"Found {len(found)} matches:")
                        for f in found[:10]:
                            st.code(f)
                    else:
                        st.warning("No matches found")
        
        # Show all target models in expander
        st.markdown("---")
        # Get fresh reference to target models
        current_target_models = sources[target_file]['models']
        
        with st.expander(f"📋 All Target Models from {target_file} ({len(current_target_models)} total)", expanded=False):
            # Show verification
            st.caption(f"📁 File: `{target_file}` | Models: {len(current_target_models)}")
            
            # Filter input
            filter_text = st.text_input("Filter models:", key="filter_all_models", 
                                       placeholder="Type to filter...")
            
            filtered_targets = list(current_target_models)  # Make a copy
            if filter_text:
                filtered_targets = [m for m in filtered_targets if filter_text.lower() in m.lower()]
            
            st.caption(f"Showing {len(filtered_targets)} / {len(current_target_models)} models")
            
            # Use a selectbox for choosing from all models (more efficient)
            if filtered_targets:
                selected_model = st.selectbox(
                    "Select a model:",
                    options=["-- Select --"] + filtered_targets,
                    key="select_target_model"
                )
                
                if selected_model != "-- Select --":
                    st.code(selected_model, language=None)
                    
                    def match_selected():
                        st.session_state.mappings[current_model] = selected_model
                        save_mappings(st.session_state.mappings, source_file, target_file)
                        st.session_state.current_idx += 1
                    
                    st.button("✅ Match This Model", use_container_width=True, 
                             on_click=match_selected, key="btn_match_selected")
                
                # Also show scrollable list (SORTED)
                st.markdown("---")
                st.caption(f"All {len(filtered_targets)} models (sorted alphabetically):")
                
                # Display all models in a scrollable text area for reference
                all_models_text = "\n".join(filtered_targets)
                st.text_area("Copy name from here:", value=all_models_text, 
                            height=300, key="all_models_textarea")
    
    # Statistics
    st.markdown("---")
    st.header("📊 Mapping Statistics")
    
    total_mappings = len(st.session_state.mappings)
    matched = sum(1 for v in st.session_state.mappings.values() if v is not None)
    no_match = total_mappings - matched
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Reviewed", total_mappings)
    with col_stat2:
        st.metric("Matched", matched)
    with col_stat3:
        st.metric("No Match", no_match)
    with col_stat4:
        remaining = len(sources[source_file]['models']) - total_mappings
        st.metric("Remaining", remaining)
    
    # Show recent mappings
    if st.session_state.mappings:
        with st.expander("📜 Recent Mappings", expanded=False):
            recent = list(st.session_state.mappings.items())[-10:]
            for src, tgt in reversed(recent):
                if tgt:
                    st.markdown(f"✅ `{src}` → `{tgt}`")
                else:
                    st.markdown(f"❌ `{src}` → (no match)")
    
    # Refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Export
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Export")
    
    mapping_filename = get_mapping_file(source_file, target_file).name
    json_str = json.dumps(st.session_state.mappings, indent=2, ensure_ascii=False)
    st.sidebar.download_button(
        "📥 Download Mappings",
        json_str,
        file_name=mapping_filename,
        mime="application/json",
    )
    
    # Show all existing mapping files
    with st.sidebar.expander("📂 All Mapping Files", expanded=False):
        MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
        mapping_files = list(MAPPINGS_DIR.glob("*.json"))
        if mapping_files:
            for mf in sorted(mapping_files):
                with open(mf) as f:
                    data = json.load(f)
                n_matched = sum(1 for v in data.values() if v is not None)
                st.caption(f"📄 {mf.name}: {len(data)} total, {n_matched} matched")
        else:
            st.caption("No mapping files yet")


if __name__ == "__main__":
    main()
