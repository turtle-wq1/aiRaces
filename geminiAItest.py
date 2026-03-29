import streamlit as st
import pandas as pd
import numpy as np
import time
import random

st.set_page_config(page_title="AI Race Lab: Nexus Edition", layout="wide")

# --- SESSION STATE ---
if 'lr' not in st.session_state: st.session_state.lr = 0.001
if 'cycles' not in st.session_state: st.session_state.cycles = 5000
if 'mult' not in st.session_state: st.session_state.mult = 5.0
if 'delay' not in st.session_state: st.session_state.delay = 0.0
if 'x_data' not in st.session_state: st.session_state.x_data = [1.0, 5.0, 10.0, 13.0]
if 'master_history' not in st.session_state: st.session_state.master_history = pd.DataFrame()
if 'leaderboard_data' not in st.session_state: st.session_state.leaderboard_data = pd.DataFrame()

def update_val(target, source):
    st.session_state[target] = st.session_state[source]

def sync_params(target_lr, target_cycles, target_mult, target_test=7.5, target_delay=0.0):
    st.session_state.lr = target_lr
    st.session_state.lr_t = target_lr
    st.session_state.cycles = target_cycles
    st.session_state.c_t = target_cycles
    st.session_state.mult = target_mult
    st.session_state.m_t = target_mult
    st.session_state.delay = target_delay

# --- SIDEBAR (Settings & Presets) ---
with st.sidebar:
    st.header("📋 Presets & Data")
    preset_dict = {
        "Basic": {"x": [1.0, 5.0, 10.0, 13.0], "lr": 0.001, "m": 5.0, "c": 5000, "test": 7.5, "delay": 0.0},
        "Original": {"x": [4.0, 14.0, 18.0, 6.0], "lr": 0.001, "m": 7.5, "c": 1102, "test": 10.0, "delay": 0.01},
        "Stress Test": {"x": [200.0, 500.0, 900.0], "lr": 0.00001, "m": 5.0, "c": 2000, "test": 10.0, "delay": 0.01},
        "Chaos Mix": {"x": [-10.0, 50.0, -100.0, 200.0], "lr": 0.0001, "m": 15.0, "c": 1200, "test": 10.0, "delay": 0.01}
    }
    selected_p = st.selectbox("Load Data Set:", list(preset_dict.keys()))
    if st.button("Apply Preset", use_container_width=True):
        p = preset_dict[selected_p]
        st.session_state.x_data = p["x"]
        sync_params(p["lr"], p["c"], p["m"], p.get("test", 7.5), p.get("delay", 0.0))
        st.rerun()

    st.divider()
    st.header("⚙️ Fine Tuning")
    st.number_input("LR Value", 0.0, 1.0, step=0.000001, format="%.6f", key="lr_t", on_change=update_val, args=('lr', 'lr_t'))
    st.slider("LR Slider", 0.0, 0.1, step=0.000001, format="%.6f", key="lr", on_change=update_val, args=('lr_t', 'lr'))
    st.number_input("Max Cycles", 1, 50000, key="c_t", on_change=update_val, args=('cycles', 'c_t'))
    st.slider("Cycles Slider", 1, 10000, key="cycles", on_change=update_val, args=('c_t', 'cycles'))
    st.slider("Animation Delay", 0.0, 1.0, step=0.01, key="delay")
    
    st.divider()
    if st.button("🗑️ Reset Benchmarks", use_container_width=True):
        st.session_state.master_history = pd.DataFrame()
        st.session_state.leaderboard_data = pd.DataFrame()
        st.rerun()

# --- MAIN INTERFACE ---
st.title("🏁 AI Race Lab")

edited_df = st.data_editor(pd.DataFrame({"x": st.session_state.x_data}), num_rows="dynamic", use_container_width=True, key="editor")
x_input = edited_df["x"].tolist()
y_input = [val * st.session_state.mult for val in x_input]

st.subheader("🏎️ Race Setup")
c1, c2 = st.columns(2)
with c1:
    mode = st.radio("Mode:", ["SINGLE", "RACE"], horizontal=True)
    fast_mode = st.toggle("⚡ Fast Mode", value=True)
with c2:
    test_val = st.number_input("Test X Value:", value=7.5 if selected_p == "Basic" else 10.0)
    stop_training = st.toggle("🚨 EMERGENCY STOP", value=False)

target_truth_p = test_val * st.session_state.mult

if mode == "SINGLE":
    active_model = st.selectbox("Select Model:", ["BASIC", "SMART", "GENIUS", "DARWIN", "QUANTUM", "SENTINEL", "NEXUS"])
    racers = [active_model]
else:
    racers = st.multiselect("Select Competitors:", ["BASIC", "SMART", "GENIUS", "DARWIN", "QUANTUM", "SENTINEL", "NEXUS"], default=["BASIC", "SMART", "GENIUS"])

col_mon, col_graph = st.columns([1, 1])
with col_mon:
    telemetry_area = st.empty()
with col_graph:
    iteration_text = st.empty()
    chart_placeholder = st.empty()

def run_training(m, x_vals, y_vals, target_w, test_v):
    w, lr, max_c, delay = 0.0, st.session_state.lr, int(st.session_state.cycles), st.session_state.delay
    history, status, start_time = [], "Success", time.perf_counter()
    q_u, vel, mom = 10.0, 0.0, 0.9

    for epoch in range(max_c):
        if stop_training: status = "🛑 Stopped"; break
        
        if m == "BASIC":
            for i in range(len(x_vals)): w -= lr * (w * x_vals[i] - y_vals[i]) * x_vals[i]
        elif m == "DARWIN":
            best_w, min_err = w, sum([abs(w*x - y) for x, y in zip(x_vals, y_vals)])
            for _ in range(10):
                mutant = w + random.uniform(-lr*20, lr*20)
                err = sum([abs(mutant*x - y) for x, y in zip(x_vals, y_vals)])
                if err < min_err: min_err, best_w = err, mutant
            w = best_w
        elif m == "QUANTUM":
            guess = w + random.uniform(-q_u, q_u)
            if sum([abs(guess*x-y) for x,y in zip(x_vals, y_vals)]) < sum([abs(w*x-y) for x,y in zip(x_vals, y_vals)]):
                w, q_u = guess, q_u * 0.95
            else: q_u *= 1.02
        elif m == "SENTINEL":
            grad = sum([(w*x-y)*x for x,y in zip(x_vals, y_vals)])/len(x_vals)
            vel = (mom*vel) - (lr*grad); w += vel
        elif m == "NEXUS":
            sa, sb = w+lr, w-lr
            ea = sum([abs(sa*x-y) for x,y in zip(x_vals, y_vals)])
            eb = sum([abs(sb*x-y) for x,y in zip(x_vals, y_vals)])
            w = sa if ea < eb else sb
        else: # SMART/GENIUS
            grad = sum([(w*x-y)*x for x,y in zip(x_vals, y_vals)])/len(x_vals)
            w -= (lr*grad)
            if m == "GENIUS" and abs(w - st.session_state.mult) < 1e-10:
                w, status = st.session_state.mult, "✨ Smart Finish"; break

        history.append(w)
        if np.isnan(w) or np.isinf(w): status = "💥 Exploded"; break

        # ANIMATION LOGIC
        if not fast_mode:
            if delay > 0 or epoch % 100 == 0:
                curr_p = w * test_v
                tele_df = pd.DataFrame({"Stat": ["Weight", "Prediction", "Error Gap"], "Value": [f"{w:.8f}", f"{curr_p:.8f}", f"{abs(curr_p - target_truth_p):.12f}"], "Target": [f"{target_w:.8f}", f"{target_truth_p:.8f}", "0.00000000"]})
                telemetry_area.table(tele_df)
                iteration_text.write(f"**Racing:** {m} | Cycle: {epoch+1}")
                if delay > 0: time.sleep(delay)
            
    dur = (time.perf_counter() - start_time) * 1000
    gap = abs((w * test_val) - target_truth_p)
    if status not in ["🛑 Stopped", "💥 Exploded", "✨ Smart Finish"]:
        status = "✅ Success" if gap < 0.00001 else ("⚠️ Inaccurate" if gap < 1.0 else "❌ Failure")
    
    return w, history, status, dur, gap

if st.button("🏁 START SIMULATION", type="primary", use_container_width=True):
    all_results = []
    st.session_state.master_history = pd.DataFrame()
    
    for m in racers:
        st.toast(f"Starting {m}...")
        if mode == "RACE": time.sleep(2) 
        
        final_w, hist, status, dur, gap = run_training(m, x_input, y_input, st.session_state.mult, test_val)
        st.session_state.master_history[m] = pd.Series(hist)
        all_results.append({"MODEL": m, "STATUS": status, "GAP": gap, "SPEED": dur})

    lb_df = pd.DataFrame(all_results).sort_values(by=["GAP", "SPEED"])
    lb_df.insert(0, "RANK", [f"#{i+1}" for i in range(len(lb_df))])
    st.session_state.leaderboard_data = lb_df
    st.rerun()

st.divider()
if not st.session_state.leaderboard_data.empty:
    st.subheader("🏆 Leaderboard")
    st.dataframe(st.session_state.leaderboard_data, use_container_width=True, hide_index=True)

if not st.session_state.master_history.empty:
    st.subheader("📈 Performance Chart")
    visible = [c for c in st.session_state.master_history.columns if st.checkbox(c, value=True, key=f"v_{c}")]
    if visible: st.line_chart(st.session_state.master_history[visible])

# to start run python -m streamlit run c:/Users/lucac/geminiAItest.py