# Filename: parse_creep.py (Revision 14 - Style & Comments Refinement)
# Purpose: Analyses StarCraft II replay files (.SC2Replay) to extract
#          creep tumor statistics for Zerg players and generate
#          summary plots using Matplotlib.
import sc2reader
from collections import defaultdict
import argparse
import math
import sys
import traceback
import os
import io
import numpy as np

# --- Plotting Setup ---
try:
    import matplotlib
    # Use 'Agg' backend for scripts/servers where GUI is not available.
    # This needs to be set *before* importing pyplot.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    # Use newer colormap access method (Matplotlib 3.5+) with fallback
    try:
        from matplotlib.cm import get_cmap as mpl_get_cmap
        from matplotlib.colormaps import get_cmap
        cmap_obj = mpl_get_cmap('tab10') # Keep ref for cmap properties if needed
        COLORMAP_FUNC = get_cmap('tab10') # Preferred way to get colors
    except ImportError: # Fallback for older Matplotlib
        from matplotlib import cm
        cmap_obj = cm.get_cmap('tab10')
        COLORMAP_FUNC = cm.get_cmap('tab10')
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib plotting enabled.")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not found. Plotting features disabled.")

# --- Game Constants ---
QUEEN_BUILD_TUMOR_ABILITY_NAME = "CreepTumor"       # Ability name used by Queen
TUMOR_BUILD_TUMOR_ABILITY_NAME = "BuildCreepTumor"  # Ability name used by Tumor
CREEP_TUMOR_UNIT_NAME = "CreepTumorBurrowed" # Unit name for active tumors
TUMOR_COUNT_MILESTONES = [5, 10, 15, 20]     # Track time to reach these active counts

# --- Helper Functions ---

def format_time(total_seconds, none_val="--:--"):
    """Converts total seconds into a MM:SS formatted string.

    Args:
        total_seconds (float | int | None): The number of seconds.
        none_val (str): The string to return if input is None or negative.

    Returns:
        str: The time formatted as MM:SS or the none_val string.
    """
    if total_seconds is None or total_seconds < 0:
        return none_val
    minutes = math.floor(total_seconds / 60)
    seconds = math.floor(total_seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_player_info(replay_file):
    """Extracts basic player information (PID, Name, Race) from a replay file.

    Loads minimal replay data for efficiency.

    Args:
        replay_file (str): Path to the .SC2Replay file.

    Returns:
        list[dict]: A list of dictionaries, one per player, containing
                    'pid', 'name', and 'race'. Returns empty list on error.
    """
    try:
        # Load level 0 is sufficient for player details
        replay = sc2reader.load_replay(replay_file, load_level=0)
        return [{"pid": p.pid, "name": p.name, "race": p.play_race}
                for p in replay.players]
    except Exception as e:
        print(f"ERROR: Could not load player info from '{replay_file}': {e}")
        return []

def is_zvz(player_info_list):
    """Checks if the game is ZvZ based on a list of player info dicts.

    Args:
        player_info_list (list[dict]): List from get_player_info().

    Returns:
        bool: True if exactly two players and both are Zerg, False otherwise.
    """
    return len(player_info_list) == 2 and all(p.get('race') == 'Zerg' for p in player_info_list)

def _build_cumulative_timeline(time_list):
    """Creates a timeline dictionary showing cumulative counts from event times.

    Handles multiple events occurring at the exact same timestamp.

    Args:
        time_list (list[float | int]): A list of timestamps when an event occurred.

    Returns:
        defaultdict[int]: A dictionary mapping time (seconds) to the
                          cumulative count up to that time. Includes time 0.
    """
    timeline = defaultdict(int)
    timeline[0] = 0 # Always start at 0 count at time 0
    if not time_list:
        return timeline

    sorted_times = sorted(time_list)
    count = 0
    last_time = 0
    for t in sorted_times:
        # Ensure we only process times moving forward.
        # If time jumps, carry forward the previous count until just before the event.
        if t > last_time + 1e-6: # Use epsilon for float comparison
             timeline[t - 1e-9] = count # Record count just before increment

        # Increment count for this event time
        count += 1
        timeline[t] = count
        last_time = t
    return timeline

# --- Main Analysis Function ---

def analyze_creep(replay_file, selected_pid=None):
    """
    Analyses a StarCraft II replay file for Zerg creep tumor statistics.

    Loads the replay, identifies Zerg players, iterates through game events
    to calculate various metrics like tumor counts, initiation timings,
    spread intervals, and milestone achievements.

    Args:
        replay_file (str): Path to the .SC2Replay file.
        selected_pid (int | None): If provided, analyse only this player ID.
                                   If None, analyse all Zerg players found.

    Returns:
        tuple[dict | None, int]: A tuple containing:
            - dict: A dictionary where keys are Zerg player PIDs and values
                    are dictionaries of calculated statistics for that player.
                    Returns None if analysis fails or no Zerg players found.
            - int: The game duration in seconds. Returns 0 on failure.
    """
    replay = None
    try:
        print(f"\nLoading replay: {os.path.basename(replay_file)}...")
        # Load level 4 for full event data, load map for dimensions if possible
        replay = sc2reader.load_replay(replay_file, load_map=True, load_level=4)
        print(" -> Load successful.")
    except Exception as e:
        print(f"ERROR: Failed loading replay '{replay_file}': {e}")
        return None, 0

    # Get game duration
    game_duration_seconds = getattr(getattr(replay, 'length', None), 'seconds', 0)
    if game_duration_seconds <= 0:
        print(f"ERROR: Replay '{replay_file}' has invalid duration ({game_duration_seconds}s).")
        return None, 0
    print(f" -> Duration: {format_time(game_duration_seconds)} ({game_duration_seconds}s)")

    # Get map details safely
    map_width, map_height = 200, 200 # Sensible defaults
    map_name = getattr(replay, 'map_name', "Unknown Map")
    if hasattr(replay, 'map_info') and replay.map_info:
        # map_name already fetched via getattr fallback
        if hasattr(replay.map_info, 'map_size') and replay.map_info.map_size:
            map_width = replay.map_info.map_size.x
            map_height = replay.map_info.map_size.y
        # else: print(" -> Warning: map_info present but map_size missing.")
    # else: print(" -> Warning: map_info attribute not found.")

    # Identify Zerg player(s) to analyse
    creep_stats, zerg_players = {}, {}
    try:
        for p in replay.players:
             is_target_player = (selected_pid is None or p.pid == selected_pid)
             is_zerg = getattr(p, 'play_race', None) == "Zerg"
             if is_target_player and is_zerg and p.pid is not None:
                 zerg_players[p.pid] = p
                 action = "Analysing selected" if selected_pid else "Found"
                 print(f"  -> {action} Zerg: {p.name} (PID: {p.pid})")
    except Exception as e:
        print(f"ERROR: accessing player info: {e}")
        return None, game_duration_seconds

    if not zerg_players:
        reason = f"Selected player PID {selected_pid} not found or not Zerg." if selected_pid else "No Zerg players found."
        print(f" -> {reason}")
        return None, game_duration_seconds

    # Determine PID for the CommandEvent workaround (only if analysing all players and only one Zerg exists)
    the_single_zerg_pid = list(zerg_players.keys())[0] if len(zerg_players) == 1 and selected_pid is None else None

    # Initialise statistics dictionary for each Zerg player
    for pid, player_obj in zerg_players.items():
        creep_stats[pid] = {
            "player_name": player_obj.name,
            "replay_file": os.path.basename(replay_file),
            "game_duration": game_duration_seconds,
            "map_name": map_name,
            "map_width": map_width,
            "map_height": map_height,
            "tumors_built_by_queen": 0,
            "tumors_built_by_tumor": 0,
            # "tumors_died": 0, # Removed as requested
            "active_tumors_timeline": defaultdict(int),
            "last_tumor_time": 0,
            "first_queen_tumor_time": None,
            "time_to_x_tumors": {c: None for c in TUMOR_COUNT_MILESTONES},
            # Internal lists for building timelines
            "_queen_tumor_times": [],
            "_tumor_tumor_times": [],
            # "_died_tumor_times": [], # Removed
            "tumor_spread_intervals": [], # Stores (completion_time, interval_secs)
            "cumulative_avg_interval_timeline": {}, # Stores (completion_time, avg_interval)
            # Pre-calculated summaries for convenience
            "peak_active_tumors": 0,
            "peak_active_tumors_time": 0,
            "final_active_tumors": 0,
            "avg_tumor_spread_interval": None,
        }
        creep_stats[pid]["active_tumors_timeline"][0] = 0 # Ensure timeline starts at zero

    # Track state during event processing
    current_tumor_count = defaultdict(int) # pid -> current active count
    last_event_time = defaultdict(int)     # pid -> last time timeline was updated
    milestones_reached = {pid: {c: False for c in TUMOR_COUNT_MILESTONES} for pid in zerg_players}
    last_tumor_spread_time = {}            # pid -> time of last tumor spread command
    interval_sum = defaultdict(float)      # pid -> sum of intervals for avg calc
    interval_count = defaultdict(int)      # pid -> number of intervals for avg calc

    print(f" -> Processing {len(replay.events)} events...")
    try: # --- Main Event Processing Loop ---
        for event in replay.events:
            # Time bounds check
            current_time_seconds = event.second
            if current_time_seconds < 0 or current_time_seconds > game_duration_seconds:
                continue

            # Determine the PID associated with unit-based events (Done, Died)
            tracker_event_pid = None
            if hasattr(event, 'unit') and event.unit and hasattr(event.unit, 'owner') and event.unit.owner:
                 unit_owner_pid = getattr(event.unit.owner, 'pid', None)
                 if unit_owner_pid in zerg_players: # Is the unit owner one we're tracking?
                     tracker_event_pid = unit_owner_pid

            # Update timeline before processing the event's effects
            # Ensures the timeline reflects the state *before* the change at this timestamp
            if tracker_event_pid is not None and current_time_seconds > last_event_time[tracker_event_pid]:
                creep_stats[tracker_event_pid]["active_tumors_timeline"][current_time_seconds] = current_tumor_count[tracker_event_pid]
                last_event_time[tracker_event_pid] = current_time_seconds

            # --- Handle Specific Event Types ---

            # CommandEvent: Track tumor initiation (Queen or Tumor)
            if isinstance(event, sc2reader.events.game.CommandEvent):
                 cmd_player_id = getattr(event, 'player_id', None)
                 # Determine which player PID to attribute this command to, applying workaround if needed
                 target_pid_for_cmd = the_single_zerg_pid # Assume single zerg if analysing all
                 if selected_pid is not None and cmd_player_id == selected_pid:
                     target_pid_for_cmd = selected_pid # Use selected if matching
                 elif selected_pid is None and len(zerg_players) > 1:
                      target_pid_for_cmd = None # Cannot determine if multiple Zerg and none selected

                 if target_pid_for_cmd is not None: # Only proceed if attributable
                     ability_name = getattr(event, 'ability_name', None)
                     if ability_name == QUEEN_BUILD_TUMOR_ABILITY_NAME:
                         creep_stats[target_pid_for_cmd]["tumors_built_by_queen"] += 1
                         creep_stats[target_pid_for_cmd]["_queen_tumor_times"].append(current_time_seconds)
                         if creep_stats[target_pid_for_cmd]["first_queen_tumor_time"] is None:
                             creep_stats[target_pid_for_cmd]["first_queen_tumor_time"] = current_time_seconds
                     elif ability_name == TUMOR_BUILD_TUMOR_ABILITY_NAME:
                         creep_stats[target_pid_for_cmd]["tumors_built_by_tumor"] += 1
                         creep_stats[target_pid_for_cmd]["_tumor_tumor_times"].append(current_time_seconds)
                         # Calculate spread interval if previous spread time recorded
                         if target_pid_for_cmd in last_tumor_spread_time:
                             interval = current_time_seconds - last_tumor_spread_time[target_pid_for_cmd]
                             if interval > 0: # Basic sanity check
                                creep_stats[target_pid_for_cmd]["tumor_spread_intervals"].append((current_time_seconds, interval))
                                # Update cumulative average tracking
                                interval_sum[target_pid_for_cmd] += interval
                                interval_count[target_pid_for_cmd] += 1
                                avg = interval_sum[target_pid_for_cmd] / interval_count[target_pid_for_cmd]
                                creep_stats[target_pid_for_cmd]["cumulative_avg_interval_timeline"][current_time_seconds] = avg
                         last_tumor_spread_time[target_pid_for_cmd] = current_time_seconds # Record time for next interval calc

            # UnitDoneEvent: Track completed tumors, update active count & milestones
            elif isinstance(event, sc2reader.events.tracker.UnitDoneEvent):
                 if tracker_event_pid is not None and event.unit and event.unit.name == CREEP_TUMOR_UNIT_NAME:
                     current_tumor_count[tracker_event_pid] += 1
                     new_count = current_tumor_count[tracker_event_pid]
                     # Update timeline with *new* count at this exact time
                     creep_stats[tracker_event_pid]["active_tumors_timeline"][current_time_seconds] = new_count
                     creep_stats[tracker_event_pid]["last_tumor_time"] = current_time_seconds
                     last_event_time[tracker_event_pid] = current_time_seconds # Mark this time as updated
                     # Check milestones
                     for c in TUMOR_COUNT_MILESTONES:
                          if not milestones_reached[tracker_event_pid][c] and new_count >= c:
                              creep_stats[tracker_event_pid]["time_to_x_tumors"][c] = current_time_seconds
                              milestones_reached[tracker_event_pid][c] = True

            # UnitDiedEvent: Only update active count timeline
            elif isinstance(event, sc2reader.events.tracker.UnitDiedEvent):
                 dying_unit_owner_pid = getattr(getattr(event.unit, 'owner', None), 'pid', None)
                 if dying_unit_owner_pid in zerg_players:
                     # Check name to be reasonably sure it's a tumor affecting the count
                     if event.unit and event.unit.name == CREEP_TUMOR_UNIT_NAME:
                         current_tumor_count[dying_unit_owner_pid] -= 1
                         if current_tumor_count[dying_unit_owner_pid] < 0:
                             current_tumor_count[dying_unit_owner_pid] = 0 # Prevent negative counts
                         # Update timeline with *new* count
                         creep_stats[dying_unit_owner_pid]["active_tumors_timeline"][current_time_seconds] = current_tumor_count[dying_unit_owner_pid]
                         last_event_time[dying_unit_owner_pid] = current_time_seconds # Mark time as updated
                         # Do NOT increment tumors_died or add to _died_tumor_times

    except Exception as e:
        print(f"ERROR: during event processing loop: {e}")
        traceback.print_exc()
        # Return whatever stats were collected before the error
        return creep_stats, game_duration_seconds

    # --- Post-processing ---
    print(" -> Finalising statistics...")
    final_time = game_duration_seconds
    for pid in zerg_players.keys():
         if pid in creep_stats:
            # Pad timelines to game end time
            final_active_count = current_tumor_count.get(pid, 0)
            timeline_keys_to_pad = ["active_tumors_timeline", "queen_tumor_init_timeline",
                                    "tumor_tumor_init_timeline", "cumulative_avg_interval_timeline"] # Removed died
            for tl_key in timeline_keys_to_pad:
                if tl_key not in creep_stats[pid]: creep_stats[pid][tl_key] = {} # Ensure dict exists
                # Determine correct final value for padding
                if tl_key == "active_tumors_timeline": final_val = final_active_count
                elif tl_key == "queen_tumor_init_timeline": final_val = creep_stats[pid]["tumors_built_by_queen"]
                elif tl_key == "tumor_tumor_init_timeline": final_val = creep_stats[pid]["tumors_built_by_tumor"]
                elif tl_key == "cumulative_avg_interval_timeline":
                    relevant_avg_times = sorted([t for t in creep_stats[pid][tl_key].keys() if t <= final_time])
                    final_val = creep_stats[pid][tl_key][relevant_avg_times[-1]] if relevant_avg_times else 0
                else: final_val = 0
                # Get latest time entry <= final_time
                relevant_times = [t for t in creep_stats[pid][tl_key].keys() if t <= final_time]
                last_recorded_time = max(relevant_times) if relevant_times else 0
                # Pad end if needed or update value at end time
                if last_recorded_time < final_time:
                     creep_stats[pid][tl_key][final_time] = creep_stats[pid][tl_key].get(last_recorded_time, final_val) # Carry forward last known value
                elif last_recorded_time == final_time:
                     creep_stats[pid][tl_key][final_time] = final_val # Ensure final value is correct if event was exactly at end

            # Build cumulative timelines from stored event times
            creep_stats[pid]["queen_tumor_init_timeline"] = _build_cumulative_timeline(creep_stats[pid]["_queen_tumor_times"])
            creep_stats[pid]["tumor_tumor_init_timeline"] = _build_cumulative_timeline(creep_stats[pid]["_tumor_tumor_times"])
            # Removed died_tumor_timeline build

            # Calculate final overall average interval
            intervals = [interval for _, interval in creep_stats[pid]["tumor_spread_intervals"]]
            creep_stats[pid]["avg_tumor_spread_interval"] = sum(intervals) / len(intervals) if intervals else None

            # Calculate final peak/active stats (safer to do this *after* padding active_tumors_timeline)
            max_act_tumors, max_act_time, final_act_tumors = 0, 0, 0
            valid_active = {t: c for t, c in creep_stats[pid]["active_tumors_timeline"].items() if t >= 0 and t <= final_time}
            if valid_active:
                # Find max value; if multiple times have the same max, max() takes the first one
                max_act_entry = max(valid_active.items(), key=lambda item: item[1])
                max_act_time = max_act_entry[0]
                max_act_tumors = max_act_entry[1]
                # Get value at final time precisely
                final_act_tumors = valid_active.get(final_time, list(valid_active.values())[-1] if valid_active else 0)
            creep_stats[pid]["peak_active_tumors"] = max_act_tumors
            creep_stats[pid]["peak_active_tumors_time"] = max_act_time
            creep_stats[pid]["final_active_tumors"] = final_act_tumors

            # Optional: Clean up internal lists
            # del creep_stats[pid]["_queen_tumor_times"]
            # del creep_stats[pid]["_tumor_tumor_times"]

    print(" -> Analysis complete.")
    return creep_stats, game_duration_seconds


# --- Plotting Functions (Matplotlib) ---

def _apply_matplotlib_dark_style(fig, ax_list):
    """Applies a dark theme styling to a Matplotlib figure and its axes."""
    fig.patch.set_facecolor('#222222')
    axes_list = ax_list.flatten() if isinstance(ax_list, (list, tuple, np.ndarray)) and hasattr(ax_list, 'flatten') else [ax_list]
    for ax in axes_list:
        if ax is None: continue # Handle empty subplots
        ax.set_facecolor('#282828') # Slightly lighter background for plot area
        ax.tick_params(axis='x', colors='lightgray', labelsize=8)
        ax.tick_params(axis='y', colors='lightgray', labelsize=8)
        ax.xaxis.label.set_color('lightgray'); ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_color('lightgray'); ax.yaxis.label.set_size(10)
        ax.title.set_color('white'); ax.title.set_size(12)
        for spine in ax.spines.values(): spine.set_color('gray')
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='dimgray') # Darker grid

    # Style legend if it exists on the figure or axes
    legend = None
    if fig.legends: legend = fig.legends[0]
    else: # Check individual axes
        for ax in axes_list:
             current_legend = ax.get_legend()
             if current_legend: legend = current_legend; break
    if legend:
         legend.get_frame().set_facecolor('#333333'); legend.get_frame().set_edgecolor('darkgray')
         for text in legend.get_texts(): text.set_color('white')

def _plot_timeline_data(ax, timeline_data, max_duration, label, color):
    """Helper to plot one timeline series using Matplotlib step plot."""
    if not timeline_data and max_duration is not None: # Handle empty timeline
         ax.step([0, max_duration], [0, 0], where='post', label=label, color=color, linewidth=1.5)
         return
    elif not timeline_data:
         return # Cannot plot if max_duration also unknown

    valid_entries = {t: c for t, c in timeline_data.items() if t >= 0 and t <= max_duration}
    if 0 not in valid_entries: valid_entries[0] = 0 # Ensure start at time 0
    # Ensure data point exists at max_duration
    times_before = sorted([t for t in valid_entries.keys() if t < max_duration])
    last_count = valid_entries[times_before[-1]] if times_before else valid_entries.get(0,0)
    valid_entries[max_duration] = last_count # Pad end time

    sorted_times = sorted(valid_entries.keys())
    times = [t for t in sorted_times]
    counts = [valid_entries[t] for t in times]
    # Plot using step plot
    ax.step(times, counts, where='post', label=label, color=color, linewidth=1.5)

def matplotlib_creep_timeline(results_list):
    """Generates comparison plot for ACTIVE tumors using Matplotlib."""
    if not MATPLOTLIB_AVAILABLE: return None
    valid_results = [(stats, dur) for stats, dur in results_list if stats and dur > 0]
    if not valid_results: return None
    print(f"Generating Matplotlib Active Tumor timeline graph...")
    max_duration = max(dur for _, dur in valid_results)
    try:
        fig, ax = plt.subplots(figsize=(10, 5)); # Use single axes
        num_colors = cmap_obj.N if hasattr(cmap_obj, 'N') else 10

        for i, (stats_dict, duration) in enumerate(valid_results):
            if not stats_dict or len(stats_dict) != 1: continue # Skip invalid/multiplayer
            pid=list(stats_dict.keys())[0]; player_stats=stats_dict[pid]; player_name = player_stats.get('player_name', f"P{pid}")
            replay_file_short = player_stats.get('replay_file', f"R{i+1}")[:20]; label = f"{replay_file_short} ({player_name})"
            color = COLORMAP_FUNC(i % num_colors) # Cycle through colormap
            _plot_timeline_data(ax, player_stats.get("active_tumors_timeline", {}), max_duration, label, color)

        # Formatting
        ax.set_title("Active Creep Tumor Count Comparison"); ax.set_xlabel("Game Time (MM:SS)"); ax.set_ylabel("Active Creep Tumors")
        ax.legend(fontsize='small'); ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_time(x)))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(max(60, int(max_duration/10)//60*60))) # Ticks approx every 10% or 60s
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right'); ax.set_ylim(bottom=0); ax.set_xlim(left=0, right=max_duration)
        _apply_matplotlib_dark_style(fig, ax); fig.tight_layout() # Apply style before saving

        # Save to buffer
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); buf.seek(0); plt.close(fig);
        return buf.getvalue()
    except Exception as e: print(f"Error generating matplotlib timeline: {e}"); plt.close('all'); return None

def matplotlib_initiated_tumors(results_list):
    """Generates comparison plot for INITIATED tumors using Matplotlib."""
    if not MATPLOTLIB_AVAILABLE: return None
    valid_results = [(stats, dur) for stats, dur in results_list if stats and dur > 0];
    if not valid_results: return None; print(f"Generating Matplotlib Initiated Tumor timeline graph...")
    max_duration = max(dur for _, dur in valid_results)
    try:
        num_replays = len(valid_results); cols = min(num_replays, 2); rows = math.ceil(num_replays / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False); fig.suptitle("Cumulative Tumors Initiated (Queen vs Tumor)")

        for i, (stats_dict, duration) in enumerate(valid_results):
            ax = axes[i // cols, i % cols] # Get correct subplot
            if not stats_dict or len(stats_dict) != 1: ax.set_title(f"Replay {i+1}: Error"); ax.axis('off'); continue
            pid=list(stats_dict.keys())[0]; player_stats=stats_dict[pid]; player_name = player_stats.get('player_name', f"P{pid}"); replay_file_short = player_stats.get('replay_file', f"R{i+1}")[:15]
            ax.set_title(f"{replay_file_short} ({player_name})", pad=5)
            # Use the generated cumulative timelines
            _plot_timeline_data(ax, player_stats.get("queen_tumor_init_timeline", {}), max_duration, "By Queen", 'aqua')
            _plot_timeline_data(ax, player_stats.get("tumor_tumor_init_timeline", {}), max_duration, "By Tumor", 'salmon')
            ax.legend(fontsize='x-small'); ax.set_xlabel("Game Time (MM:SS)"); ax.set_ylabel("Cumulative Count")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_time(x))); ax.xaxis.set_major_locator(ticker.MultipleLocator(max(60, int(max_duration/5)//60*60)))
            plt.setp(ax.get_xticklabels(), rotation=20, ha='right'); ax.set_ylim(bottom=0); ax.set_xlim(left=0, right=max_duration);

        _apply_matplotlib_dark_style(fig, axes) # Apply style to all axes
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); buf.seek(0); plt.close(fig); return buf.getvalue()
    except Exception as e: print(f"Error generating matplotlib initiated plot: {e}"); plt.close('all'); return None

# --- REMOVED matplotlib_died_tumors ---

def matplotlib_cumulative_avg_interval(results_list): # Renamed function
    """Plots cumulative average tumor spread intervals over time using Matplotlib."""
    if not MATPLOTLIB_AVAILABLE: return None
    valid_results = [(stats, dur) for stats, dur in results_list if stats and dur > 0]
    if not valid_results: return None
    print(f"Generating Matplotlib Cumulative Avg Spread Interval graph...")
    max_duration = max(dur for _, dur in valid_results)
    try:
        num_replays = len(valid_results); cols = min(num_replays, 2); rows = math.ceil(num_replays / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False); fig.suptitle("Cumulative Average Tumor Spread Interval")
        max_avg_interval_overall = 0

        for i, (stats_dict, duration) in enumerate(valid_results):
            ax = axes[i // cols, i % cols]
            if not stats_dict or len(stats_dict) != 1: ax.set_title(f"Replay {i+1}: Error"); ax.axis('off'); continue
            pid=list(stats_dict.keys())[0]; player_stats = stats_dict[pid]; player_name = player_stats.get('player_name', f"P{pid}"); replay_file_short = player_stats.get('replay_file', f"R{i+1}")[:15]
            ax.set_title(f"{replay_file_short} ({player_name})", pad=5)
            # Use the generated cumulative average timeline
            avg_interval_timeline = player_stats.get("cumulative_avg_interval_timeline", {})
            if avg_interval_timeline:
                sorted_times = sorted(avg_interval_timeline.keys())
                times = [t for t in sorted_times if t <= max_duration]
                avg_intervals = [avg_interval_timeline[t] for t in times]
                if times:
                    ax.plot(times, avg_intervals, marker='.', linestyle='-', markersize=4, alpha=0.8)
                    current_max = max(avg_intervals) if avg_intervals else 0; max_avg_interval_overall = max(max_avg_interval_overall, current_max)
                else: ax.text(0.5, 0.5, 'No Spread Data', ha='center', va='center', transform=ax.transAxes)
            else: ax.text(0.5, 0.5, 'No Spread Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel("Game Time (MM:SS)"); ax.set_ylabel("Cumulative Avg Interval (s)"); ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: format_time(x)))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(max(60, int(max_duration/5)//60*60))); plt.setp(ax.get_xticklabels(), rotation=20, ha='right'); ax.set_xlim(left=0, right=max_duration)

        for r in range(rows): # Apply consistent Y limits and style after plotting all
            for c in range(cols):
                ax_current = axes[r,c]
                if r * cols + c < num_replays: # Check if subplot corresponds to a replay
                   ax_current.set_ylim(bottom=0, top=max(30, max_avg_interval_overall * 1.1))
                   _apply_matplotlib_dark_style(fig, ax_current) # Apply style here

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); buf.seek(0); plt.close(fig); return buf.getvalue()
    except Exception as e: print(f"Error generating matplotlib interval plot: {e}"); plt.close('all'); return None

# --- NEW Bar chart plotting functions ---

def matplotlib_summary_bars(results_list):
    """Generates bar chart comparing summary stats using Matplotlib."""
    if not MATPLOTLIB_AVAILABLE: return None
    valid_results = [(stats, dur) for stats, dur in results_list if stats and dur > 0]
    if not valid_results: return None
    print(f"Generating Matplotlib summary bar chart...")
    num_replays = len(valid_results)

    labels = []; peak_tumors = []; final_tumors = []; total_initiated = []; avg_interval = []
    # Removed total_died list

    for i, (stats_dict, duration) in enumerate(valid_results):
        if not stats_dict or len(stats_dict) != 1: continue
        pid=list(stats_dict.keys())[0]; player_stats = stats_dict[pid]
        replay_file_short = player_stats.get('replay_file', f"R{i+1}")[:15]; player_name = player_stats.get('player_name', f"P{pid}")
        labels.append(f"{replay_file_short}\n({player_name})")
        peak_tumors.append(player_stats.get("peak_active_tumors", 0))
        final_tumors.append(player_stats.get("final_active_tumors", 0))
        total_initiated.append(player_stats.get("tumors_built_by_queen", 0) + player_stats.get("tumors_built_by_tumor", 0))
        # Removed total_died append
        avg_interval.append(player_stats.get("avg_tumor_spread_interval", 0) or 0)

    if not labels: return None
    x = np.arange(len(labels)); num_metrics = 4 # Adjusted number of metrics
    width = 0.8 / num_metrics # Adjusted width calculation

    try:
        fig, ax = plt.subplots(figsize=(max(8, (num_metrics+1)*0.6 * num_replays), 6)) # Adjust figure size dynamically
        # Adjusted bar positions
        rects1 = ax.bar(x - 1.5*width, peak_tumors, width, label='Peak Active')
        rects2 = ax.bar(x - 0.5*width, final_tumors, width, label='Final Active')
        rects3 = ax.bar(x + 0.5*width, total_initiated, width, label='Total Initiated')
        rects4 = ax.bar(x + 1.5*width, avg_interval, width, label='Avg Interval (s)')
        # Removed rects for total_died

        ax.set_ylabel('Count / Seconds'); ax.set_title('Creep Summary Stat Comparison')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.legend(fontsize='small')
        ax.bar_label(rects1, padding=3, fmt='%.0f', fontsize=8); ax.bar_label(rects2, padding=3, fmt='%.0f', fontsize=8)
        ax.bar_label(rects3, padding=3, fmt='%.0f', fontsize=8); ax.bar_label(rects4, padding=3, fmt='%.1f', fontsize=8)
        # Removed bar label for total_died

        ax.set_ylim(bottom=0); _apply_matplotlib_dark_style(fig, ax); fig.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); buf.seek(0); plt.close(fig); return buf.getvalue()
    except Exception as e: print(f"Error generating summary bars: {e}"); plt.close('all'); return None

def matplotlib_timing_bars(results_list):
    """Generates bar chart comparing timing stats using Matplotlib."""
    if not MATPLOTLIB_AVAILABLE: return None
    valid_results = [(stats, dur) for stats, dur in results_list if stats and dur > 0]
    if not valid_results: return None
    print(f"Generating Matplotlib timing bar chart...")
    num_replays = len(valid_results)

    labels = []; first_tumor_times = []; time_to_milestones = defaultdict(list)
    for i, (stats_dict, duration) in enumerate(valid_results):
        if not stats_dict or len(stats_dict) != 1: continue
        pid=list(stats_dict.keys())[0]; player_stats = stats_dict[pid]
        replay_file_short = player_stats.get('replay_file', f"R{i+1}")[:15]; player_name = player_stats.get('player_name', f"P{pid}")
        labels.append(f"{replay_file_short}\n({player_name})")
        first_tumor_times.append(player_stats.get("first_queen_tumor_time", 0) or 0)
        for ms in TUMOR_COUNT_MILESTONES: time_to_milestones[ms].append(player_stats.get("time_to_x_tumors", {}).get(ms, 0) or 0)

    if not labels: return None
    x = np.arange(len(labels)); num_metrics = 1 + len(TUMOR_COUNT_MILESTONES); width = 0.8 / num_metrics
    try:
        fig, ax = plt.subplots(figsize=(max(8, (num_metrics+1)*0.6 * num_replays), 6))
        offset = - (num_metrics / 2.0 - 0.5) * width
        rects_list = [] # Store rects for labeling
        rects = ax.bar(x + offset, first_tumor_times, width, label='First Queen Tumor'); rects_list.append(rects)
        offset += width
        for ms in TUMOR_COUNT_MILESTONES:
            rects = ax.bar(x + offset, time_to_milestones[ms], width, label=f'Time to {ms}'); rects_list.append(rects)
            offset += width

        ax.set_ylabel('Game Time (Seconds)'); ax.set_title('Creep Timing Stat Comparison'); ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8); ax.legend(fontsize='small')
        for rects in rects_list: # Add value labels
             ax.bar_label(rects, padding=3, fmt='%.f', labels=[format_time(val) if val > 0 else '--:--' for val in [r.get_height() for r in rects]], fontsize=7)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: format_time(y)))
        ax.set_ylim(bottom=0); _apply_matplotlib_dark_style(fig, ax); fig.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png', dpi=100); buf.seek(0); plt.close(fig); return buf.getvalue()
    except Exception as e: print(f"Error generating timing bars: {e}"); plt.close('all'); return None

# --- Text Summary Function ---
def format_stats_text(stats_dict, game_duration_seconds):
    """Formats the stats from a single replay analysis into a string."""
    output = [];
    if not stats_dict: return "No creep statistics generated."
    for pid, player_stats in stats_dict.items():
        output.append("=" * 40); output.append(f"Player: {player_stats.get('player_name', 'Unknown')} (PID: {pid})")
        output.append(f"Replay: {player_stats.get('replay_file', 'Unknown')}"); output.append(f"Map: {player_stats.get('map_name', 'Unknown')}")
        output.append("=" * 40); output.append(f"  Tumors Initiated by Queens: {player_stats.get('tumors_built_by_queen', 0)}")
        output.append(f"  Tumors Initiated by Tumors: {player_stats.get('tumors_built_by_tumor', 0)}")
        total_built = player_stats.get('tumors_built_by_queen', 0) + player_stats.get('tumors_built_by_tumor', 0)
        output.append(f"  Total Tumors Initiated:   {total_built}")
        # output.append(f"  Tumors Died:              {player_stats.get('tumors_died', 0)}") # Removed
        max_tumors = player_stats.get("peak_active_tumors", 0); max_tumor_time = player_stats.get("peak_active_tumors_time", 0)
        final_tumors = player_stats.get("final_active_tumors", 0)
        last_completion_time = min(player_stats.get('last_tumor_time', 0), game_duration_seconds) if player_stats.get('last_tumor_time', 0) > 0 else 0
        output.append(f"  Peak Active Tumors:       {max_tumors} at {format_time(max_tumor_time)}"); output.append(f"  Final Active Tumors:      {final_tumors} at {format_time(game_duration_seconds)}")
        output.append(f"  Last Tumor Completed at:  {format_time(last_completion_time)}"); output.append("-" * 20)
        output.append(f"  First Queen Tumor Cmd at: {format_time(player_stats.get('first_queen_tumor_time'))}")
        output.append(f"  Time to Reach X Tumors:")
        for count in TUMOR_COUNT_MILESTONES: time_val = player_stats.get("time_to_x_tumors", {}).get(count); output.append(f"    {count:>2} Tumors: {format_time(time_val)}")
        avg_interval = player_stats.get("avg_tumor_spread_interval");
        output.append(f"  Avg. Tumor Spread Interval: {avg_interval:.1f}s" if avg_interval is not None else "  Avg. Tumor Spread Interval: N/A")
        output.append("=" * 40)
    return "\n".join(output)

# --- Main execution (Standalone test) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse creep stats and generate Matplotlib plots.")
    parser.add_argument("replay_files", nargs='+', help="Path(s) to replay file(s)")
    parser.add_argument("--output-dir", default="creep_plots_mpl", help="Directory to save plots")
    parser.add_argument("--no-plots", action="store_true", help="Disable plots.")
    args = parser.parse_args()

    if not MATPLOTLIB_AVAILABLE and not args.no_plots: print("Error: Matplotlib required.", file=sys.stderr); sys.exit(1)

    print(f"Analysing {len(args.replay_files)} replay file(s)...")
    all_results = []
    for replay_file_path in args.replay_files:
        if not os.path.exists(replay_file_path): print(f"Error: File not found: {replay_file_path}."); all_results.append((None, 0)); continue
        results, duration = analyze_creep(replay_file_path)
        all_results.append((results, duration))

    valid_results_list = [res for res in all_results if res[0] is not None and res[1] > 0]
    if not valid_results_list: print("\nNo valid data analysed."); sys.exit(1)

    print("\n--- Individual Replay Summaries ---")
    for res, dur in valid_results_list: print(format_stats_text(res, dur))

    if not args.no_plots:
        os.makedirs(args.output_dir, exist_ok=True)
        print("-" * 20)
        # --- Updated plot functions dictionary ---
        plot_functions = {
            "timeline": matplotlib_creep_timeline,
            "initiated": matplotlib_initiated_tumors,
            # "died": matplotlib_died_tumors, # Removed
            "cumulative_avg_interval": matplotlib_cumulative_avg_interval, # Renamed
            "summary_bars": matplotlib_summary_bars,
            "timing_bars": matplotlib_timing_bars,
        }
        for plot_name, plot_func in plot_functions.items():
            img_bytes = plot_func(valid_results_list)
            if img_bytes:
                filename = os.path.join(args.output_dir, f"creep_mpl_{plot_name}.png")
                try:
                    with open(filename, "wb") as f: f.write(img_bytes)
                    print(f"{plot_name.replace('_', ' ').capitalize()} plot saved to: {filename}")
                except Exception as e: print(f"Error saving {plot_name} plot: {e}")
            else: print(f"Could not generate {plot_name} plot.")

    print("\nProcessing finished.")