# -------------------------------------------------------------------------
# Code Snippet for Creep Analysis Integration into Existing Discord Bot
# (Restored Tumors Died Feature)
# -------------------------------------------------------------------------
# Assumes the bot's main file already imports discord, commands, os, io, traceback, sys
# Assumes 'parse_creep.py' (Revision 12.1 - Matplotlib, Died stats restored) is present.

# --- 1. Add Imports ---
try:
    import parse_creep
    # Check if matplotlib is available
    CREEP_PLOTTING_AVAILABLE = parse_creep.MATPLOTLIB_AVAILABLE
except ImportError:
    print("ERROR: parse_creep.py not found or missing dependencies.")
    CREEP_PLOTTING_AVAILABLE = False
    # Dummy class to prevent attribute errors
    class parse_creep:
        @staticmethod
        def get_player_info(*args): return []
        @staticmethod
        def is_zvz(*args): return False
        @staticmethod
        def analyze_creep(*args): return None, 0
        @staticmethod
        def matplotlib_creep_timeline(*args): return None
        @staticmethod
        def matplotlib_initiated_tumors(*args): return None
        @staticmethod
        def matplotlib_died_tumors(*args): return None # Restored
        @staticmethod
        def matplotlib_cumulative_avg_interval(*args): return None
        @staticmethod
        def matplotlib_summary_bars(*args): return None
        @staticmethod
        def matplotlib_timing_bars(*args): return None
        @staticmethod
        def format_time(s, none_val="--:--"): return none_val
        TUMOR_COUNT_MILESTONES = []


# --- 2. Ensure Reaction Intent is Enabled ---
# Check the bot's intent setup includes: intents.reactions = True

# --- 3. Ensure Interaction Context Dict Exists ---
# The original bot likely has `reaction_contexts = {}`. We will reuse this.

# --- 4. Add Helper Functions for Creep ---
# (Can be placed anywhere, maybe rename with _creep_ prefix)

def _creep_create_error_embed(title, description):
    return discord.Embed(title=f":x: {title}", description=description, color=discord.Color.red())

def _creep_create_info_embed(title, description):
     return discord.Embed(title=f":information_source: {title}", description=description, color=discord.Color.blue())

def _creep_create_stats_embed(results_dict, game_duration_seconds):
    """Creates a Discord embed summarizing the creep stats. (Restored Died)"""
    if not results_dict or len(results_dict) != 1: return None
    pid = list(results_dict.keys())[0]; stats = results_dict[pid]
    player_name=stats.get('player_name', f'P{pid}'); replay_file=stats.get('replay_file','N/A'); map_name=stats.get('map_name','N/A')
    embed = discord.Embed(title=f"Creep Analysis: {player_name}", description=f"Replay: `{replay_file}` | Map: `{map_name}` | Duration: {parse_creep.format_time(game_duration_seconds)}", color=discord.Color.dark_purple())
    embed.add_field(name="<:CreepTumorBurrowed:111111111111111111> Tumors Initiated", value=f"Queen: {stats.get('tumors_built_by_queen', 0)}\nTumor: {stats.get('tumors_built_by_tumor', 0)}\n**Total: {stats.get('tumors_built_by_queen', 0) + stats.get('tumors_built_by_tumor', 0)}**", inline=True)
    max_tumors=stats.get("peak_active_tumors", 0); max_tumor_time=stats.get("peak_active_tumors_time", 0); final_tumors=stats.get("final_active_tumors", 0)
    # Restored Died count display
    embed.add_field(name="<:CreepTumor:111111111111111111> Active / Died", value=f"Peak: {max_tumors} at {parse_creep.format_time(max_tumor_time)}\nFinal: {final_tumors}\nDied: {stats.get('tumors_died', 0)}", inline=True)
    embed.add_field(name="\u200B", value="\u200B", inline=True)
    last_completion_time = min(stats.get('last_tumor_time', 0), game_duration_seconds) if stats.get('last_tumor_time', 0) > 0 else 0
    timing_str = f"First Queen Cmd: {parse_creep.format_time(stats.get('first_queen_tumor_time'))}\n"
    timing_str += f"Last Tumor Done: {parse_creep.format_time(last_completion_time)}\n"
    avg_interval = stats.get("avg_tumor_spread_interval"); timing_str += f"Avg Spread Interval: {avg_interval:.1f}s" if avg_interval is not None else "Avg Spread Interval: N/A"
    embed.add_field(name=":timer: Key Timings", value=timing_str, inline=True)
    milestone_str = "";
    for count in parse_creep.TUMOR_COUNT_MILESTONES: time_val = stats.get("time_to_x_tumors", {}).get(count); milestone_str += f"{count:>2} Tumors: {parse_creep.format_time(time_val)}\n"
    embed.add_field(name=":chart_with_upwards_trend: Time to X Tumors", value=milestone_str.strip(), inline=True)
    embed.set_footer(text="Generated by Creep Analyser")
    return embed

def _creep_create_comparison_embed(results_list):
     """Creates embed comparing stats. (Restored Died)"""
     if not results_list: return None
     valid_results = [(stats, dur) for stats, dur in results_list if stats and dur > 0 and stats and len(stats) == 1]
     if not valid_results: return _creep_create_info_embed("Comparison Failed", "No valid analyses found.")
     num_replays = len(valid_results); embed = discord.Embed(title=f"Creep Comparison ({num_replays} Replays)", color=discord.Color.gold())
     stats_to_compare = [
        ("Tumors (Queen)", "tumors_built_by_queen", False), ("Tumors (Tumor)", "tumors_built_by_tumor", False),
        ("Peak Active", "peak_active_tumors", False), ("Peak Time", "peak_active_tumors_time", True),
        ("Final Active", "final_active_tumors", False), ("Total Died", "tumors_died", False), # Restored
        ("First Queen T.", "first_queen_tumor_time", True), ("Avg Spread (s)", "avg_tumor_spread_interval", False, ".1f"),
     ]
     for count in parse_creep.TUMOR_COUNT_MILESTONES: stats_to_compare.append( (f"Time to {count}", lambda s, c=count: s.get("time_to_x_tumors", {}).get(c), True) )
     for label, key_or_func, is_time, *fmt in stats_to_compare:
         value_str = ""
         for i, (stats_dict, duration) in enumerate(valid_results):
             pid=list(stats_dict.keys())[0]; player_stats=stats_dict[pid]; player_name=player_stats.get('player_name',f'P{pid}')[:10]; replay_name=player_stats.get('replay_file',f'R{i+1}')[:10]
             value = "N/A";
             try:
                 if callable(key_or_func): val = key_or_func(player_stats)
                 else: val = player_stats.get(key_or_func)
                 if val is not None: value = parse_creep.format_time(val) if is_time else f"{val:{fmt[0]}}" if fmt else str(val)
                 else: value = "--:--" if is_time else "N/A"
             except Exception: value = "Error"
             value_str += f"**`{replay_name}({player_name})`**: {value}\n"
         inline_choice = len(valid_results) <= 2 or len(value_str) < 70
         embed.add_field(name=label, value=value_str.strip() if value_str.strip() else "No data", inline=inline_choice)
     inline_fields = sum(1 for f in embed.fields if f.inline);
     if inline_fields % 3 == 2 : embed.add_field(name="\u200B", value="\u200B", inline=True)
     embed.set_footer(text="Generated by Creep Analyser")
     return embed

async def _creep_process_and_send_results(ctx_or_user, replay_file_path, selected_pid=None, is_comparison=False, results_list_override=None):
    """Runs creep analysis and sends results/plots to Discord."""
    target_channel = ctx_or_user.channel if isinstance(ctx_or_user, commands.Context) else ctx_or_user
    analysis_success = False; files_to_send = []
    unique_id = ctx_or_user.message.id if isinstance(ctx_or_user, commands.Context) else int(discord.utils.utcnow().timestamp())

    try:
        results_list_for_plot = []
        if results_list_override: # Comparison path
            valid_results = [(stats, dur) for stats, dur in results_list_override if stats and dur > 0]
            if not valid_results: raise ValueError("No valid pre-analyzed results.")
            print("Processing pre-analyzed creep results for comparison...")
            comparison_embed = _creep_create_comparison_embed(valid_results) # Use helper
            if comparison_embed: await target_channel.send(embed=comparison_embed)
            analysis_success = True; results_list_for_plot = valid_results
        else: # Single replay path
            print(f"Running creep analysis for {os.path.basename(replay_file_path)}...")
            results, duration = parse_creep.analyze_creep(replay_file_path, selected_pid=selected_pid)
            if not results or duration <= 0: await target_channel.send(embed=_creep_create_error_embed("Analysis Failed", f"Could not extract valid data from `{os.path.basename(replay_file_path)}`.")); return
            analysis_success = True; stats_embed = _creep_create_stats_embed(results, duration) # Use helper
            if stats_embed: await target_channel.send(embed=stats_embed)
            results_list_for_plot = [(results, duration)]

        # Generate Matplotlib Plots
        if CREEP_PLOTTING_AVAILABLE and analysis_success:
            print("Generating creep plots...")
            # --- UPDATED Plot list (Restored Died Plot) ---
            plot_funcs = {
                "timeline": parse_creep.matplotlib_creep_timeline,
                "initiated": parse_creep.matplotlib_initiated_tumors,
                "died": parse_creep.matplotlib_died_tumors, # Restored
                "cumulative_avg_interval": parse_creep.matplotlib_cumulative_avg_interval,
                "summary_bars": parse_creep.matplotlib_summary_bars,
                "timing_bars": parse_creep.matplotlib_timing_bars,
            }
            # Determine base filename
            filename_base = f"comp_{unique_id}_creep_" if is_comparison else f"single_{unique_id}_creep_"
            if not is_comparison:
                 first_res = results_list_for_plot[0][0]; pid = list(first_res.keys())[0]
                 replay_name = os.path.splitext(first_res[pid].get('replay_file', 'r'))[0][:15]
                 filename_base = f"{replay_name}_p{pid}_creep_"

            for name, func in plot_funcs.items():
                img_bytes = func(results_list_for_plot)
                if img_bytes: files_to_send.append(discord.File(fp=io.BytesIO(img_bytes), filename=f"{filename_base}{name}.png"))

            if not files_to_send: await target_channel.send("*(Could not generate plot images)*")
        elif analysis_success: await target_channel.send("*(Graphing disabled: Matplotlib library not available)*")

        # Send Files
        if files_to_send:
             print(f"Sending {len(files_to_send)} creep plot(s)...")
             for i in range(0, len(files_to_send), 10): await target_channel.send(files=files_to_send[i:i+10])

    except Exception as e:
        await target_channel.send(embed=_creep_create_error_embed("Processing Error", f"Error processing creep results: {type(e).__name__}"))
        print(f"Error during _creep_process_and_send_results: {e}"); traceback.print_exc()
    finally: # Cleanup downloaded file for single analysis
         if not is_comparison and replay_file_path and os.path.exists(replay_file_path):
             try: os.remove(replay_file_path);
             except OSError as rm_err: print(f"Error removing temp file {replay_file_path}: {rm_err}")


# --- Step 5: Add Bot Commands ---
# (Place these within the bot's class or globally depending on structure)

@bot.command(name='creep', help="Analyses creep spread from one SC2 replay.\nUsage: `!creep` (attach one .SC2Replay file)")
async def creep(ctx):
    """Command to analyse a single replay file for creep stats."""
    if not ctx.message.attachments or len(ctx.message.attachments) != 1: await ctx.send(embed=_creep_create_error_embed("Invalid Attachment", "Attach exactly one `.SC2Replay` file.")); return
    attachment = ctx.message.attachments[0];
    if not attachment.filename.lower().endswith('.sc2replay'): await ctx.send(embed=_creep_create_error_embed("Invalid File Type", "File must be `.SC2Replay`.")); return
    temp_filename = f"temp_creep_{ctx.message.id}_{attachment.filename}"; replay_file_path = f'./{temp_filename}'
    await attachment.save(replay_file_path); ack_msg = await ctx.send(f":hourglass_flowing_sand: Processing creep for: `{attachment.filename}`...")
    try:
        player_info_list = parse_creep.get_player_info(replay_file_path)
        if not player_info_list: await ack_msg.edit(content="", embed=_creep_create_error_embed("Analysis Failed", "Could not read player info.")); return
        if parse_creep.is_zvz(player_info_list):
            embed = discord.Embed(title="ZvZ Detected - Choose Player", description=f"Replay `{attachment.filename}` is ZvZ. React:", color=discord.Color.dark_purple())
            reactions = [];
            for i, player in enumerate(player_info_list): reaction_emoji = f"{i+1}\uFE0F\u20E3"; embed.add_field(name=f"{reaction_emoji} P{player['pid']}", value=f"{player['name']}", inline=False); reactions.append(reaction_emoji)
            await ack_msg.delete(); choose_msg = await ctx.send(embed=embed)
            for r in reactions: await choose_msg.add_reaction(r)
            # Use the existing reaction_contexts dict
            reaction_contexts[ctx.author.id] = { "type": "creep_single_zvz_select", "message_id": choose_msg.id, "replay_file": replay_file_path, "players": player_info_list, "original_ctx": ctx }
        else: # Not ZvZ
            await ack_msg.delete(); zerg_pid = next((p['pid'] for p in player_info_list if p['race'] == 'Zerg'), None)
            await _creep_process_and_send_results(ctx, replay_file_path, selected_pid=zerg_pid, is_comparison=False)
    except Exception as e:
        await ack_msg.edit(content="", embed=_creep_create_error_embed("Analysis Error", f"Error: {type(e).__name__}"))
        print(f"Error during !creep: {e}"); traceback.print_exc()
    finally: # Cleanup only if no interaction pending
        if ctx.author.id not in reaction_contexts and os.path.exists(replay_file_path):
             try: os.remove(replay_file_path)
             except OSError: pass

@bot.command(name='compare_creep', help="Compares creep stats between two SC2 replays.\nUsage: `!compare_creep` (attach two .SC2Replay files)")
async def compare_creep(ctx):
    """Command to compare creep stats between two replays."""
    if not ctx.message.attachments or len(ctx.message.attachments) != 2: await ctx.send(embed=_creep_create_error_embed("Invalid Attachment", "Attach exactly two `.SC2Replay` files.")); return
    replay_paths = []; temp_files = []
    for i, attachment in enumerate(ctx.message.attachments):
         if not attachment.filename.lower().endswith('.sc2replay'): await ctx.send(embed=_creep_create_error_embed("Invalid File Type", f"`{attachment.filename}` isn't `.SC2Replay`.")); return
         temp_filename = f"temp_comp_creep_{ctx.message.id}_{i}_{attachment.filename}"; file_path = f'./{temp_filename}'
         await attachment.save(file_path); replay_paths.append(file_path); temp_files.append(file_path)
    ack_msg = await ctx.send(f":hourglass_flowing_sand: Processing {len(replay_paths)} replays for creep comparison...")
    all_results = []; analysis_success = False
    try:
        for replay_file_path in replay_paths:
             results, duration = parse_creep.analyze_creep(replay_file_path) # Basic: analyses first zerg
             all_results.append((results, duration))
             if results and duration > 0: analysis_success = True
        await ack_msg.delete()
        await _creep_process_and_send_results(ctx, None, results_list_override=all_results, is_comparison=True) # Use helper
    except Exception as e:
        await ack_msg.edit(content="", embed=_creep_create_error_embed("Comparison Error", f"Error: {type(e).__name__}"))
        print(f"Error during !compare_creep: {e}"); traceback.print_exc()
    finally: # Cleanup
        for p in temp_files:
            if os.path.exists(p):
                 try: os.remove(p);
                 except OSError as rm_err: print(f"Error removing temp file {p}: {rm_err}")


# --- Step 6: Add to existing Reaction Handler ---

@bot.event
async def on_reaction_add(reaction, user):
    if user == bot.user: return # Ignore self
    # --- !! Use the reaction_contexts dictionary from the main bot file !! ---
    context = reaction_contexts.get(user.id)
    if not context: return

    target_message_id = context.get("message_id")
    # Add handling for potential list of message_ids if compare ZvZ is implemented later
    is_relevant_message = (reaction.message.id == target_message_id)
    if not is_relevant_message: return


    # --- ADD THIS BLOCK to the existing on_reaction_add function ---
    if context.get("type") == "creep_single_zvz_select": # Check context type for creep
        print(f"Processing Creep ZvZ reaction for user {user.id}...") # Debug log
        try:
            player_choice_index = -1
            for i, p_info in enumerate(context["players"]):
                if reaction.emoji == f"{i+1}\uFE0F\u20E3": player_choice_index = i; break

            if player_choice_index != -1:
                selected_player_info = context["players"][player_choice_index]; selected_pid = selected_player_info["pid"]
                replay_file = context["replay_file"]; original_ctx = context["original_ctx"]
                try: await reaction.message.delete()
                except (discord.Forbidden, discord.NotFound): pass
                await original_ctx.send(f":hourglass_flowing_sand: Analysing creep for player **{selected_player_info['name']}** (PID {selected_pid})...")
                # Remove context *before* processing
                del reaction_contexts[user.id]
                # Call the main processing logic
                await _creep_process_and_send_results(original_ctx, replay_file, selected_pid=selected_pid, is_comparison=False)
            # else: Ignore invalid reactions quietly
        except Exception as e:
             print(f"Error processing Creep ZvZ reaction: {e}"); traceback.print_exc()
             if user.id in reaction_contexts: del reaction_contexts[user.id] # Cleanup context
             await reaction.message.channel.send(":warning: An error occurred processing creep selection.")
             if "replay_file" in context and os.path.exists(context["replay_file"]): # Cleanup file
                  try: os.remove(context["replay_file"])
                  except OSError: pass
        return # Prevent falling through to other reaction type handlers
    # --- END BLOCK TO ADD ---

    # --- Existing reaction handling logic from thelurker.py below ---
    # elif context.get("type") == "larva_analyze_zvz_select":
    #     # ... original larva handling ...
    # elif context.get("type") == "larva_compare_zvz_select":
    #     # ... original larva handling ...

# -------------------------------------------------------------------------
# End of Snippet
# -------------------------------------------------------------------------