# SC2 Discord Creep Coverage Analyzer

A Python-based Discord bot that analyzes StarCraft II (`.SC2Replay`) files to provide detailed statistics and visualizations on Zerg creep spread.

## Features

* **Single Replay Analysis:** Analyze a single replay file attached to the `!creep` command.
* **Comparison Analysis:** Compare creep spread metrics between two replay files using the `!compare_creep` command.
* **ZvZ Handling:** Prompts the user to select which Zerg player to analyze in Zerg vs Zerg matchups via reactions.
* **Detailed Statistics:** Calculates various metrics including:
    * Tumors initiated (by Queen, by Tumor, Total).
    * Peak and final active tumor counts.
    * Key timings (First Queen tumor command, last tumor completion, time to reach 5/10/15/20 active tumors).
    * Average tumor spread interval.
* **Data Visualization (Optional):** If `matplotlib` and `numpy` are installed, generates various plots:
    * Active Tumor Count Timeline (Comparison).
    * Cumulative Initiated Tumors (Queen vs Tumor) per Replay.
    * Cumulative Average Spread Interval per Replay.
    * Summary Statistics Bar Chart (Comparison).
    * Timing Statistics Bar Chart (Comparison).
* **Discord Integration:** Presents analysis results and plots directly in Discord using embeds and file uploads.
* **Integration Snippet:** Provides code (`integration_snippet.py`) to help integrate this functionality into an existing discord.py bot.

## Requirements

* Python 3.x
* `discord.py` library (`pip install -U discord.py`)
* `sc2reader` library (`pip install -U sc2reader`)
* **Optional (for plotting):**
    * `matplotlib` (`pip install -U matplotlib`)
    * `numpy` (`pip install -U numpy`)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -U discord.py sc2reader matplotlib numpy
    ```
    *(Note: `matplotlib` and `numpy` are optional but required for plot generation)*
3.  **Configure Bot Token:**
    * Open `creep_bot.py`.
    * Replace `'YOUR_BOT_TOKEN_HERE'` with your actual Discord Bot Token. **Keep your token secret!**
4.  **Run the bot:**
    ```bash
    python creep_bot.py
    ```

## Usage

* **Analyze a single replay:**
    In a Discord channel where the bot is present, type `!creep` and attach one `.SC2Replay` file to the message.
* **Compare two replays:**
    In a Discord channel where the bot is present, type `!compare_creep` and attach exactly two `.SC2Replay` files to the message.
* **ZvZ Player Selection:**
    If a ZvZ replay is detected during `!creep`, the bot will post a message asking you to react with the corresponding number emoji to choose which player to analyze.

## Files

* `creep_bot.py`: The main Discord bot script.
* `parse_creep.py`: Handles replay parsing, analysis, and plotting logic.
* `integration_snippet.py`: Example code for integrating the analysis into an existing bot.
* `README.md`: This file.

## Dependencies

* [discord.py](https://github.com/Rapptz/discord.py)
* [sc2reader](https://github.com/ggtracker/sc2reader)
* [matplotlib](https://matplotlib.org/) (Optional)
* [NumPy](https://numpy.org/) (Optional)