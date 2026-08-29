#!/usr/bin/env python3
"""
check_api_spend.py — Calculate API costs since a given date.
Hybrid approach: OpenAI via Usage API, Anthropic via local logs.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from typing import Dict, Any

import httpx

# Add src to sys.path to allow importing from cost_tracker and utils
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from cost_tracker import CostTracker, MODEL_PRICING
    from utils import SECRETS
except ImportError:
    print("Error: Could not import cost_tracker or utils from src/.")
    sys.exit(1)


def get_openai_usage(date_str: str, api_key: str) -> Dict[str, Any]:
    """Fetch OpenAI usage for a specific date (YYYY-MM-DD)."""
    url = f"https://api.openai.com/v1/usage?date={date_str}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers)
            if response.status_code != 200:
                print(f"  [OpenAI] Warning: Failed to fetch usage for {date_str} (HTTP {response.status_code})")
                return {}
            return response.json()
    except Exception as e:
        print(f"  [OpenAI] Error fetching usage for {date_str}: {e}")
        return {}


def normalize_model_name(model: str) -> str:
    """Coalesce specific model versions into base names (e.g. gpt-5.1-2025 -> gpt-5.1)."""
    if model in MODEL_PRICING:
        return model
    # Try prefix matching, longest first
    for key in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
        if model.startswith(key):
            return key
    return model


def process_openai_usage(usage_data: Dict[str, Any], tracker: CostTracker):
    """Aggregate OpenAI usage data into the tracker."""
    data = usage_data.get("data", [])
    if not data:
        return

    for entry in data:
        model = normalize_model_name(entry.get("snapshot_id", "unknown"))
        input_tokens = entry.get("n_context_tokens_total", 0)
        output_tokens = entry.get("n_generated_tokens_total", 0)
        
        if not input_tokens and not output_tokens:
            continue
            
        tracker.record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )


def aggregate_local_reports(start_date: datetime, tracker: CostTracker):
    """Scan output/ for cost_report.json files since start_date.
    
    Returns:
        tuple: (count, ablation_costs) where ablation_costs is a dict mapping 
               ablation name to total cost for that ablation.
    """
    output_dir = Path(__file__).parent.parent / "output"
    if not output_dir.is_dir():
        print(f"Warning: output directory not found at {output_dir}")
        return 0, {}

    count = 0
    ablation_costs = {}
    
    # Search for all cost_report.json files
    for cost_file in output_dir.rglob("cost_report.json"):
        # SKIP merged artifacts to avoid double-counting
        if "my_merge" in cost_file.parts or "artifacts" in cost_file.parts:
            continue

        file_date = None
        match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", str(cost_file))
        if match:
            try:
                file_date = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except ValueError:
                pass
        
        if file_date is None:
            mtime = os.path.getmtime(cost_file)
            file_date = datetime.fromtimestamp(mtime)

        if file_date >= start_date:
            # Try to infer ablation name from the path
            # Typical path: output/ablation_sweeps/YYYYMMDD_HHMMSS/ABLATION_NAME/accuracy_test_artifacts/...
            ablation_name = "unknown"
            parts = cost_file.parts
            try:
                if "ablation_sweeps" in parts:
                    idx = parts.index("ablation_sweeps")
                    if len(parts) > idx + 2:
                        ablation_name = parts[idx + 2]
                elif "pipeline_outputs" in parts:
                    ablation_name = "pipeline_run"
            except (ValueError, IndexError):
                pass

            try:
                with open(cost_file, "r") as f:
                    report = json.load(f)
                    
                    if ablation_name not in ablation_costs:
                        ablation_costs[ablation_name] = 0.0

                    for model_name, data in report.get("models", {}).items():
                        base_model = normalize_model_name(model_name)
                        
                        input_tokens = data.get("input_tokens", 0)
                        output_tokens = data.get("output_tokens", 0)
                        cached_tokens = data.get("cached_input_tokens", 0)
                        creation_tokens = data.get("cache_creation_tokens", 0)
                        reported_cost = data.get("cost_usd", 0.0)
                        
                        is_batch = False
                        pricing = MODEL_PRICING.get(base_model)
                        if pricing and reported_cost > 0:
                            calc = (
                                input_tokens / 1_000_000.0 * pricing["input"] +
                                output_tokens / 1_000_000.0 * pricing["output"]
                            )
                            if cached_tokens:
                                cp = pricing.get("cached_input", pricing["input"] * 0.1)
                                calc += (cached_tokens / 1_000_000.0 * cp)
                            if creation_tokens:
                                crp = pricing.get("cache_creation", pricing["input"] * 1.25)
                                calc += (creation_tokens / 1_000_000.0 * crp)
                            
                            if abs(reported_cost - (calc * 0.5)) < abs(reported_cost - calc):
                                is_batch = True

                        tracker.record(
                            model=base_model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_input_tokens=cached_tokens,
                            cache_creation_tokens=creation_tokens,
                            is_batch=is_batch
                        )
                        
                    ablation_costs[ablation_name] += report.get("total_cost_usd", 0.0)
                    count += 1
            except Exception as e:
                print(f"Warning: Failed to parse {cost_file}: {e}")
    
    return count, ablation_costs


def main():
    parser = argparse.ArgumentParser(description="Check API spend (Local Logs + OpenAI API Sync).")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD). Defaults to first of current month.")
    parser.add_argument("--use-api", action="store_true", help="Attempt to sync OpenAI usage from API (may require specific permissions).")
    args = parser.parse_args()

    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print("Error: Invalid date format. Use YYYY-MM-DD.")
            sys.exit(1)
    else:
        # Default to first of current month
        today = date_type.today()
        start_date = datetime(today.year, today.month, 1)

    print(f"Checking API spend since {start_date.strftime('%Y-%m-%d')}...")

    tracker = CostTracker()
    
    # 1. Local Reports (Anthropic + OpenAI)
    print("Aggregating usage from local logs (output/)...")
    local_count, ablation_costs = aggregate_local_reports(start_date, tracker)
    print(f"  Processed {local_count} local cost reports.")

    # 2. OpenAI (API Sync - Optional)
    if args.use_api:
        openai_key = SECRETS.get("openai_key")
        if openai_key:
            print("Syncing OpenAI usage from API...")
            current_date = start_date
            end_date = datetime.now()
            
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                usage = get_openai_usage(date_str, openai_key)
                process_openai_usage(usage, tracker)
                current_date += timedelta(days=1)
        else:
            print("Warning: 'openai_key' not found in secrets. Cannot sync from API.")

    # 3. Report
    output_dir = Path(__file__).parent.parent / "output"
    report = tracker.get_report()
    total_cost = report.get("total_cost_usd", 0.0)
    
    # Split by provider
    openai_cost = 0.0
    anthropic_cost = 0.0
    
    print("\n" + "=" * 60)
    print(f"{'Model Breakdown':<40} | {'Cost (USD)':>12}")
    print("-" * 60)
    
    models = report.get("models", {})
    for model, data in sorted(models.items(), key=lambda x: x[1].get('cost_usd', 0), reverse=True):
        cost = data.get("cost_usd", 0.0)
        print(f"{model:<40} | ${cost:>11.4f}")
        if "claude" in model.lower():
            anthropic_cost += cost
        else:
            openai_cost += cost
            
    print("-" * 60)
    print(f"{'OpenAI Total':<40} | ${openai_cost:>11.4f}")
    print(f"{'Anthropic Total':<40} | ${anthropic_cost:>11.4f}")
    print("=" * 60)
    
    if ablation_costs:
        print("\n" + "=" * 60)
        print(f"{'Ablation Test':<40} | {'Cost (USD)':>12}")
        print("-" * 60)
        for ablation, cost in sorted(ablation_costs.items(), key=lambda x: x[1], reverse=True):
            print(f"{ablation:<40} | ${cost:>11.4f}")
        print("=" * 60)

    print(f"{'GRAND TOTAL':<40} | ${total_cost:>11.4f}")
    print("=" * 60)
    print("Note: Costs are aggregated from local logs by default.")
    if not args.use_api:
        print("Run with --use-api to attempt syncing actual usage from OpenAI API.")



if __name__ == "__main__":
    main()
