#!/usr/bin/env python3
import argparse
import json
import os
import sys
from stem_agent import StemAgent

def main():
    parser = argparse.ArgumentParser(description="Run the stem agent on a problem domain")
    parser.add_argument("domain", nargs="?", default="Deep Research",
                        help='Problem domain to specialize into (default: "Deep Research")')
    parser.add_argument("--force-rediscover", action="store_true",
                        help="Bypass discovery cache and re-run discovery")
    parser.add_argument("--results-dir", default="results",
                        help="Directory to save iteration scores and final agent (default: results)")
    args = parser.parse_args()

    print(f"Starting stem agent for domain: '{args.domain}'")
    print("=" * 60)

    agent = StemAgent(results_dir=args.results_dir)
    final_state = agent.run(args.domain, force_rediscover=args.force_rediscover)

    print("\n" + "=" * 60)
    print(f"Evolution complete.")
    print(f"Final version: v{final_state.version}")
    print(f"Final composite score: {final_state.composite_score:.1f} / 100")
    print(f"Active tools: {', '.join(final_state.active_tools)}")
    print(f"Specialized agent saved to: {args.results_dir}/final_agent/")
    print(f"Run standalone agent: python {args.results_dir}/final_agent/run.py")

    scores_path = f"{args.results_dir}/scores.json"
    if os.path.exists(scores_path):
        with open(scores_path) as f:
            history = json.load(f)
        print("\nIteration scores:")
        print(f"{'Iter':>4} {'Composite':>10} {'Accuracy':>9} {'Coverage':>9} {'Synthesis':>10} {'Citation':>9}")
        for h in history:
            print(f"{h['iteration']:>4} {h['composite']:>10.1f} {h['accuracy']:>9.1f} "
                  f"{h['coverage']:>9.1f} {h['synthesis']:>10.1f} {h['citation']:>9.1f}")

if __name__ == "__main__":
    main()
