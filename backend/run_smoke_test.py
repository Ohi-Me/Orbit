"""Run the full pipeline directly (no server) and print a summary.
Usage: python run_smoke_test.py
"""
import json
import sys

from app.orchestrator import run_research_pipeline

if __name__ == "__main__":
    tickers = ["ACME", "BETA", "GAMMA", "DELTA", "EPS", "ZETA"]
    result = run_research_pipeline(tickers=tickers, n_days=756, seed=42, n_folds=4)

    print("=== PIPELINE LOG ===")
    for step in result["pipeline_log"]:
        print(f"  {step['agent']:32s} {step['seconds']:.3f}s")
    print(f"  TOTAL: {result['total_seconds']}s\n")

    print("=== ML RESULT SUMMARY ===")
    print(json.dumps(result["ml_result"].get("summary"), indent=2))

    print("\n=== BACKTEST METRICS ===")
    print(json.dumps(result["backtest_result"].get("metrics"), indent=2))

    print("\n=== RISK ===")
    print(json.dumps({k: v for k, v in result["risk_result"].items() if k != "stress_scenarios"}, indent=2))

    print("\n=== CRITIC VERDICT ===")
    print(result["critic_result"]["overall_verdict"])
    for c in result["critic_result"]["checks"]:
        print(f"  [{c['passed']}] {c['check']}")

    print("\n=== PORTFOLIO (equal_weight) ===")
    print(json.dumps(result["portfolio_result"]["allocations"]["equal_weight"], indent=2))

    print("\n=== FUNDAMENTAL RAG (extractive fallback, no API key) ===")
    print(json.dumps(result.get("fundamental"), indent=2)[:1500])

    print("\n=== REPORT (first 800 chars) ===")
    print(result["report_markdown"][:800])

    ok = (
        result["ml_result"].get("status") == "ok"
        and result["backtest_result"].get("status") == "ok"
        and result["risk_result"].get("status") == "ok"
        and result["portfolio_result"].get("status") == "ok"
    )
    print(f"\nSMOKE TEST {'PASSED' if ok else 'FAILED'}")
    sys.exit(0 if ok else 1)
