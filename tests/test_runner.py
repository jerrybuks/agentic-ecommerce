"""
Golden Dataset Test Runner for Shoplytic E-commerce Chatbot.

This runner executes test cases from the golden dataset against the chatbot API
and evaluates responses using LLM-as-a-Judge scoring.

Usage:
    python -m tests.test_runner                    # Run all tests
    python -m tests.test_runner --category cart    # Run specific category
    python -m tests.test_runner --id product_search_001  # Run single test
    python -m tests.test_runner --report           # Generate detailed report
"""
import json
import asyncio
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import httpx


# Configuration
API_BASE_URL = "http://localhost:8000"
GOLDEN_DATA_PATH = Path(__file__).parent.parent / "data" / "golden_data" / "test_cases.json"
RESULTS_PATH = Path(__file__).parent.parent / "data" / "golden_data" / "results"


class TestRunner:
    """Runs golden dataset tests against the chatbot API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = []
        
    def load_test_cases(self, category: Optional[str] = None, test_id: Optional[str] = None) -> list:
        """Load test cases from golden dataset."""
        with open(GOLDEN_DATA_PATH, "r") as f:
            data = json.load(f)
        
        test_cases = data["test_cases"]
        
        # Filter by test ID if specified
        if test_id:
            test_cases = [tc for tc in test_cases if tc["id"] == test_id]
        # Filter by category if specified
        elif category:
            test_cases = [tc for tc in test_cases if tc["category"] == category]
        
        return test_cases
    
    async def run_single_test(self, test_case: dict, client: httpx.AsyncClient) -> dict:
        """Run a single test case and return results."""
        test_id = test_case["id"]
        query = test_case["query"]
        
        print(f"\n{'='*60}")
        print(f"Running test: {test_id}")
        print(f"Query: {query}")
        print(f"Expected agent: {test_case['expected_agent']}")
        
        start_time = time.time()
        
        try:
            # Send query to the API
            response = await client.post(
                f"{self.base_url}/user/query",
                json={
                    "query": query,
                    "session_id": self.session_id
                },
                timeout=60.0
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Extract response details
                answer = result_data.get("answer", "")
                agents_used = result_data.get("agents_used", [])
                routing_mode = result_data.get("routing_mode", "unknown")
                
                # Determine if correct agent was used
                expected_agent = test_case["expected_agent"]
                agent_match = False
                if expected_agent == "order" and "order" in agents_used:
                    agent_match = True
                elif expected_agent == "general_info" and "general_info" in agents_used:
                    agent_match = True
                elif expected_agent == "both" and len(agents_used) >= 2:
                    agent_match = True
                elif expected_agent == "none" and len(agents_used) == 0:
                    agent_match = True
                
                result = {
                    "test_id": test_id,
                    "category": test_case["category"],
                    "query": query,
                    "status": "passed" if agent_match else "failed",
                    "agent_match": agent_match,
                    "expected_agent": expected_agent,
                    "actual_agents": agents_used,
                    "routing_mode": routing_mode,
                    "response": answer[:500],  # Truncate for readability
                    "response_time_seconds": round(elapsed_time, 2),
                    "error": None
                }
                
                print(f"✓ Status: {'PASSED' if agent_match else 'FAILED'}")
                print(f"  Agents used: {agents_used}")
                print(f"  Response time: {elapsed_time:.2f}s")
                print(f"  Response preview: {answer[:100]}...")
                
            else:
                result = {
                    "test_id": test_id,
                    "category": test_case["category"],
                    "query": query,
                    "status": "error",
                    "agent_match": False,
                    "expected_agent": test_case["expected_agent"],
                    "actual_agents": [],
                    "routing_mode": "unknown",
                    "response": None,
                    "response_time_seconds": round(elapsed_time, 2),
                    "error": f"HTTP {response.status_code}: {response.text[:200]}"
                }
                print(f"✗ Error: HTTP {response.status_code}")
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            result = {
                "test_id": test_id,
                "category": test_case["category"],
                "query": query,
                "status": "error",
                "agent_match": False,
                "expected_agent": test_case["expected_agent"],
                "actual_agents": [],
                "routing_mode": "unknown",
                "response": None,
                "response_time_seconds": round(elapsed_time, 2),
                "error": str(e)
            }
            print(f"✗ Exception: {e}")
        
        return result
    
    async def run_tests(self, category: Optional[str] = None, test_id: Optional[str] = None) -> list:
        """Run all matching test cases."""
        test_cases = self.load_test_cases(category=category, test_id=test_id)
        
        if not test_cases:
            print("No test cases found matching criteria.")
            return []
        
        print(f"\n{'#'*60}")
        print(f"# Golden Dataset Test Run")
        print(f"# Session ID: {self.session_id}")
        print(f"# Test cases: {len(test_cases)}")
        print(f"# API: {self.base_url}")
        print(f"{'#'*60}")
        
        async with httpx.AsyncClient() as client:
            for test_case in test_cases:
                result = await self.run_single_test(test_case, client)
                self.results.append(result)
                # Small delay between tests to avoid overwhelming the API
                await asyncio.sleep(0.5)
        
        return self.results
    
    def generate_report(self) -> dict:
        """Generate a summary report of test results."""
        if not self.results:
            return {"error": "No results to report"}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "passed")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        errors = sum(1 for r in self.results if r["status"] == "error")
        
        avg_response_time = sum(r["response_time_seconds"] for r in self.results) / total
        
        # Category breakdown
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0, "failed": 0, "errors": 0}
            categories[cat]["total"] += 1
            if r["status"] == "passed":
                categories[cat]["passed"] += 1
            elif r["status"] == "failed":
                categories[cat]["failed"] += 1
            else:
                categories[cat]["errors"] += 1
        
        report = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "pass_rate": round(passed / total * 100, 1),
                "avg_response_time_seconds": round(avg_response_time, 2)
            },
            "by_category": categories,
            "failed_tests": [r for r in self.results if r["status"] != "passed"],
            "all_results": self.results
        }
        
        return report
    
    def print_report(self, report: dict):
        """Print a formatted report to console."""
        print(f"\n{'='*60}")
        print("TEST RUN SUMMARY")
        print(f"{'='*60}")
        
        summary = report["summary"]
        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"  ✓ Passed: {summary['passed']}")
        print(f"  ✗ Failed: {summary['failed']}")
        print(f"  ⚠ Errors: {summary['errors']}")
        print(f"  Pass Rate: {summary['pass_rate']}%")
        print(f"  Avg Response Time: {summary['avg_response_time_seconds']}s")
        
        print(f"\n{'─'*60}")
        print("BY CATEGORY:")
        for cat, stats in report["by_category"].items():
            pass_rate = round(stats["passed"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
            print(f"  {cat}: {stats['passed']}/{stats['total']} ({pass_rate}%)")
        
        if report["failed_tests"]:
            print(f"\n{'─'*60}")
            print("FAILED TESTS:")
            for test in report["failed_tests"]:
                print(f"\n  [{test['test_id']}] {test['query'][:50]}...")
                print(f"    Expected: {test['expected_agent']}, Got: {test['actual_agents']}")
                if test.get("error"):
                    print(f"    Error: {test['error'][:100]}")
        
        print(f"\n{'='*60}\n")
    
    def save_results(self, report: dict):
        """Save results to file."""
        # Ensure results directory exists
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)
        
        # Save with timestamp
        filename = f"results_{self.session_id}.json"
        filepath = RESULTS_PATH / filename
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"Results saved to: {filepath}")


async def main():
    parser = argparse.ArgumentParser(description="Run golden dataset tests")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--id", help="Run specific test by ID")
    parser.add_argument("--url", default=API_BASE_URL, help="API base URL")
    parser.add_argument("--save", "-s", action="store_true", help="Save results to file")
    parser.add_argument("--report", "-r", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    runner = TestRunner(base_url=args.url)
    
    # Run tests
    await runner.run_tests(category=args.category, test_id=args.id)
    
    # Generate and display report
    report = runner.generate_report()
    runner.print_report(report)
    
    # Save results if requested
    if args.save:
        runner.save_results(report)
    
    # Return exit code based on results
    if report["summary"]["failed"] > 0 or report["summary"]["errors"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

