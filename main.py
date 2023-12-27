from uagents import Bureau
from agents.data_fetcher import DataFetcher
from agents.analyzer_agent import Analyzer

bureau = Bureau()
bureau.add(DataFetcher)
bureau.add(Analyzer)

if __name__ == '__main__':
    try:
        # Run the agent bureau
        
        bureau.run()
    except KeyboardInterrupt:
        # Handle a KeyboardInterrupt (e.g., when the user presses Ctrl+C)

        print("Process Interrupted.")