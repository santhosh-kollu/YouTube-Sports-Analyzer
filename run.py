import sys
from src.app import app
from src.cli import run_pipeline

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        import argparse
        parser = argparse.ArgumentParser(description="YouTube Sports Commentary Analyzer")
        parser.add_argument("mode", help="Mode to run (cli or web)")
        parser.add_argument("--url", help="YouTube video URL")
        parser.add_argument("--model", default="base", help="Whisper model size (tiny/base/small/medium)")
        parser.add_argument("--output", default=".", help="Output directory for the report .txt file")
        parser.add_argument("--demo", action="store_true", help="Runs with built-in sample commentary")
        args = parser.parse_args()
        
        if not args.url and not args.demo:
            parser.print_help()
            sys.exit(1)
            
        run_pipeline(args.url, args.model, args.output, args.demo)
    else:
        # Run Flask web app by default (Hugging Face Spaces uses port 7860)
        app.run(host="0.0.0.0", port=7860, debug=False)
