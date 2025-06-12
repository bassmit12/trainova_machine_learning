"""
Trainova ML - Main CLI Module
"""
import argparse
import sys
from typing import List, Optional

from .commands import CommandHandler

def setup_parser():
    """
    Setup and return the argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Trainova Machine Learning - Workout Prediction System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Pretrain command
    pretrain_parser = subparsers.add_parser("pretrain", help="Pretrain the model with existing data")
    pretrain_parser.add_argument("--generate-mock", action="store_true", help="Generate mock data for pretraining")
    pretrain_parser.add_argument("--samples", type=int, default=1000, help="Number of mock data samples to generate")
    pretrain_parser.add_argument("--exercises", type=str, help="Comma-separated list of exercises to generate data for")
    pretrain_parser.add_argument("--import-file", type=str, help="Import data from a CSV file for pretraining")
    pretrain_parser.add_argument("--model-type", type=str, choices=["random_forest", "neural_network", "lstm"], 
                              help="Type of model to use for training")
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect workout data interactively")
    collect_parser.add_argument("--exercise", type=str, help="Specify the exercise type")
    collect_parser.add_argument("--pretraining", action="store_true", help="Save as pretraining data")
    
    # Interactive training command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive training session")
    interactive_parser.add_argument("--exercise", type=str, help="Specify the exercise type")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make a weight prediction")
    predict_parser.add_argument("--exercise", type=str, help="Specify the exercise type")
    predict_parser.add_argument("--debug", action="store_true", help="Print debug information")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset model data")
    reset_parser.add_argument("--type", type=str, choices=["all", "model", "feedback"], 
                           help="Type of data to reset (default: all)")
    reset_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export training data")
    export_parser.add_argument("--file", type=str, help="File path to export to")
    export_parser.add_argument("--exclude-pretraining", action="store_true", help="Exclude pretraining data from export")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import workout data")
    import_parser.add_argument("file", type=str, help="File path to import from")
    import_parser.add_argument("--pretraining", action="store_true", help="Save as pretraining data")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    evaluate_parser.add_argument("--test-file", type=str, help="Test data file path")
    
    # Switch model command
    switch_model_parser = subparsers.add_parser("switch-model", help="Switch model type")
    switch_model_parser.add_argument("--model-type", type=str, choices=["random_forest", "neural_network", "lstm"], 
                                   help="Type of model to switch to")
    
    # Chain command
    chain_parser = subparsers.add_parser("chain", help="Execute multiple commands in sequence")
    chain_parser.add_argument("commands", nargs="+", help="Commands to execute in sequence")
    
    return parser

def execute_command(handler, command, args):
    """
    Execute a single command.
    
    Args:
        handler: CommandHandler instance
        command: Command name
        args: Command arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        if command == "pretrain":
            handler.handle_pretrain(args)
        elif command == "collect":
            handler.handle_collect(args)
        elif command == "interactive":
            handler.handle_interactive_training(args)
        elif command == "predict":
            handler.handle_predict(args)
        elif command == "reset":
            handler.handle_reset(args)
        elif command == "export":
            handler.handle_export(args)
        elif command == "import":
            handler.handle_import(args)
        elif command == "evaluate":
            handler.handle_evaluate(args)
        elif command == "switch-model":
            handler.handle_switch_model(args)
        else:
            print(f"Unknown command: {command}")
            return 1
        return 0
    except Exception as e:
        print(f"Error executing {command}: {e}")
        return 1

def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the Trainova ML CLI.
    
    Args:
        args: Command line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if args is None:
        args = sys.argv[1:]
    
    # Check if we're using the sequential syntax
    if len(args) >= 2 and args[0] != "chain" and "--" not in args[0]:
        # Find potential command boundaries
        command_boundaries = []
        current_commands = []
        
        for i, arg in enumerate(args):
            if arg in ["pretrain", "collect", "interactive", "predict", 
                      "reset", "export", "import", "evaluate", "switch-model"]:
                if current_commands:
                    command_boundaries.append((current_commands[0], current_commands[1:]))
                    current_commands = [arg]
                else:
                    current_commands = [arg]
            else:
                current_commands.append(arg)
        
        if current_commands:
            command_boundaries.append((current_commands[0], current_commands[1:]))
        
        if len(command_boundaries) > 1:
            # Modify args to use the chain command
            args = ["chain"] + [cmd for cmd, _ in command_boundaries]
    
    # Parse arguments
    parser = setup_parser()
    parsed_args = parser.parse_args(args)
    
    # If no command is provided, show help
    if not parsed_args.command:
        parser.print_help()
        return 1
    
    # Initialize the command handler
    handler = CommandHandler()
    
    # Handle chain command specially
    if parsed_args.command == "chain":
        print(f"Executing commands in sequence: {', '.join(parsed_args.commands)}")
        
        exit_code = 0
        for cmd in parsed_args.commands:
            print(f"\nExecuting command: {cmd}")
            # Parse arguments for this command
            try:
                cmd_args = parser.parse_args([cmd])
                cmd_result = execute_command(handler, cmd, cmd_args)
                if cmd_result != 0:
                    exit_code = cmd_result
                    print(f"Command {cmd} failed with exit code {cmd_result}")
                    break
            except SystemExit:
                print(f"Error: Invalid arguments for command {cmd}")
                exit_code = 1
                break
        
        return exit_code
    
    # Dispatch to the appropriate handler for single command
    try:
        return execute_command(handler, parsed_args.command, parsed_args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130

if __name__ == "__main__":
    sys.exit(main())