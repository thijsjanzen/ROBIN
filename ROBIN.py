from scripts import dependencies
from scripts import main

if __name__ == "__main__":
    # Well behaved unix programs exits with 0 on success...
    sys.exit(main.main())