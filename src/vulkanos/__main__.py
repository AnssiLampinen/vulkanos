from .cli import main

# Guard needed: DataLoader workers re-import this module on spawn platforms.
if __name__ == "__main__":
    main()
