# same as main.py, but the compare step uses every CPU core (compare_parallel.py)
import compare
import compare_parallel
import main

if __name__ == "__main__":
    compare.score_chips = compare_parallel.score_chips
    main.main()
