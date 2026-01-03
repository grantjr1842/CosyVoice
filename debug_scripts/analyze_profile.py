import pstats

p = pstats.Stats("wetext.prof")
p.strip_dirs().sort_stats("cumtime").print_stats(20)
p.sort_stats("tottime").print_stats(20)
