from multiprocessing import cpu_count

# Get the number of CPU cores
num_cpus = cpu_count()

print(f"Number of CPU cores: {num_cpus}")