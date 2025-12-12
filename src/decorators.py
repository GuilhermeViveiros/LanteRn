import time

# decorator that measures the time of the function
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken of {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper