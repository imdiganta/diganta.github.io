import time
from concurrent.futures import ThreadPoolExecutor

# Define some example functions
def function_one():
    print("Function One is starting...")
    time.sleep(2)
    print("Function One is done.")

def function_two():
    print("Function Two is starting...")
    time.sleep(3)
    print("Function Two is done.")

def function_three():
    print("Function Three is starting...")
    time.sleep(1)
    print("Function Three is done.")

# Main block to execute the functions concurrently
if __name__ == "__main__":
    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Submit the functions to be run
        futures = [
            executor.submit(function_one),
            executor.submit(function_two),
            executor.submit(function_three)
        ]
        
        # Wait for the results (optional)
        for future in futures:
            future.result()  # This will raise exceptions if any occurred

    print("All functions have completed.")
