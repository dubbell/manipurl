import cProfile
import os
import datetime
import functools


def profile(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        do_profiling = os.environ.get('MANIPURL_PROFILING', False)

        if do_profiling:
            profiler = cProfile.Profile()
            profiler.enable()

        run_name = func(*args, **kwargs)

        if run_name is None:
            run_name = "noname_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if do_profiling:
            profiler.disable()
            os.makedirs("data/profiling", exist_ok=True)
            profiler.dump_stats(f"data/profiling/{run_name}.prof")
        
        return run_name
    
    return wrapper