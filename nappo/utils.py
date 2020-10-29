import os
import shutil

def cleanup_log_dir(log_dir):
    """
    Create log directory and remove old files.

    Parameters
    ----------
    log_dir : str
        Path to log directory.
    """
    try:
        os.makedirs(log_dir)
    except OSError:
        if os.path.isdir(os.path.join(log_dir, "train")):
            shutil.rmtree(os.path.join(log_dir, "train"))
        if os.path.isdir(os.path.join(log_dir, "test")):
            shutil.rmtree(os.path.join(log_dir, "test"))
        if os.path.isdir(os.path.join(log_dir, "tensorboard_logs")):
            shutil.rmtree(os.path.join(log_dir, "tensorboard_logs"))

