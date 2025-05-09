import pynvml

class Logger:
    """
    A utility class for logging GPU stats, debug messages, and metrics.
    """

    def __init__(self, enable_wandb=False, debug=False):
        """
        Initialize the logger.

        Args:
            enable_wandb (bool): Whether to enable WandB logging.
            debug (bool): Whether to enable debug printing.
        """
        self.enable_wandb = enable_wandb
        self.debug = debug

        # Initialize WandB if enabled
        if self.enable_wandb:
            import wandb
            wandb.init(project="streaminator", config={})
        self.tables = {}

        # Initialize NVML for GPU stats logging
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def Table(self, columns):
        """
        Create or retrieve a WandB table with the specified columns.
        If WandB is not enabled, return a mock table.
        """
        if not self.enable_wandb:
            return MockTable(columns)
        import wandb
        table_key = tuple(columns)
        if table_key not in self.tables:
            self.tables[table_key] = wandb.Table(columns=columns)
        return self.tables[table_key]

    def log_gpu_stats(self, step, active_seqs, stream_width):
        """
        Log GPU stats, including memory usage and utilization.

        Args:
            step (int): Current generation step.
            active_seqs (list): List of active sequences.
            stream_width (int): Total stream width.
        """
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            stream_utilization = sum(not seq.is_finished() for seq in active_seqs) / stream_width * 100

            log_data = {
                "gpu SM utilization (%)": util.gpu,
                "gpu memory (MB)": mem_info.used / 1024**2,
                "gpu memory usage (%)": mem_info.used / mem_info.total * 100,
                "generation step": step,
                "stream utilization (%)": stream_utilization,
            }

            if self.enable_wandb:
                import wandb
                wandb.log(log_data)

            if self.debug:
                print(f"[DEBUG] GPU Stats: {log_data}")

        except pynvml.NVMLError as e:
            if self.debug:
                print(f"[DEBUG] Failed to log GPU stats: {e}")

    def log(self, data):
        """
        Log data to WandB if enabled.
        """
        if self.enable_wandb:
            import wandb
            wandb.log(data)

    def debug_print(self, msg):
        """
        Print debug messages if debug mode is enabled.

        Args:
            msg (str): The debug message to print.
        """
        if self.debug:
            print(f"[DEBUG] {msg}")

    def log_metrics(self, metrics):
        """
        Log custom metrics to WandB.

        Args:
            metrics (dict): A dictionary of metrics to log.
        """
        if self.enable_wandb:
            import wandb
            wandb.log(metrics)

    def shutdown(self):
        """
        Shutdown the logger and clean up resources.
        """
        if self.enable_wandb:
            import wandb
            wandb.finish()
        pynvml.nvmlShutdown()


class MockTable:
    """
    A mock table to use when WandB is not enabled.
    """
    def __init__(self, columns):
        self.columns = columns
        self.data = []

    def add_data(self, *args):
        """
        Mock method to add data to the table.
        """
        self.data.append(args)

    def __repr__(self):
        return f"MockTable(columns={self.columns}, data={self.data})"