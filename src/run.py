from pipelines.sr_pipeline import SRPipeline
from configs.sr_config_v1 import config

# todo add validation and inference pipelines
if __name__ == '__main__':
    pipeline = SRPipeline()
    pipeline.run(config)
