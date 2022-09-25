from pipelines.sr_pipeline import SRPipeline
from configs.sr_config_v1 import config


if __name__ == '__main__':
    pipeline = SRPipeline()
    pipeline.run(config)
