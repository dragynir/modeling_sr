from pipelines.sr_pipeline import SRPipeline
from configs.benchmark_config_ import config

# todo add validation and inference pipelines
# TODO надо сделать общий интерфейс для train, чтобы был только один evaluate
if __name__ == '__main__':
    pipeline = SRPipeline()
    pipeline.run(config)
