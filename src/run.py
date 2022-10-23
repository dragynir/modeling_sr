from pipelines.sr_pipeline import SRPipeline
from configs import config


# CUDA_VISIBLE_DEVICES="0" python run.py
# CUDA_VISIBLE_DEVICES="0" nohup python run.py &
if __name__ == '__main__':
    pipeline = SRPipeline()
    pipeline.run(config)
