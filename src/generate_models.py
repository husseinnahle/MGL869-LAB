from src.modules.model import generate_model

versions = ["2_0_0", "2_1_0", "2_2_0", "2_3_0", "3_0_0", "3_1_0"]

for version in versions:
    generate_model(version, recalculate_models=False)
