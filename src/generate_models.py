from src.modules.model import generate_model

versions = [
    {"current_version": "2_0_0", "previous_version": None},
    {"current_version": "2_1_0", "previous_version": "2_0_0"},
    {"current_version": "2_2_0", "previous_version": "2_1_0"},
    {"current_version": "2_3_0", "previous_version": "2_2_0"},
    {"current_version": "3_0_0", "previous_version": "2_3_0"},
    {"current_version": "3_1_0", "previous_version": "3_0_0"},
]

for version in versions:
    generate_model(version["current_version"], version["previous_version"])