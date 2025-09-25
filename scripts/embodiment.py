from gr00t.experiment.data_config import load_data_config
cfg = load_data_config("agibot_genie1")
mc = cfg.modality_config()

print("Video inputs:", getattr(mc["video"], "inputs", []))
print("State inputs:", getattr(mc.get("state", None), "inputs", []))
print("Action outputs:", mc["action"].names)
print("Delta indices (video):", mc["video"].delta_indices)
if "state" in mc:
    print("Delta indices (state):", mc["state"].delta_indices)
