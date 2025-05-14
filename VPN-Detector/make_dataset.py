# save as make_dataset.py  (đặt ở thư mục gốc VPN-Detector)
import pandas as pd
import os

root = "clean_data"
vpn  = pd.read_csv(os.path.join(root, "vpn15s_cleaned.csv"))
nov  = pd.read_csv(os.path.join(root, "novpn15s_cleaned.csv"))

vpn ["label"] = 1      # 1 = VPN
nov ["label"] = 0      # 0 = Non‑VPN

df = pd.concat([vpn, nov], ignore_index=True)
df.to_csv(os.path.join(root, "vpn_15s_labeled.csv"), index=False)
print("✅ Saved clean_data/vpn_15s_labeled.csv  →  rows:", len(df))
